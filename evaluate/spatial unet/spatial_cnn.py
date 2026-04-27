import os
import csv
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage.measure import label

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# 1. ARCHITECTURE: 3D U-Net (Returning Logits)
# ==========================================
class UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c), nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c), nn.ReLU(inplace=True),
                nn.Dropout3d(p=0.2)
            )

        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = conv_block(128, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.final = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, size=(128, 128, 128), mode='trilinear', align_corners=False)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)

# ==========================================
# 2. LOSS FUNCTIONS (Autocast Safe)
# ==========================================
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        probs = torch.sigmoid(inputs)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(probs.sum() + targets.sum() + smooth)  
        return BCE + dice_loss

# ==========================================
# 3. DATASET PIPELINE
# ==========================================
class SpatialBraTSDataset(Dataset):
    def __init__(self, threshold_dir, brats_dir, t1=500, t2=700):
        self.thresh_dir = Path(threshold_dir)
        self.brats_dir = Path(brats_dir)
        self.samples = []

        # Reverted back to unsorted iterdir to replicate the magic Run 1 split
        for subj_dir in self.thresh_dir.iterdir():
            if not subj_dir.is_dir(): continue
            name = f"{subj_dir.name}.nii.gz"
            lbl = self.brats_dir / "labelsTr" / name
            if lbl.exists():
                self.samples.append({
                    'subj_name': subj_dir.name,
                    'orig': self.brats_dir / "imagesTr" / name,
                    't1': subj_dir / f"t_{t1}_{name}",
                    't2': subj_dir / f"t_{t2}_{name}",
                    'label': lbl
                })

    def normalize(self, data):
        data = np.nan_to_num(data)
        p1, p99 = np.percentile(data, [1, 99])
        data = np.clip(data, p1, p99)
        return (data - data.min()) / (data.max() - data.min() + 1e-8)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        def load_vol(p, ch=None):
            v = nib.as_closest_canonical(nib.load(p)).get_fdata()
            if v.ndim > 3: v = v[..., ch if ch is not None else 1]
            return self.normalize(v)

        orig = load_vol(s['orig'], ch=1)
        t1 = load_vol(s['t1'])
        t2 = load_vol(s['t2'])
        
        label_vol = nib.as_closest_canonical(nib.load(s['label'])).get_fdata()
        if label_vol.ndim > 3: label_vol = label_vol[..., 0]
        
        x = torch.from_numpy(np.stack([orig, t1, t2])).float()
        y = torch.from_numpy(label_vol).unsqueeze(0).float()
        y = F.interpolate(y.unsqueeze(0), size=(128, 128, 128), mode='nearest').squeeze(0)
        
        return x, y > 0

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    full_dataset = SpatialBraTSDataset("../threshold_dataset", "../testing data")
    
    # 1. Revert to standard PyTorch random_split (Run 1 logic)
    torch.manual_seed(42)
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # 2. Export the ledger so you know exactly what the OS and PyTorch did
    if local_rank == 0 and not Path('dataset_splits.csv').exists():
        with open('dataset_splits.csv', 'w', newline='') as f:
            split_writer = csv.writer(f)
            split_writer.writerow(['subj_name', 'split'])
            for idx in train_ds.indices:
                split_writer.writerow([full_dataset.samples[idx]['subj_name'], 'train'])
            for idx in val_ds.indices:
                split_writer.writerow([full_dataset.samples[idx]['subj_name'], 'val'])
        print(f">> Immutable dataset split ledger saved to dataset_splits.csv ({train_size} Train | {val_size} Val)")

    train_loader = DataLoader(train_ds, batch_size=1, sampler=DistributedSampler(train_ds), num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, sampler=DistributedSampler(val_ds, shuffle=False), num_workers=4)

    model = UNet3D().to(device)
    model = DDP(model, device_ids=[local_rank])
    
    start_epoch, best_val, last_lr = 0, float('inf'), 1e-4

    if Path('spatial_unet_best.pt').exists():
        map_loc = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        model.module.load_state_dict(torch.load('spatial_unet_best.pt', map_location=map_loc, weights_only=True))
        
        if Path('spatial_history.csv').exists():
            with open('spatial_history.csv', 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if len(rows) > 0:
                    start_epoch = int(rows[-1]['epoch']) + 1
                    best_val = min(float(r['val_loss']) for r in rows)
                    last_lr = float(rows[-1]['lr'])
        if local_rank == 0:
            print(f">> Resuming from Epoch {start_epoch} | Best Val: {best_val:.4f} | LR: {last_lr:.2e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=last_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scheduler.best = best_val 

    criterion = DiceBCELoss()
    scaler = torch.amp.GradScaler('cuda')

    if local_rank == 0:
        if start_epoch > 0:
            csv_file = open('spatial_history.csv', 'a', newline='')
            writer = csv.writer(csv_file)
        else:
            csv_file = open('spatial_history.csv', 'w', newline='')
            writer = csv.writer(csv_file)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])

    patience, stall_count = 12, 0

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        train_running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(local_rank != 0))
        for x, y in pbar:
            x, y = x.to(device), y.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(x)
                loss = criterion(pred, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_running_loss += loss.item()
            if local_rank == 0: pbar.set_postfix({'loss': loss.item()})

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).float()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    val_running_loss += criterion(model(x), y).item()
        
        stats = torch.tensor([train_running_loss / len(train_loader), val_running_loss / len(val_loader)]).to(device)
        dist.all_reduce(stats, op=dist.ReduceOp.AVG)
        
        if local_rank == 0:
            tr_l, vl_l = stats[0].item(), stats[1].item()
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"--- Epoch {epoch} | Train: {tr_l:.4f} | Val: {vl_l:.4f} | LR: {curr_lr:.2e} ---")
            writer.writerow([epoch, tr_l, vl_l, curr_lr]); csv_file.flush()
            
            scheduler.step(vl_l)
            
            if vl_l < best_val:
                best_val = vl_l
                stall_count = 0
                torch.save(model.module.state_dict(), 'spatial_unet_best.pt')
                print(">> Saved new best model!")
            else:
                stall_count += 1
                if stall_count >= patience: 
                    print("Early stopping triggered. Training halted.")
                    break

    if local_rank == 0: csv_file.close()

if __name__ == "__main__":
    train()