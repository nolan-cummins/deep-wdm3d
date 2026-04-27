import json
import csv
import copy
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

from model import WaveletTransform3D, GaussianDiffusion, UnconditionalWavUNet

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class NiftiDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, size=256):
        self.files = [f for f in Path(data_dir).rglob('*.nii.gz') if not f.name.startswith('._')]
        self.size = size

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        nifti = nib.as_closest_canonical(nib.load(str(self.files[idx])))
        img = nifti.get_fdata()
        img = np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        brain_mask_np = img > 1e-4 
        if brain_mask_np.any():
            p1, p99 = np.percentile(img[brain_mask_np], [1, 99])
            img = np.clip(img, p1, p99)
            img_min, img_max = img.min(), img.max()
            img = 2.0 * (img - img_min) / (img_max - img_min + 1e-8) - 1.0
            img[brain_mask_np == 0] = -1.0
        else:
            img.fill(-1.0)

        # 1. Center crop if larger than 256 (rare for BraTS, but safe)
        D, H, W = img.shape
        d_s = max(0, (D - self.size) // 2)
        h_s = max(0, (H - self.size) // 2)
        w_s = max(0, (W - self.size) // 2)
        img = img[d_s:d_s + self.size, h_s:h_s + self.size, w_s:w_s + self.size]

        # 2. Pad to exactly 256x256x256 with background value (-1.0)
        D, H, W = img.shape
        pad_d_before = (self.size - D) // 2
        pad_d_after = self.size - D - pad_d_before
        pad_h_before = (self.size - H) // 2
        pad_h_after = self.size - H - pad_h_before
        pad_w_before = (self.size - W) // 2
        pad_w_after = self.size - W - pad_w_before

        img_padded = np.pad(
            img, 
            ((pad_d_before, pad_d_after), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)), 
            mode='constant', 
            constant_values=-1.0
        )
        
        return torch.from_numpy(img_padded).float().unsqueeze(0)

def get_or_compute_dwt_stats(dataloader, wavelet, device, num_batches=30):
    stats_file = Path('dwt_stds_256.pt')
    if stats_file.exists():
        stats = torch.load(stats_file, map_location=device, weights_only=True)
        return stats['means'], stats['stds']
    
    means, variances = [], []
    wavelet.set_stats(torch.zeros(8, device=device), torch.ones(8, device=device)) 
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            dwt_out = wavelet.dwt(batch.to(device))
            means.append(torch.mean(dwt_out, dim=(0, 2, 3, 4)))
            variances.append(torch.var(dwt_out, dim=(0, 2, 3, 4)))
            
    channel_means = torch.stack(means).mean(dim=0)
    channel_stds = torch.sqrt(torch.stack(variances).mean(dim=0)).clamp(min=1e-6) 
    
    torch.save({'means': channel_means, 'stds': channel_stds}, stats_file)
    return channel_means, channel_stds

def train():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    with open('config.json') as f: cfg = json.load(f)
    
    model = UnconditionalWavUNet(base_dim=cfg['model_channels']).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    ema_model = copy.deepcopy(model.module)
    for param in ema_model.parameters(): param.requires_grad = False

    wavelet = WaveletTransform3D().to(device)
    diffusion = GaussianDiffusion(steps=cfg['diffusion_steps'])
    
    dataset = NiftiDataset(cfg['data_dir'], size=cfg['image_size'])
    sampler = DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], sampler=sampler, num_workers=8, pin_memory=True)

    if local_rank == 0:
        dwt_means, dwt_stds = get_or_compute_dwt_stats(dataloader, wavelet, device)
    dist.barrier()
    
    if local_rank != 0:
        stats = torch.load('dwt_stds_256.pt', map_location=device, weights_only=True)
        dwt_means, dwt_stds = stats['means'], stats['stds']

    wavelet.set_stats(dwt_means, dwt_stds)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    
    total_steps = cfg['epochs'] * len(dataloader)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=cfg['lr'], 
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    best_loss = float('inf')
    start_epoch = 0

    if local_rank == 0: 
        print("Starting wdm-3d 256^3 training loop...")
        epoch_log = open('loss_epochs.csv', 'a', newline='')
        epoch_writer = csv.writer(epoch_log)
        if os.stat('loss_epochs.csv').st_size == 0:
            epoch_writer.writerow(['epoch', 'avg_mse_loss'])

    try:
        for epoch in range(start_epoch, cfg['epochs']):
            sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0.0
            step_losses = [] 
            
            iterable = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['epochs']-1}") if local_rank == 0 else dataloader
            
            for step, batch in enumerate(iterable):
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):                        
                    x_dwt = wavelet.dwt(batch.bfloat16())
                    t = torch.randint(0, cfg['diffusion_steps'], (batch.size(0),), device=device).long()
                    
                    noise_dwt = torch.randn_like(x_dwt)
                    x_noisy = diffusion.q_sample(x_dwt, t, noise_dwt)
                    predicted_noise = model(x_noisy, t)
                    
                    loss = F.mse_loss(predicted_noise.float(), noise_dwt.float())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step() 
                
                with torch.no_grad():
                    decay = min(0.9999, (1 + epoch*len(dataloader) + step) / (10 + epoch*len(dataloader) + step))
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.lerp_(p, 1 - decay)

                loss_sync = loss.detach().clone()
                dist.all_reduce(loss_sync, op=dist.ReduceOp.AVG)
                val = loss_sync.item()
                
                epoch_loss += val
                step_losses.append(val)
                
                if local_rank == 0: 
                    run_avg = np.mean(step_losses)
                    current_lr = scheduler.get_last_lr()[0]
                    iterable.set_postfix(mse=f"{val:.4f}", avg=f"{run_avg:.4f}", lr=f"{current_lr:.2e}")

            avg_loss = epoch_loss / len(dataloader)
            
            if local_rank == 0:
                print(f"Epoch {epoch} Complete | Avg MSE: {avg_loss:.4f}")
                epoch_writer.writerow([epoch, avg_loss])
                epoch_log.flush()
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(ema_model.state_dict(), 'best_ema_weights_256.pt')
                    torch.save(model.state_dict(), 'best_weights_256.pt')
                    print("  -> New best loss! Saved weights.")
                    
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'ema_model_state': ema_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_loss': best_loss
                }, 'latest_checkpoint_256.pt')

    except KeyboardInterrupt:
        if local_rank == 0:
            print("\nTraining interrupted!")
    finally:
        if local_rank == 0:
            epoch_log.close()

if __name__ == '__main__':
    train()