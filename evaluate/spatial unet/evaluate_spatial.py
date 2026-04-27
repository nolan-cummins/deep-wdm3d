import os
import random
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import label, find_contours
import cv2

# ==========================================
# 1. MODEL ARCHITECTURE
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
# 2. EVALUATION PIPELINE
# ==========================================
def normalize(data):
    data = np.nan_to_num(data)
    p1, p99 = np.percentile(data, [1, 99])
    data = np.clip(data, p1, p99)
    return (data - data.min()) / (data.max() - data.min() + 1e-8)

def get_alpha_mask(heatmap, threshold=0.5):
    binary = (heatmap > threshold).astype(np.uint8)
    labeled, num_features = label(binary, return_num=True)
    if num_features == 0: return np.zeros_like(binary)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(np.uint8)

def evaluate_random_sample(thresh_dir="../threshold_dataset", test_dir="../testing data", weights="spatial_unet_best.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet3D().to(device)
    state_dict = torch.load(weights, map_location=device, weights_only=True)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    t_dir = Path(thresh_dir)
    test_d = Path(test_dir)
    
    val_subjects = []
    if not Path('dataset_splits.csv').exists():
        raise FileNotFoundError("dataset_splits.csv not found! Make sure you run the training script first.")

    with open('dataset_splits.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'val':
                val_subjects.append(row['subj_name'])

    if not val_subjects:
        raise ValueError("No validation subjects found in dataset_splits.csv.")

    subj_name = random.choice(val_subjects)
    print(f"Selected Unseen Validation Subject: {subj_name}")

    subj_dir = t_dir / subj_name
    orig_matches = list(test_d.rglob(f"{subj_name}.nii.gz"))
    if not orig_matches:
        raise FileNotFoundError(f"Original scan for {subj_name} not found in {test_dir}")

    orig_path = orig_matches[0]
    label_path = Path(str(orig_path).replace('imagesTr', 'labelsTr').replace('imagesTs', 'labelsTs'))

    if not label_path.exists():
        raise FileNotFoundError(f"Ground truth label for {subj_name} not found.")
    
    # Grab the original affine matrix for 3D saving
    orig_nifti = nib.load(orig_path)
    orig_affine = orig_nifti.affine
    
    def load_vol(p, ch=None):
        v = nib.as_closest_canonical(nib.load(p)).get_fdata()
        if v.ndim > 3: v = v[..., ch if ch is not None else 1]
        return v, normalize(v)

    orig_raw, orig_norm = load_vol(orig_path, ch=1)
    _, t1_norm = load_vol(subj_dir / f"t_500_{subj_name}.nii.gz")
    _, t2_norm = load_vol(subj_dir / f"t_700_{subj_name}.nii.gz")
    
    gt_raw = nib.as_closest_canonical(nib.load(label_path)).get_fdata()
    if gt_raw.ndim > 3: gt_raw = gt_raw[..., 0]

    # --- INFERENCE ---
    x_in = torch.from_numpy(np.stack([orig_norm, t1_norm, t2_norm])).float().unsqueeze(0).to(device)
    
    print("Running 3D U-Net Inference...")
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x_in)
        probs = torch.sigmoid(logits) 
        
    orig_shape = orig_norm.shape
    probs_upscaled = F.interpolate(probs, size=orig_shape, mode='trilinear', align_corners=False)
    
    heatmap_3d = probs_upscaled.squeeze().float().cpu().numpy() 
    
    # Raw thresholded predictions
    pred_mask_3d = (heatmap_3d > 0.5).astype(np.uint8)

    # --- SAVE 3D MASK ---
    pred_nifti = nib.Nifti1Image(pred_mask_3d, orig_affine)
    out_mask_path = f"eval_spatial_mask_{subj_name}.nii.gz"
    nib.save(pred_nifti, out_mask_path)
    print(f"Saved 3D prediction mask to {out_mask_path}")

    # --- PLOTTING ---
    best_slice_idx = int(np.argmax(np.sum(gt_raw > 0, axis=(0, 1))))

    s_orig = orig_norm[:, :, best_slice_idx].T
    s_t1 = t1_norm[:, :, best_slice_idx].T
    s_t2 = t2_norm[:, :, best_slice_idx].T
    s_heat = heatmap_3d[:, :, best_slice_idx].T
    s_mask = pred_mask_3d[:, :, best_slice_idx].T
    s_gt = (gt_raw[:, :, best_slice_idx].T > 0).astype(np.uint8)

    s_orig_vis = np.clip(s_orig, 0, 1)
    s_orig_vis = (s_orig_vis * 255).astype(np.uint8)
    overlay = cv2.cvtColor(s_orig_vis, cv2.COLOR_GRAY2RGB)
    
    red_idx = s_mask > 0
    overlay[red_idx] = overlay[red_idx] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    contours = find_contours(s_gt, level=0.5)
    for contour in contours:
        plt_contour = np.array([contour[:, 1], contour[:, 0]]).T.astype(np.int32)
        cv2.polylines(overlay, [plt_contour], isClosed=True, color=(0, 255, 0), thickness=1)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    axes[0].imshow(s_orig, cmap='gray', origin='lower')
    axes[0].set_title(f"Original (Slice {best_slice_idx})")
    
    axes[1].imshow(s_t1, cmap='gray', origin='lower')
    axes[1].set_title("t=500 (Baseline)")
    
    axes[2].imshow(s_t2, cmap='gray', origin='lower')
    axes[2].set_title("t=700 (Hallucinated)")
    
    im3 = axes[3].imshow(s_heat, cmap='magma', origin='lower', vmin=0, vmax=1)
    axes[3].set_title("U-Net Probability Heatmap")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    axes[4].imshow(overlay, origin='lower')
    axes[4].set_title("Raw Predicted Mask\nRed=Pred, Green=GT")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"eval_spatial_{subj_name}.png", dpi=200, bbox_inches='tight')
    print(f"Saved evaluation plot to eval_spatial_{subj_name}.png")

if __name__ == '__main__':
    evaluate_random_sample()