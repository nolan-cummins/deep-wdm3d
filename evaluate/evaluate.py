import os
import time
import argparse
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import find_contours
import cv2

from model import WaveletTransform3D, GaussianDiffusion, UnconditionalWavUNet

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# 1. 3D U-NET ARCHITECTURE
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
# 2. EVALUATION HELPER FUNCTIONS
# ==========================================
def normalize(data):
    data = np.nan_to_num(data)
    p1, p99 = np.percentile(data, [1, 99])
    data = np.clip(data, p1, p99)
    return (data - data.min()) / (data.max() - data.min() + 1e-8)

def compute_dice(pred, gt):
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)

def save_plot(image, filename, cmap='gray', overlay=None, is_heatmap=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    if overlay is not None:
        ax.imshow(overlay, origin='lower')
    else:
        im = ax.imshow(image, cmap=cmap, origin='lower')
        if is_heatmap:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()

def export_video(orig_vol, mask_vol, gt_vol, axis, filename, fps=24, has_gt=True):
    frames = []
    
    if axis == 0: valid_mask = np.any(orig_vol > 1e-4, axis=(1, 2))
    elif axis == 1: valid_mask = np.any(orig_vol > 1e-4, axis=(0, 2))
    else: valid_mask = np.any(orig_vol > 1e-4, axis=(0, 1))
        
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0: return
        
    start_idx, end_idx = valid_indices[0], valid_indices[-1]
    
    for i in range(start_idx, end_idx + 1):
        if axis == 0:
            orig_s, mask_s, gt_s = orig_vol[i, :, :].T, mask_vol[i, :, :].T, gt_vol[i, :, :].T
        elif axis == 1:
            orig_s, mask_s, gt_s = orig_vol[:, i, :].T, mask_vol[:, i, :].T, gt_vol[:, i, :].T
        else:
            orig_s, mask_s, gt_s = orig_vol[:, :, i].T, mask_vol[:, :, i].T, gt_vol[:, :, i].T

        orig_vis = (np.clip(orig_s, 0, 1) * 255).astype(np.uint8)
        frame = cv2.cvtColor(orig_vis, cv2.COLOR_GRAY2BGR)
        
        frame[mask_s > 0] = frame[mask_s > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
        
        if has_gt:
            for contour in find_contours(gt_s, level=0.5):
                plt_contour = np.array([contour[:, 1], contour[:, 0]]).T.astype(np.int32)
                cv2.polylines(frame, [plt_contour], isClosed=True, color=(0, 255, 0), thickness=1)
            
        h, w, _ = frame.shape
        size = max(h, w)
        square_frame = np.zeros((size, size, 3), dtype=np.uint8)
        y_off, x_off = (size - h) // 2, (size - w) // 2
        square_frame[y_off:y_off+h, x_off:x_off+w] = frame
        frames.append(square_frame)

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(str(filename), 0, fps, (width, height))
    for f in frames: out.write(f)
    out.release()

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def process_single_scan(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_path = Path(args.input)
    subj_name = input_path.name.replace('.nii.gz', '')
    
    out_dir = Path(args.output_dir) / subj_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.config) as f: cfg = json.load(f)
    size = cfg.get('image_size', 256)
    
    timing = {}
    
    print(f"--- Initializing Models ---")
    # Load Diffusion
    diff_model = UnconditionalWavUNet(base_dim=cfg['model_channels']).to(device)
    diff_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.diff_weights, map_location=device, weights_only=True).items()})
    diff_model.eval()

    wavelet = WaveletTransform3D().to(device)
    stats = torch.load('dwt_stds_256.pt', map_location=device, weights_only=True)
    wavelet.set_stats(stats['means'], stats['stds'])
    diffusion = GaussianDiffusion(steps=cfg['diffusion_steps'])

    # Load Spatial U-Net
    unet_model = UNet3D().to(device)
    unet_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.unet_weights, map_location=device, weights_only=True).items()})
    unet_model.eval()

    # 1. Loading & Preprocessing
    t_start = time.time()
    img_nifti = nib.as_closest_canonical(nib.load(str(input_path)))
    raw_data = img_nifti.get_fdata()
    
    if raw_data.ndim == 4:
        raw_data = raw_data[:, :, :, 1]
    else:
        raw_data = np.transpose(raw_data, (2, 0, 1))
        raw_data = raw_data[:, :, ::-1]

    orig_img = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    brain_mask = orig_img > 1e-4 
    if brain_mask.any():
        p1, p99 = np.percentile(orig_img[brain_mask], [1, 99])
        img_norm = np.clip(orig_img, p1, p99)
        img_min, img_max = img_norm.min(), img_norm.max()
        img_norm = 2.0 * (img_norm - img_min) / (img_max - img_min + 1e-8) - 1.0
        img_norm[~brain_mask] = -1.0
    else:
        img_norm = np.full(orig_img.shape, -1.0)

    D_orig, H_orig, W_orig = img_norm.shape
    d_s, h_s, w_s = max(0, (D_orig - size)//2), max(0, (H_orig - size)//2), max(0, (W_orig - size)//2)
    img_cropped = img_norm[d_s:d_s + size, h_s:h_s + size, w_s:w_s + size]
    
    D_c, H_c, W_c = img_cropped.shape
    pad_d_b, pad_d_a = (size - D_c) // 2, size - D_c - ((size - D_c) // 2)
    pad_h_b, pad_h_a = (size - H_c) // 2, size - H_c - ((size - H_c) // 2)
    pad_w_b, pad_w_a = (size - W_c) // 2, size - W_c - ((size - W_c) // 2)

    img_padded = np.pad(img_cropped, ((pad_d_b, pad_d_a), (pad_h_b, pad_h_a), (pad_w_b, pad_w_a)), mode='constant', constant_values=-1.0)
    patch_t = torch.tensor(img_padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Ground Truth Loading
    if "PED" in subj_name:
        possible_labels = list(input_path.parent.glob("*seg.nii.gz"))
        label_path = possible_labels[0] if possible_labels else input_path.parent / input_path.name.replace('t1n', 'seg')
    else:
        label_path = Path(str(input_path).replace('imagesTr', 'labelsTr').replace('imagesTs', 'labelsTs'))

    has_gt = False
    gt_mask_bool = np.zeros(img_norm.shape, dtype=np.uint8)
    
    if label_path.exists():
        try:
            gt_raw = nib.as_closest_canonical(nib.load(str(label_path))).get_fdata()
            if gt_raw.ndim == 4:
                gt_raw = gt_raw[..., 0]
            else:
                gt_raw = np.transpose(gt_raw, (2, 0, 1))
                gt_raw = gt_raw[:, :, ::-1]
                
            if gt_raw.shape == img_norm.shape:
                gt_mask_bool = (gt_raw > 0).astype(np.uint8)
                has_gt = True
            else:
                print(f"Warning: GT shape {gt_raw.shape} mismatches input shape {img_norm.shape}. Ignoring GT.")
        except Exception as e:
            print(f"Warning: Failed to process ground truth ({e}). Ignoring GT.")
    else:
        print("No ground truth label found. Skipping Dice score and contours.")
    
    orig_norm_01 = normalize(img_norm)
    timing['1_Data_Prep'] = time.time() - t_start

    # 2 & 3. Diffusion Step (Robust Caching)
    recons = {}
    noisy_vols = {}
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        # Calculate wavelets and cache for visualization
        dwt_orig = wavelet.dwt(patch_t.bfloat16())
        dwt_vis_cache = dwt_orig.squeeze(0).float().cpu().numpy()
        
        for t_val in [args.t1, args.t2]:
            t0 = time.time()
            recon_path = out_dir / f"t_{t_val}_{subj_name}.nii.gz"
            noisy_cache_path = out_dir / f"noisy_t_{t_val}_{subj_name}.npy"
            
            if recon_path.exists():
                print(f"--- Cached NIfTI found for t={t_val}, skipping DDIM generation ---")
                full_recon = nib.load(str(recon_path)).get_fdata()
                recons[t_val] = normalize(full_recon)
                timing[f'2_Diffusion_t{t_val}'] = 0.0
                
                # If NIfTI exists but the numpy plot cache doesn't, quickly regenerate just the noise
                if noisy_cache_path.exists():
                    noisy_vols[t_val] = np.load(str(noisy_cache_path))
                else:
                    t_eval = t_val - 1
                    noise = torch.randn_like(dwt_orig)
                    noisy_dwt = diffusion.q_sample(dwt_orig, t_eval, noise)
                    noisy_patch = wavelet.idwt(noisy_dwt.float()).squeeze().float().cpu().numpy()
                    noisy_vols[t_val] = noisy_patch[pad_d_b:size-pad_d_a, pad_h_b:size-pad_h_a, pad_w_b:size-pad_w_a]
                    np.save(str(noisy_cache_path), noisy_vols[t_val])
                continue

            print(f"--- Running full Diffusion steps for t={t_val} ---")
            t_eval = t_val - 1
            noise = torch.randn_like(dwt_orig)
            
            noisy_dwt = diffusion.q_sample(dwt_orig, t_eval, noise)
            
            noisy_patch = wavelet.idwt(noisy_dwt.float()).squeeze().float().cpu().numpy()
            noisy_vols[t_val] = noisy_patch[pad_d_b:size-pad_d_a, pad_h_b:size-pad_h_a, pad_w_b:size-pad_w_a]
            np.save(str(noisy_cache_path), noisy_vols[t_val])

            stride = max(10, int(t_eval / 6))
            healthy_dwt = diffusion.ddim_sample_loop(diff_model, noisy_dwt.shape, t_eval, noisy_dwt, device, strides=stride)
            
            recon_patch = wavelet.idwt(healthy_dwt.float()).squeeze().float().cpu().numpy()
            recon_patch = np.clip(recon_patch, -1.0, 1.0)
            recon_unpad = recon_patch[pad_d_b:size-pad_d_a, pad_h_b:size-pad_h_a, pad_w_b:size-pad_w_a]
            
            full_recon = np.copy(img_norm)
            full_recon[d_s:d_s + D_c, h_s:h_s + H_c, w_s:w_s + W_c] = recon_unpad
            
            nib.save(nib.Nifti1Image(full_recon, img_nifti.affine, img_nifti.header), recon_path)
            recons[t_val] = normalize(full_recon) 
            timing[f'2_Diffusion_t{t_val}'] = time.time() - t0

    # 4. U-Net Inference
    print(f"--- Running 3D U-Net Inference ---")
    t0 = time.time()
    x_in = torch.from_numpy(np.stack([orig_norm_01, recons[args.t1], recons[args.t2]])).float().unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        probs = torch.sigmoid(unet_model(x_in))
        
    probs_upscaled = F.interpolate(probs, size=orig_norm_01.shape, mode='trilinear', align_corners=False)
    heatmap_3d = probs_upscaled.squeeze().float().cpu().numpy() 
    pred_mask_3d = (heatmap_3d > 0.5).astype(np.uint8)
    
    nib.save(nib.Nifti1Image(pred_mask_3d, img_nifti.affine, img_nifti.header), out_dir / f"{subj_name}_pred_mask.nii.gz")
    
    if has_gt:
        dice_score = compute_dice(pred_mask_3d, gt_mask_bool)
        print(f"-> Dice Score: {dice_score:.4f}")
    else:
        dice_score = None
        print("-> Skipped Dice Score (No GT)")
        
    timing['3_UNet_Inference'] = time.time() - t0

    # 5. Export Visualizations
    print(f"--- Exporting Visualizations & Video ---")
    t0 = time.time()
    best_slice_idx = int(np.argmax(np.sum(gt_mask_bool > 0, axis=(0, 1)))) if has_gt else orig_norm_01.shape[2] // 2
    
    s_orig = orig_norm_01[:, :, best_slice_idx].T
    s_gt = gt_mask_bool[:, :, best_slice_idx].T
    s_mask = pred_mask_3d[:, :, best_slice_idx].T
    
    s_orig_vis = (np.clip(s_orig, 0, 1) * 255).astype(np.uint8)
    overlay = cv2.cvtColor(s_orig_vis, cv2.COLOR_GRAY2RGB)
    overlay[s_mask > 0] = overlay[s_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    if has_gt:
        for contour in find_contours(s_gt, level=0.5):
            plt_contour = np.array([contour[:, 1], contour[:, 0]]).T.astype(np.int32)
            cv2.polylines(overlay, [plt_contour], isClosed=True, color=(0, 255, 0), thickness=1)

    save_plot(s_orig, out_dir / f"{subj_name}_original.png")
    save_plot(noisy_vols[args.t1][:, :, best_slice_idx].T, out_dir / f"{subj_name}_noisy_t{args.t1}.png", cmap='gray')
    save_plot(noisy_vols[args.t2][:, :, best_slice_idx].T, out_dir / f"{subj_name}_noisy_t{args.t2}.png", cmap='gray')
    save_plot(recons[args.t1][:, :, best_slice_idx].T, out_dir / f"{subj_name}_recon_t{args.t1}.png")
    save_plot(recons[args.t2][:, :, best_slice_idx].T, out_dir / f"{subj_name}_recon_t{args.t2}.png")
    save_plot(heatmap_3d[:, :, best_slice_idx].T, out_dir / f"{subj_name}_heatmap.png", cmap='magma', is_heatmap=True)
    save_plot(None, out_dir / f"{subj_name}_overlay.png", overlay=overlay)

    # ---------------- ADDED: Export 8 Wavelet Subbands ----------------
    # Calculate corresponding DWT slice index (accounting for padding and 2x downsampling)
    dwt_slice_idx = (best_slice_idx + pad_w_b) // 2
    for i in range(8):
        save_plot(dwt_vis_cache[i, :, :, dwt_slice_idx].T, out_dir / f"{subj_name}_wavelet_subband_{i}.png", cmap='gray')
    # ------------------------------------------------------------------

    for axis, name in enumerate(['sagittal', 'coronal', 'axial']):
        export_video(orig_norm_01, pred_mask_3d, gt_mask_bool, axis=axis, filename=out_dir / f"{subj_name}_{name}_traversal.avi", has_gt=has_gt)
        
    timing['4_Export_Vis'] = time.time() - t0
    timing['Total_Time'] = sum(timing.values())

    # 6. Save CSV
    csv_path = out_dir / f"{subj_name}_metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Dice Score', f"{dice_score:.4f}" if has_gt else "N/A"])
        for k, v in timing.items():
            writer.writerow([f"Time_{k} (s)", f"{v:.2f}"])
            
    print(f"All processing complete! Results saved to {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Target input nii.gz file")
    parser.add_argument("-o", "--output_dir", type=str, default="./eval_results", help="Output directory root")
    parser.add_argument("--diff_weights", type=str, default="best_ema_weights_256.pt", help="Diffusion weights")
    parser.add_argument("--unet_weights", type=str, default="spatial_unet_best.pt", help="Spatial U-Net weights")
    parser.add_argument("--config", type=str, default="config.json", help="Diffusion config file")
    parser.add_argument("--t1", type=int, default=500)
    parser.add_argument("--t2", type=int, default=700)
    
    args = parser.parse_args()
    process_single_scan(args)