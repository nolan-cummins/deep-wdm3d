# Deep WDM-3D

Pipeline for 3D medical image segmentation using Wavelet Diffusion Models and Spatial U-Nets.

<img width="5969" height="2846" alt="representative_figure" src="https://github.com/user-attachments/assets/7bf70525-8ba6-47a8-a56d-cfd0e3b987b1" />


## Hardware Specifications

* **Training:** 8x NVIDIA RTX A6000 GPUs.
  * Ensure your `config.json` reflects the correct `batch_size` per GPU. 
  * Total effective batch size = `batch_size` × number of GPUs.
* **Inference:** Requires a single GPU with at least **13 GB VRAM** (Peak usage observed: ~12,988 MiB).

## Training

### 1. Wavelet U-Net (Diffusion)
Trains the unconditional diffusion model in the wavelet domain. Requires a `config.json` in the working directory.

```bash
cd train
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 TMPDIR=/tmp uv run torchrun --nproc_per_node=7 train.py
```

### 2. Spatial U-Net (Segmentation)
Trains the 3D U-Net using the original volumes and diffusion-generated reconstructions. Generates a `dataset_splits.csv` ledger to track the validation split.

```bash
cd "evaluate/spatial unet"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TMPDIR=/dev/shm uv run torchrun --nproc_per_node=8 spatial_cnn.py --epochs 1000
```

## Evaluation

### Wavelet U-Net Pipeline (Full Inference)
Runs the full pipeline on a single `.nii.gz` scan: computes DWT, runs DDIM sampling, applies the Spatial U-Net, and exports 3D masks, 2D overlays, and traversal videos.

```bash
cd evaluate
uv run python evaluate.py -i <path_to_input.nii.gz> -o ./eval_results
```
*Required weights: `best_ema_weights_256.pt` (diffusion), `dwt_stds_256.pt` (stats), and `spatial_unet_best.pt` (segmentation).*

#### Inference Performance Benchmark
Approximate execution times for a single volume (based on a single RTX A6000):
* **Data Preparation:** 1.14s
* **Diffusion Sampling (t=500):** 48.54s
* **Diffusion Sampling (t=700):** 67.05s
* **Spatial U-Net Inference:** 0.21s
* **Visualization Export:** 1.61s
* **Total Execution Time:** ~118.55s

### Spatial U-Net Standalone
Evaluates the Spatial U-Net on a randomly selected unseen validation subject (using `dataset_splits.csv`) and generates comparison plots.

```bash
cd "evaluate/spatial unet"
uv run python evaluate_spatial.py
```
