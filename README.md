# 🔬 AI-Based Restoration of Degraded Semiconductor Inspection Images

> SwinIR-based joint denoising + 2× super-resolution for semiconductor manufacturing quality control.

## Quick Start — Inference (for KLA Benchmarking)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/kla-image-restoration.git
cd kla-image-restoration

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run evaluation on test images
python evaluate.py --input_dir /path/to/test/images --output_dir /path/to/output
```

That's it. The script auto-detects GPU (CUDA/MPS/CPU), loads `weights/best_model.pt`, restores all images, and saves outputs.

### Evaluation Script Details

```bash
# Basic usage (auto-detects H100 GPU)
python evaluate.py --input_dir /path/to/Test_NoisyLR --output_dir /path/to/restored

# With torch.compile for faster H100 inference
python evaluate.py --input_dir /path/to/Test_NoisyLR --output_dir /path/to/restored --compile

# With custom weights
python evaluate.py --input_dir /path/to/Test_NoisyLR --output_dir /path/to/restored --weights /path/to/model.pt
```

**Input**: Directory of degraded `.npy` images (128×128, float32, grayscale)  
**Output**: Directory of restored `.npy` images (256×256, float32, grayscale, values in [0, 1])

---

## Training from Scratch

```bash
python train.py \
    --degraded_dir /path/to/train/NoisyLR \
    --gt_dir /path/to/train/GT \
    --output_dir experiments/run_01 \
    --no_perceptual
```

### Training Hyperparameters (all in `config.py`)

| Setting | Value |
|---|---|
| Model | SwinIR (96-dim, 4 RSTB blocks, 3.34M params) |
| Optimizer | AdamW (β₁=0.9, β₂=0.99, wd=1e-4) |
| Learning Rate | 2×10⁻⁴ → 1×10⁻⁶ (cosine annealing) |
| Warmup | 2 epochs linear |
| Batch Size | 8 |
| Patch Size | 64×64 (LR) → 128×128 (HR) |
| Epochs | 30 |
| Loss | Charbonnier + FFT + Edge (Sobel) |
| Augmentation | Flips, 90° rotations, intensity scaling |

### Training Results

| Metric | Value |
|---|---|
| Best PSNR | 29.00 dB |
| Best SSIM | 0.7698 |
| Final Loss | 0.1831 |

---

## Repository Structure

```
├── README.md                   # This file
├── evaluate.py                 # ★ STANDALONE evaluation script (CRITICAL)
├── train.py                    # Training script
├── config.py                   # All hyperparameters
├── inference.py                # Demo with visual comparisons
├── requirements.txt            # Pinned dependencies
│
├── weights/
│   └── best_model.pt           # Trained model (51 MB, Git LFS)
│
├── outputs/
│   └── test_results/           # Restored test images (400 × .npy)
│
├── results/                    # Visualization comparisons
│   ├── summary_grid.png        # 4×3 overview grid
│   ├── comparison_train_*.png  # Before/after with GT + metrics
│   └── comparison_test_*.png   # Before/after on test set
│
├── models/
│   ├── swinir.py               # SwinIR architecture
│   └── losses.py               # Combined loss functions
│
├── data/
│   ├── dataset.py              # Dataset class (.npy + image support)
│   └── augmentations.py        # Data augmentation pipeline
│
├── utils/
│   ├── metrics.py              # PSNR, SSIM, LPIPS
│   └── visualization.py        # Comparison plotting
│
└── experiments/                # Training logs & checkpoints
    └── run_01/
```

## Model Architecture

**SwinIR** (Swin Transformer for Image Restoration):

| Component | Details |
|---|---|
| Backbone | 4 RSTB blocks, each with 6 Swin Transformer layers |
| Attention | Shifted-window self-attention, window size 8 |
| Embedding | 96-dimensional, 6 heads |
| Upsampling | PixelShuffle 2× |
| Parameters | 3.34M |
| Input | Grayscale `.npy` (1 channel), any resolution |
| Output | 2× upscaled, denoised `.npy` image |

## Loss Function

```
L_total = 1.0 × L_charbonnier + 0.1 × L_fft + 0.05 × L_edge
```

| Loss | Purpose |
|---|---|
| **Charbonnier** | Pixel accuracy, robust to speckle noise outliers |
| **FFT** | Frequency-domain accuracy for periodic semiconductor patterns |
| **Edge (Sobel)** | Sharp edge preservation for defect detection |

## Data Format

- **Input**: `.npy` float32 arrays, shape (128, 128), values may exceed [0, 1] due to speckle noise
- **Output**: `.npy` float32 arrays, shape (256, 256), values in [0, 1]
- **Ground Truth**: `.npy` float32 arrays, shape (256, 256), values in [0, 1]

## Hardware

| Phase | Tested On | Notes |
|---|---|---|
| Training | Apple M-series (MPS), 16GB | ~5 hours for 30 epochs |
| Evaluation | Apple M-series (MPS) | 0.24s per image |
| Target | NVIDIA H100 (CUDA) | torch.compile supported |

## References

- Liang et al., "SwinIR: Image Restoration Using Swin Transformer", ICCVW 2021
- Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
