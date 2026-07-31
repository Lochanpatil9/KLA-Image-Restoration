"""
Visualization utilities for semiconductor image restoration.
─────────────────────────────────────────────────────────────
Creates before/after comparison figures:
  Degraded (input) → Restored (model output) → Ground Truth (target)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image


def plot_comparison(degraded, restored, gt, filename='comparison.png',
                    save_dir='.', psnr=None, ssim=None):
    """Plot side-by-side comparison: Degraded → Restored → Ground Truth.

    Args:
        degraded: (H_lr, W_lr) numpy array or torch tensor
        restored: (H_hr, W_hr) numpy array or torch tensor
        gt: (H_hr, W_hr) numpy array or torch tensor
        filename: Output filename
        save_dir: Directory to save the figure
        psnr: Optional PSNR value to display
        ssim: Optional SSIM value to display
    """
    import torch

    # Convert tensors to numpy
    def to_np(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x.squeeze()

    degraded = to_np(degraded)
    restored = to_np(restored)
    gt = to_np(gt)

    # Clamp for display
    restored_disp = np.clip(restored, 0, 1)
    gt_disp = np.clip(gt, 0, 1)
    degraded_disp = np.clip(degraded, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(degraded_disp, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Degraded Input\n({degraded.shape[0]}×{degraded.shape[1]})',
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')

    title = f'Restored Output\n({restored.shape[0]}×{restored.shape[1]})'
    if psnr is not None and ssim is not None:
        title += f'\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}'
    axes[1].imshow(restored_disp, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(title, fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(gt_disp, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Ground Truth\n({gt.shape[0]}×{gt.shape[1]})',
                      fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {save_path}")


def plot_training_curves(log_file, save_path='training_curves.png'):
    """Plot training loss and validation metrics from TensorBoard logs.

    Args:
        log_file: Path to a CSV file with columns: epoch, train_loss, val_psnr, val_ssim
        save_path: Where to save the plot
    """
    data = np.loadtxt(log_file, delimiter=',', skiprows=1)
    epochs = data[:, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training loss
    axes[0].plot(epochs, data[:, 1], color='#E63946', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Validation PSNR
    axes[1].plot(epochs, data[:, 2], color='#457B9D', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('Validation PSNR', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Validation SSIM
    axes[2].plot(epochs, data[:, 3], color='#2A9D8F', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('SSIM')
    axes[2].set_title('Validation SSIM', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {save_path}")


def create_grid(images, titles=None, ncols=4, save_path='grid.png'):
    """Create a grid of images for quick visual inspection.

    Args:
        images: List of numpy arrays (H, W)
        titles: Optional list of titles for each image
        ncols: Number of columns in grid
        save_path: Where to save the grid
    """
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(np.clip(images[i].squeeze(), 0, 1), cmap='gray', vmin=0, vmax=1)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grid saved: {save_path}")
