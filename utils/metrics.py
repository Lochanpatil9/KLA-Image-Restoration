"""
Image quality metrics for semiconductor image restoration.
───────────────────────────────────────────────────────────
Implements the three metrics required by the competition:
  - PSNR  (Peak Signal-to-Noise Ratio)
  - SSIM  (Structural Similarity Index Measure)
  - LPIPS (Learned Perceptual Image Patch Similarity)
"""

import os
import glob
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

# LPIPS is optional — only needed for final scoring, not for training
try:
    import lpips as _lpips_module
    LPIPS_AVAILABLE = True
except ImportError:
    _lpips_module = None
    LPIPS_AVAILABLE = False


def calculate_psnr(pred, target, data_range=1.0):
    """Calculate PSNR between two images.

    Args:
        pred: Predicted image (torch.Tensor or np.ndarray)
        target: Ground truth image (torch.Tensor or np.ndarray)
        data_range: Maximum possible pixel value range

    Returns:
        PSNR value in dB (higher is better)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Remove batch/channel dims if present
    pred = pred.squeeze()
    target = target.squeeze()

    return peak_signal_noise_ratio(target, pred, data_range=data_range)


def calculate_ssim(pred, target, data_range=1.0):
    """Calculate SSIM between two images.

    Args:
        pred: Predicted image (torch.Tensor or np.ndarray)
        target: Ground truth image (torch.Tensor or np.ndarray)
        data_range: Maximum possible pixel value range

    Returns:
        SSIM value in [0, 1] (higher is better)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = pred.squeeze()
    target = target.squeeze()

    return structural_similarity(target, pred, data_range=data_range)


def calculate_lpips(pred, target, net='alex'):
    """Calculate LPIPS between two images.

    Uses a pretrained AlexNet/VGG network to compute perceptual distance.
    Requires: pip install lpips

    Args:
        pred: Predicted image (torch.Tensor, shape [1, 1, H, W] or [1, H, W])
        target: Ground truth image (same shape as pred)
        net: Network to use ('alex' or 'vgg')

    Returns:
        LPIPS distance (lower is better)
    """
    if not LPIPS_AVAILABLE:
        raise ImportError(
            "lpips package not installed. Install with: pip install lpips")

    # Ensure 4D tensors
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(0)

    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(0)

    # LPIPS expects 3-channel images in [-1, 1]
    if pred.shape[1] == 1:
        pred = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)

    # Normalize from [0, 1] to [-1, 1]
    pred = pred * 2 - 1
    target = target * 2 - 1

    # Create LPIPS model (cached to avoid reloading)
    if not hasattr(calculate_lpips, '_model') or calculate_lpips._net != net:
        calculate_lpips._model = _lpips_module.LPIPS(net=net, verbose=False)
        calculate_lpips._net = net

    model = calculate_lpips._model.to(pred.device)

    with torch.no_grad():
        distance = model(pred, target)

    return distance.item()


def compute_all_metrics(pred_dir, gt_dir, data_range=1.0, compute_lpips_flag=True):
    """Compute PSNR, SSIM, LPIPS for all image pairs in two directories.

    Args:
        pred_dir: Directory containing predicted/restored images.
        gt_dir: Directory containing ground truth images.
        data_range: Image value range for PSNR/SSIM computation.
        compute_lpips_flag: Whether to compute LPIPS (requires lpips package).

    Returns:
        Dictionary with per-image and average metrics.
    """
    valid_ext = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}

    pred_files = sorted([f for f in os.listdir(pred_dir)
                         if os.path.splitext(f)[1].lower() in valid_ext])
    gt_files = sorted([f for f in os.listdir(gt_dir)
                       if os.path.splitext(f)[1].lower() in valid_ext])

    assert len(pred_files) == len(gt_files), (
        f"File count mismatch: {len(pred_files)} predictions vs {len(gt_files)} ground truth")

    results = []
    psnr_sum, ssim_sum, lpips_sum = 0, 0, 0

    print(f"Computing metrics for {len(pred_files)} image pairs...")
    print(f"{'Image':<30} {'PSNR (dB)':<12} {'SSIM':<10} {'LPIPS':<10}")
    print("─" * 62)

    for pred_name, gt_name in zip(pred_files, gt_files):
        # Load images
        pred_img = np.array(Image.open(os.path.join(pred_dir, pred_name))).astype(np.float32)
        gt_img = np.array(Image.open(os.path.join(gt_dir, gt_name))).astype(np.float32)

        # Normalize
        if pred_img.max() > 255:
            pred_img /= 65535.0
            gt_img /= 65535.0
        elif pred_img.max() > 1.0:
            pred_img /= 255.0
            gt_img /= 255.0

        # Squeeze to 2D
        if pred_img.ndim == 3:
            pred_img = pred_img[:, :, 0]
        if gt_img.ndim == 3:
            gt_img = gt_img[:, :, 0]

        # PSNR & SSIM
        psnr_val = peak_signal_noise_ratio(gt_img, pred_img, data_range=data_range)
        ssim_val = structural_similarity(gt_img, pred_img, data_range=data_range)
        psnr_sum += psnr_val
        ssim_sum += ssim_val

        # LPIPS
        lpips_val = 0.0
        if compute_lpips_flag:
            try:
                pred_t = torch.from_numpy(pred_img).float()
                gt_t = torch.from_numpy(gt_img).float()
                lpips_val = calculate_lpips(pred_t, gt_t)
                lpips_sum += lpips_val
            except Exception as e:
                print(f"  Warning: LPIPS failed for {pred_name}: {e}")
                compute_lpips_flag = False

        results.append({
            'filename': pred_name,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'lpips': lpips_val
        })

        print(f"{pred_name:<30} {psnr_val:<12.2f} {ssim_val:<10.4f} {lpips_val:<10.4f}")

    n = len(results)
    avg_psnr = psnr_sum / n
    avg_ssim = ssim_sum / n
    avg_lpips = lpips_sum / n if compute_lpips_flag else 0.0

    print("─" * 62)
    print(f"{'AVERAGE':<30} {avg_psnr:<12.2f} {avg_ssim:<10.4f} {avg_lpips:<10.4f}")

    return {
        'per_image': results,
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'avg_lpips': avg_lpips,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute image quality metrics')
    parser.add_argument('--pred_dir', type=str, required=True, help='Predicted images directory')
    parser.add_argument('--gt_dir', type=str, required=True, help='Ground truth images directory')
    parser.add_argument('--no-lpips', action='store_true', help='Skip LPIPS computation')
    args = parser.parse_args()

    compute_all_metrics(args.pred_dir, args.gt_dir, compute_lpips_flag=not args.no_lpips)
