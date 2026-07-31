#!/usr/bin/env python3
"""
Quick inference demo with visual comparisons.
──────────────────────────────────────────────
Runs the model on a few sample images and creates side-by-side
comparison plots (degraded → restored → ground truth).

Usage:
    # With ground truth (shows metrics)
    python inference.py --degraded_dir data_raw/val/degraded \
                        --gt_dir data_raw/val/ground_truth \
                        --output_dir outputs/demo

    # Without ground truth (inference only)
    python inference.py --degraded_dir test_images/ \
                        --output_dir outputs/test

    # Limit number of images
    python inference.py --degraded_dir ... --max_images 10
"""

import os
import sys
import argparse
import time
import numpy as np
from PIL import Image
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.swinir import SwinIR
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import plot_comparison


MODEL_CONFIG = {
    'img_size': 64, 'patch_size': 1, 'in_chans': 1,
    'embed_dim': 96, 'depths': [6, 6, 6, 6],
    'num_heads': [6, 6, 6, 6], 'window_size': 8,
    'mlp_ratio': 4.0, 'upscale': 2, 'img_range': 1.0,
    'resi_connection': '1conv',
}

VALID_EXT = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.npy'}


def main():
    parser = argparse.ArgumentParser(description='Inference demo with visualizations')
    parser.add_argument('--degraded_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Ground truth directory (optional — enables metrics)')
    parser.add_argument('--output_dir', type=str, default='outputs/demo')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--max_images', type=int, default=20)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    weights_path = args.weights or os.path.join(SCRIPT_DIR, 'weights', 'best_model.pt')

    # Load model
    model = SwinIR(**MODEL_CONFIG)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # Find images
    degraded_files = sorted([os.path.join(args.degraded_dir, f)
                             for f in os.listdir(args.degraded_dir)
                             if os.path.splitext(f)[1].lower() in VALID_EXT])
    degraded_files = degraded_files[:args.max_images]

    gt_files = None
    if args.gt_dir:
        gt_files = sorted([os.path.join(args.gt_dir, f)
                           for f in os.listdir(args.gt_dir)
                           if os.path.splitext(f)[1].lower() in VALID_EXT])
        gt_files = gt_files[:args.max_images]

    # Output dirs
    restored_dir = os.path.join(args.output_dir, 'restored')
    vis_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(restored_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print(f"\nProcessing {len(degraded_files)} images...")
    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for i, deg_path in enumerate(degraded_files):
            fname = os.path.basename(deg_path)

            # Load degraded
            deg_img = np.array(Image.open(deg_path)).astype(np.float32)
            max_val = 65535.0 if deg_img.max() > 255 else (255.0 if deg_img.max() > 1 else 1.0)
            deg_img = deg_img / max_val
            if deg_img.ndim == 3:
                deg_img = deg_img[:, :, 0]

            deg_tensor = torch.from_numpy(deg_img).float().unsqueeze(0).unsqueeze(0).to(device)

            # Inference
            t = time.time()
            output = model(deg_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - t

            # To numpy
            restored = output.squeeze().cpu().numpy()
            restored = np.clip(restored, 0, 1)

            # Save restored
            save_arr = (restored * (65535 if max_val > 255 else 255)).astype(
                np.uint16 if max_val > 255 else np.uint8)
            Image.fromarray(save_arr).save(os.path.join(restored_dir, fname))

            # Compute metrics if GT available
            psnr_val, ssim_val = None, None
            gt_img = None
            if gt_files and i < len(gt_files):
                gt_img = np.array(Image.open(gt_files[i])).astype(np.float32)
                gt_img = gt_img / (65535.0 if gt_img.max() > 255 else (255.0 if gt_img.max() > 1 else 1.0))
                if gt_img.ndim == 3:
                    gt_img = gt_img[:, :, 0]
                psnr_val = calculate_psnr(restored, gt_img)
                ssim_val = calculate_ssim(restored, gt_img)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)

                # Visual comparison
                plot_comparison(deg_img, restored, gt_img,
                                filename=f'cmp_{fname}.png',
                                save_dir=vis_dir,
                                psnr=psnr_val, ssim=ssim_val)

            metrics_str = ""
            if psnr_val is not None:
                metrics_str = f" | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}"
            print(f"[{i + 1}/{len(degraded_files)}] {fname} | {elapsed:.3f}s{metrics_str}")

    # Summary
    if psnr_list:
        print(f"\nAverage PSNR: {np.mean(psnr_list):.2f} dB")
        print(f"Average SSIM: {np.mean(ssim_list):.4f}")
    print(f"Restored images: {restored_dir}")
    print(f"Comparisons: {vis_dir}")


if __name__ == '__main__':
    main()
