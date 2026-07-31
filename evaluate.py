#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  STANDALONE EVALUATION SCRIPT — KLA Semiconductor Image Restoration
═══════════════════════════════════════════════════════════════════════

  !! CRITICAL !! — This script is used AS-IS by KLA's benchmarking
  team on an H100 GPU. It must run WITHOUT manual edits.

Usage:
    python evaluate.py --input_dir /path/to/test/images \
                       --output_dir /path/to/output

    # With custom model weights
    python evaluate.py --input_dir /path/to/test/images \
                       --output_dir /path/to/output \
                       --weights /path/to/model.pt

    # Enable torch.compile for faster H100 inference
    python evaluate.py --input_dir ... --output_dir ... --compile

The script:
  1. Loads the trained SwinIR model from weights/best_model.pt
  2. Iterates over all images in input_dir
  3. Auto-detects resolution and applies 2× upscaling + denoising
  4. Saves restored images to output_dir (same filename, same format)
  5. Reports per-image and total inference time
"""

import os
import sys
import argparse
import time
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.swinir import SwinIR


# ─── Model Configuration (must match training) ───────────────────────
MODEL_CONFIG = {
    'img_size': 64,
    'patch_size': 1,
    'in_chans': 1,
    'embed_dim': 96,
    'depths': [6, 6, 6, 6],
    'num_heads': [6, 6, 6, 6],
    'window_size': 8,
    'mlp_ratio': 4.0,
    'upscale': 2,
    'img_range': 1.0,
    'resi_connection': '1conv',
}

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.npy'}


def load_model(weights_path, device, use_compile=False):
    """Load trained SwinIR model.

    Args:
        weights_path: Path to .pt checkpoint file
        device: torch device
        use_compile: If True, apply torch.compile() for faster inference

    Returns:
        Model in eval mode
    """
    print(f"Loading model from: {weights_path}")
    model = SwinIR(**MODEL_CONFIG)

    # Load weights
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Optional: torch.compile for faster inference on H100
    if use_compile and hasattr(torch, 'compile'):
        print("Applying torch.compile() for optimized inference...")
        model = torch.compile(model, mode='reduce-overhead')

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params / 1e6:.2f}M parameters")
    return model


def load_image(path):
    """Load a single image and return tensor + metadata for saving.

    Returns:
        img_tensor: (1, 1, H, W) float32 tensor normalized to [0, 1]
        save_info: dict with format info for output saving
    """
    ext = os.path.splitext(path)[1].lower()
    is_npy = (ext == '.npy')

    if is_npy:
        # NumPy array — already float32, values in ~[0, 1] range
        img_np = np.load(path).astype(np.float32)
        save_info = {'is_npy': True, 'max_val': 1.0, 'save_dtype': np.float32}
    else:
        img = Image.open(path)
        img_np = np.array(img).astype(np.float32)
        # Detect bit depth
        if img_np.max() > 255:
            max_val = 65535.0
            save_dtype = np.uint16
        elif img_np.max() > 1.0:
            max_val = 255.0
            save_dtype = np.uint8
        else:
            max_val = 1.0
            save_dtype = np.uint8
        img_np = img_np / max_val
        save_info = {'is_npy': False, 'max_val': max_val, 'save_dtype': save_dtype}

    # Ensure single channel
    if img_np.ndim == 3:
        img_np = img_np[:, :, 0]

    # To tensor: (1, 1, H, W)
    img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)

    return img_tensor, save_info


def save_image(tensor, path, save_info):
    """Save output tensor as image or .npy file.

    Args:
        tensor: (1, 1, H, W) or (H, W) tensor
        path: Output file path
        save_info: Dict from load_image with format metadata
    """
    img_np = tensor.squeeze().cpu().numpy()

    # Clamp to valid range
    img_np = np.clip(img_np, 0, 1)

    if save_info.get('is_npy', False):
        # Save as .npy (preserving float32 precision)
        np.save(path, img_np.astype(np.float32))
    else:
        # Save as image
        img_np = (img_np * save_info['max_val']).astype(save_info['save_dtype'])
        img = Image.fromarray(img_np)
        img.save(path)


def find_images(directory):
    """Find all valid image files in a directory."""
    files = []
    for f in sorted(os.listdir(directory)):
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS:
            files.append(os.path.join(directory, f))
    return files


def main():
    parser = argparse.ArgumentParser(
        description='KLA Semiconductor Image Restoration — Evaluation Script')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing degraded test images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save restored images')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights (default: weights/best_model.pt)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (default: auto-detect)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for faster inference on H100')
    args = parser.parse_args()

    # ── Device ─────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # ── Weights path ───────────────────────────────────────────────────
    if args.weights is None:
        args.weights = os.path.join(SCRIPT_DIR, 'weights', 'best_model.pt')

    if not os.path.exists(args.weights):
        print(f"ERROR: Model weights not found at {args.weights}")
        print("Please ensure 'weights/best_model.pt' exists or specify --weights")
        sys.exit(1)

    # ── Create output directory ────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────
    model = load_model(args.weights, device, use_compile=args.compile)

    # ── Find input images ──────────────────────────────────────────────
    input_files = find_images(args.input_dir)

    if len(input_files) == 0:
        print(f"ERROR: No images found in {args.input_dir}")
        sys.exit(1)

    print(f"\n{'═' * 65}")
    print(f"  Restoring {len(input_files)} images")
    print(f"  Device:  {device}")
    print(f"  Input:   {args.input_dir}")
    print(f"  Output:  {args.output_dir}")
    print(f"{'═' * 65}\n")
    print(f"{'Image':<35} {'Input Res':<12} {'Output Res':<12} {'Time':<8}")
    print(f"{'─' * 67}")

    # ── Inference ──────────────────────────────────────────────────────
    total_time = 0.0

    # Warmup run (for accurate timing)
    if device.type in ('cuda', 'mps'):
        dummy = torch.randn(1, 1, 64, 64, device=device)
        with torch.no_grad():
            _ = model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

    with torch.no_grad():
        for i, filepath in enumerate(input_files):
            filename = os.path.basename(filepath)

            # Load
            img_tensor, save_info = load_image(filepath)
            img_tensor = img_tensor.to(device)
            input_shape = f"{img_tensor.shape[2]}×{img_tensor.shape[3]}"

            # Inference with timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            t_start = time.time()

            output = model(img_tensor)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            elapsed = time.time() - t_start
            total_time += elapsed

            output_shape = f"{output.shape[2]}×{output.shape[3]}"

            # Save
            out_path = os.path.join(args.output_dir, filename)
            save_image(output, out_path, save_info)

            print(f"{filename:<35} {input_shape:<12} {output_shape:<12} {elapsed:.3f}s")

    # ── Summary ────────────────────────────────────────────────────────
    avg_time = total_time / len(input_files)
    print(f"{'─' * 67}")
    print(f"\n  Total time:   {total_time:.2f}s")
    print(f"  Average time: {avg_time:.3f}s per image")
    print(f"  Throughput:   {1.0 / avg_time:.1f} images/sec")
    print(f"  Outputs:      {args.output_dir}")
    print(f"\n{'═' * 65}")
    print(f"  ✓ Done! {len(input_files)} images restored successfully.")
    print(f"{'═' * 65}\n")


if __name__ == '__main__':
    main()
