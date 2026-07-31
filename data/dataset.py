"""
Dataset class for paired semiconductor inspection images.
─────────────────────────────────────────────────────────
Handles:
  - Mixed resolution pairs (128→256 and 256→512)
  - Multiple image formats (PNG, TIFF, BMP, JPEG)
  - 8-bit and 16-bit images
  - Beyond-range intensity values (speckle noise artifact)
  - Patch-based training with random cropping
  - Data augmentation (flips, rotations, intensity scaling)
"""

import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class SemiconductorDataset(Dataset):
    """Paired dataset for degraded → clean semiconductor image restoration.

    Expects two directories with matching filenames:
      degraded_dir/  — noisy, low-resolution inputs
      gt_dir/        — clean, full-resolution ground truth

    Files are matched by sorted order (name-based matching).
    """

    VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.npy'}

    def __init__(self, degraded_dir, gt_dir, patch_size=64, upscale=2,
                 augment=True, is_train=True):
        """
        Args:
            degraded_dir: Path to degraded (noisy, low-res) images.
            gt_dir: Path to ground truth (clean, full-res) images.
            patch_size: Patch size in LR space. GT patch = patch_size * upscale.
            upscale: Super-resolution scale factor.
            augment: Whether to apply data augmentation.
            is_train: If True, use random crops. If False, use full images.
        """
        super().__init__()
        self.degraded_dir = degraded_dir
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.upscale = upscale
        self.augment = augment and is_train
        self.is_train = is_train

        # Discover image files
        self.degraded_files = self._find_images(degraded_dir)
        self.gt_files = self._find_images(gt_dir)

        # Verify pairing
        assert len(self.degraded_files) > 0, f"No images found in {degraded_dir}"
        assert len(self.gt_files) > 0, f"No images found in {gt_dir}"
        assert len(self.degraded_files) == len(self.gt_files), (
            f"Image count mismatch: {len(self.degraded_files)} degraded "
            f"vs {len(self.gt_files)} ground truth")

        print(f"{'Train' if is_train else 'Val'} dataset: "
              f"{len(self.degraded_files)} pairs from {degraded_dir}")

    def _find_images(self, directory):
        """Find all valid image files in directory, sorted by name."""
        files = []
        for f in sorted(os.listdir(directory)):
            if os.path.splitext(f)[1].lower() in self.VALID_EXTENSIONS:
                files.append(os.path.join(directory, f))
        return files

    @staticmethod
    def load_image(path):
        """Load image as float32 numpy array, normalized to [0, 1].

        Handles:
          - .npy files (NumPy arrays) — loaded directly
          - 8-bit (0-255) and 16-bit (0-65535) image files
          - Grayscale and RGB (converts to grayscale)
          - Speckle noise artifacts (values > 1.0 or < 0 are preserved, NOT clamped)
        """
        ext = os.path.splitext(path)[1].lower()

        if ext == '.npy':
            # NumPy array — load directly (already float32 in [0,1] range,
            # though degraded images may exceed this range due to speckle noise)
            img = np.load(path).astype(np.float32)
        else:
            # Image file — load via PIL
            img = np.array(Image.open(path)).astype(np.float32)
            # Normalize based on detected bit depth
            if img.max() > 255:
                img = img / 65535.0   # 16-bit
            elif img.max() > 1.0:
                img = img / 255.0    # 8-bit
            # else: already in [0, 1]

        # Convert to single channel if needed
        if img.ndim == 3:
            if img.shape[2] == 3:
                # RGB → grayscale (luminance)
                img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
            else:
                img = img[:, :, 0]

        return img

    def __len__(self):
        return len(self.degraded_files)

    def __getitem__(self, idx):
        # Load pair
        degraded = self.load_image(self.degraded_files[idx])
        gt = self.load_image(self.gt_files[idx])

        # Verify resolution relationship
        lr_h, lr_w = degraded.shape
        gt_h, gt_w = gt.shape
        assert gt_h == lr_h * self.upscale and gt_w == lr_w * self.upscale, (
            f"Resolution mismatch: degraded {degraded.shape} × {self.upscale} "
            f"≠ GT {gt.shape} for {os.path.basename(self.degraded_files[idx])}")

        if self.is_train:
            # ── Random crop (in LR space) ──────────────────────────────
            lr_patch = self.patch_size
            gt_patch = self.patch_size * self.upscale

            top_lr = random.randint(0, lr_h - lr_patch)
            left_lr = random.randint(0, lr_w - lr_patch)

            degraded = degraded[top_lr:top_lr + lr_patch,
                                left_lr:left_lr + lr_patch]

            top_gt = top_lr * self.upscale
            left_gt = left_lr * self.upscale
            gt = gt[top_gt:top_gt + gt_patch,
                    left_gt:left_gt + gt_patch]

        # Convert to tensors: (1, H, W)
        degraded = torch.from_numpy(degraded.copy()).unsqueeze(0)
        gt = torch.from_numpy(gt.copy()).unsqueeze(0)

        if self.augment:
            degraded, gt = self._augment(degraded, gt)

        return {
            'degraded': degraded,
            'gt': gt,
            'filename': os.path.basename(self.degraded_files[idx])
        }

    def _augment(self, degraded, gt):
        """Apply paired augmentation (same transform to both images)."""
        # Random horizontal flip
        if random.random() > 0.5:
            degraded = torch.flip(degraded, [2])
            gt = torch.flip(gt, [2])

        # Random vertical flip
        if random.random() > 0.5:
            degraded = torch.flip(degraded, [1])
            gt = torch.flip(gt, [1])

        # Random 90° rotation (k × 90°)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            degraded = torch.rot90(degraded, k, [1, 2])
            gt = torch.rot90(gt, k, [1, 2])

        # Random intensity scaling (±10%) — helps OOD generalization
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            degraded = degraded * scale
            gt = gt * scale

        return degraded, gt


class SemiconductorTestDataset(Dataset):
    """Test dataset — single directory of degraded images (no ground truth).

    Used during inference when GT is not available.
    """

    VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.npy'}

    def __init__(self, input_dir):
        super().__init__()
        self.input_dir = input_dir
        self.files = []
        for f in sorted(os.listdir(input_dir)):
            if os.path.splitext(f)[1].lower() in self.VALID_EXTENSIONS:
                self.files.append(os.path.join(input_dir, f))
        assert len(self.files) > 0, f"No images found in {input_dir}"
        print(f"Test dataset: {len(self.files)} images from {input_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = SemiconductorDataset.load_image(self.files[idx])
        img_tensor = torch.from_numpy(img.copy()).unsqueeze(0)
        return {
            'image': img_tensor,
            'filename': os.path.basename(self.files[idx]),
            'is_npy': self.files[idx].endswith('.npy'),
        }
