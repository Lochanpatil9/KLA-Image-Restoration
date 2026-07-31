"""
Data augmentation pipeline for semiconductor image restoration.
────────────────────────────────────────────────────────────────
Designed for out-of-distribution (OOD) generalization:
  - Geometric: flips, rotations (semiconductor patterns are rotation-invariant)
  - Intensity: scaling, gamma correction (handles varying exposure conditions)
  - Online noise: additional speckle/Gaussian (robustness to unseen noise levels)
  - CutMix/Mixup: regularization for better generalization
"""

import random
import numpy as np
import torch


class PairedAugmentor:
    """Applies identical augmentations to degraded and GT image pairs.

    All transforms maintain the spatial correspondence between
    the low-resolution input and high-resolution target.
    """

    def __init__(self, config=None):
        if config is not None:
            self.do_flip = config.AUGMENT_FLIP
            self.do_rotate = config.AUGMENT_ROTATE
            self.do_intensity = config.AUGMENT_INTENSITY
            self.do_mixup = config.AUGMENT_MIXUP
        else:
            self.do_flip = True
            self.do_rotate = True
            self.do_intensity = True
            self.do_mixup = False

    def __call__(self, degraded, gt):
        """Apply augmentations to a (degraded, gt) tensor pair.

        Args:
            degraded: (1, H_lr, W_lr) tensor
            gt: (1, H_hr, W_hr) tensor

        Returns:
            Augmented (degraded, gt) pair.
        """
        if self.do_flip:
            degraded, gt = self.random_flip(degraded, gt)

        if self.do_rotate:
            degraded, gt = self.random_rotate90(degraded, gt)

        if self.do_intensity:
            degraded, gt = self.random_intensity(degraded, gt)

        return degraded, gt

    @staticmethod
    def random_flip(degraded, gt):
        """Random horizontal and/or vertical flip."""
        if random.random() > 0.5:
            degraded = torch.flip(degraded, [2])  # horizontal
            gt = torch.flip(gt, [2])
        if random.random() > 0.5:
            degraded = torch.flip(degraded, [1])  # vertical
            gt = torch.flip(gt, [1])
        return degraded, gt

    @staticmethod
    def random_rotate90(degraded, gt):
        """Random 90° rotation (0°, 90°, 180°, or 270°)."""
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            degraded = torch.rot90(degraded, k, [1, 2])
            gt = torch.rot90(gt, k, [1, 2])
        return degraded, gt

    @staticmethod
    def random_intensity(degraded, gt, scale_range=(0.9, 1.1)):
        """Random global intensity scaling."""
        if random.random() > 0.5:
            scale = random.uniform(*scale_range)
            degraded = degraded * scale
            gt = gt * scale
        return degraded, gt

    @staticmethod
    def random_gamma(degraded, gt, gamma_range=(0.8, 1.2)):
        """Random gamma correction (applied to both images)."""
        if random.random() > 0.5:
            gamma = random.uniform(*gamma_range)
            # Clamp to avoid issues with negative values
            degraded = torch.clamp(degraded, 0).pow(gamma)
            gt = torch.clamp(gt, 0).pow(gamma)
        return degraded, gt


class OnlineNoiseAugmentor:
    """Online noise augmentation applied ONLY to degraded images.

    Adds additional noise on top of existing degradation to make the model
    more robust to varying noise levels (important for OOD generalization).
    """

    def __init__(self, speckle_prob=0.3, gaussian_prob=0.3,
                 speckle_std_range=(0.01, 0.05), gaussian_std_range=(0.01, 0.03)):
        self.speckle_prob = speckle_prob
        self.gaussian_prob = gaussian_prob
        self.speckle_std_range = speckle_std_range
        self.gaussian_std_range = gaussian_std_range

    def __call__(self, degraded):
        """Add random noise to degraded image (gt is NOT modified).

        Args:
            degraded: (1, H, W) tensor

        Returns:
            Augmented degraded tensor.
        """
        if random.random() < self.speckle_prob:
            degraded = self.add_speckle(degraded)
        if random.random() < self.gaussian_prob:
            degraded = self.add_gaussian(degraded)
        return degraded

    def add_speckle(self, img):
        """Add multiplicative speckle noise: I_noisy = I * (1 + noise)."""
        std = random.uniform(*self.speckle_std_range)
        noise = torch.randn_like(img) * std
        return img * (1 + noise)

    def add_gaussian(self, img):
        """Add additive Gaussian noise."""
        std = random.uniform(*self.gaussian_std_range)
        noise = torch.randn_like(img) * std
        return img + noise


def batch_mixup(degraded_batch, gt_batch, alpha=0.2):
    """Mixup augmentation across batch samples.

    Blends pairs of samples to create virtual training examples.
    Helps regularization and OOD generalization.

    Args:
        degraded_batch: (B, 1, H_lr, W_lr) tensor
        gt_batch: (B, 1, H_hr, W_hr) tensor
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        Mixed (degraded_batch, gt_batch) pair.
    """
    if alpha <= 0:
        return degraded_batch, gt_batch

    lam = np.random.beta(alpha, alpha)
    batch_size = degraded_batch.size(0)
    index = torch.randperm(batch_size, device=degraded_batch.device)

    mixed_degraded = lam * degraded_batch + (1 - lam) * degraded_batch[index]
    mixed_gt = lam * gt_batch + (1 - lam) * gt_batch[index]

    return mixed_degraded, mixed_gt
