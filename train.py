#!/usr/bin/env python3
"""
Training script for SwinIR-based semiconductor image restoration.
─────────────────────────────────────────────────────────────────
Trains SwinIR to jointly denoise (speckle + Gaussian) and super-resolve (2×)
degraded semiconductor inspection images.

Usage:
    python train.py --degraded_dir data_raw/train/degraded \
                    --gt_dir data_raw/train/ground_truth \
                    --output_dir experiments/run_01

    # Resume from checkpoint
    python train.py --degraded_dir ... --gt_dir ... \
                    --resume experiments/run_01/weights/checkpoint_epoch100.pt
"""

import os
import sys
import argparse
import time
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from models.swinir import SwinIR, build_swinir
from models.losses import CombinedLoss
from data.dataset import SemiconductorDataset
from utils.metrics import calculate_psnr, calculate_ssim


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_lr_scheduler(optimizer, cfg):
    """Create learning rate scheduler with optional warmup."""

    # Cosine annealing
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS - cfg.WARMUP_EPOCHS, eta_min=cfg.MIN_LR)

    if cfg.WARMUP_EPOCHS > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=cfg.WARMUP_EPOCHS)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[cfg.WARMUP_EPOCHS])
    else:
        scheduler = cosine_scheduler

    return scheduler


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device,
                    epoch, cfg, use_amp=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_dict_accum = {}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.EPOCHS}",
                leave=False, ncols=100)

    for batch_idx, batch in enumerate(pbar):
        degraded = batch['degraded'].to(device, non_blocking=True)
        gt = batch['gt'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass (CUDA only)
        with torch.amp.autocast('cuda', enabled=use_amp):
            output = model(degraded)
            loss, loss_dict = criterion(output, gt)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_dict_accum[k] = loss_dict_accum.get(k, 0) + v

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    n = len(dataloader)
    avg_loss = total_loss / n
    avg_dict = {k: v / n for k, v in loss_dict_accum.items()}
    return avg_loss, avg_dict


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate and compute metrics."""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    count = 0

    for batch in dataloader:
        degraded = batch['degraded'].to(device, non_blocking=True)
        gt = batch['gt'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=False):
            output = model(degraded)
            loss, _ = criterion(output, gt)

        total_loss += loss.item()

        # Compute per-image metrics
        output_clamped = torch.clamp(output, 0, 1)
        for i in range(output.shape[0]):
            pred_np = output_clamped[i].cpu().numpy().squeeze()
            gt_np = gt[i].cpu().numpy().squeeze()
            total_psnr += calculate_psnr(pred_np, gt_np)
            total_ssim += calculate_ssim(pred_np, gt_np)
            count += 1

    return {
        'loss': total_loss / len(dataloader),
        'psnr': total_psnr / max(count, 1),
        'ssim': total_ssim / max(count, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train SwinIR for semiconductor image restoration')
    parser.add_argument('--degraded_dir', type=str, required=True,
                        help='Path to degraded (noisy, low-res) training images')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Path to ground truth (clean, full-res) training images')
    parser.add_argument('--output_dir', type=str, default='./experiments/default',
                        help='Output directory for checkpoints, logs, etc.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--no_perceptual', action='store_true',
                        help='Disable perceptual loss (avoids VGG download)')
    args = parser.parse_args()

    # ── Configuration ──────────────────────────────────────────────────
    cfg = Config()
    if args.epochs:
        cfg.EPOCHS = args.epochs
    if args.batch_size:
        cfg.BATCH_SIZE = args.batch_size
    if args.lr:
        cfg.LEARNING_RATE = args.lr

    set_seed(cfg.SEED)

    # Device detection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Mixed precision only on CUDA
    use_amp = (device.type == 'cuda')

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    weights_dir = os.path.join(args.output_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    # ── Dataset ────────────────────────────────────────────────────────
    full_dataset = SemiconductorDataset(
        degraded_dir=args.degraded_dir,
        gt_dir=args.gt_dir,
        patch_size=cfg.PATCH_SIZE,
        upscale=cfg.UPSCALE,
        augment=True,
        is_train=True)

    # Train/Val split
    val_size = max(1, int(cfg.VAL_SPLIT * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.SEED))

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False)

    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False)

    # ── Model ──────────────────────────────────────────────────────────
    model = build_swinir(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Loss ───────────────────────────────────────────────────────────
    use_perceptual = not args.no_perceptual
    criterion = CombinedLoss(
        pixel_weight=cfg.PIXEL_WEIGHT,
        perceptual_weight=cfg.PERCEPTUAL_WEIGHT,
        fft_weight=cfg.FFT_WEIGHT,
        edge_weight=cfg.EDGE_WEIGHT,
        use_perceptual=use_perceptual).to(device)

    # ── Optimizer ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.99))

    scheduler = get_lr_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Resume ─────────────────────────────────────────────────────────
    start_epoch = 0
    best_ssim = 0.0

    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_ssim = checkpoint.get('best_ssim', 0.0)
        print(f"Resumed from epoch {start_epoch}, best SSIM: {best_ssim:.4f}")

    # ── Print Training Info ────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  SwinIR Training — Semiconductor Image Restoration")
    print("═" * 60)
    print(f"  Device:       {device}")
    print(f"  Parameters:   {n_params / 1e6:.2f}M")
    print(f"  Train/Val:    {train_size} / {val_size}")
    print(f"  Batch size:   {cfg.BATCH_SIZE}")
    print(f"  Patch size:   {cfg.PATCH_SIZE} (LR) → {cfg.PATCH_SIZE * cfg.UPSCALE} (HR)")
    print(f"  Epochs:       {cfg.EPOCHS}")
    print(f"  LR:           {cfg.LEARNING_RATE} → {cfg.MIN_LR} (cosine)")
    print(f"  Perceptual:   {'enabled' if use_perceptual else 'disabled'}")
    print(f"  Output:       {args.output_dir}")
    print("═" * 60 + "\n")

    # ── Training Loop ──────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.EPOCHS):
        t_start = time.time()

        # Train
        train_loss, train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg,
            use_amp=use_amp)

        # Step scheduler
        scheduler.step()

        # Log training metrics
        writer.add_scalar('train/loss', train_loss, epoch)
        for k, v in train_losses.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        # Validate periodically
        if (epoch + 1) % cfg.VAL_EVERY == 0 or epoch == cfg.EPOCHS - 1:
            val_metrics = validate(model, val_loader, criterion, device)

            writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            writer.add_scalar('val/psnr', val_metrics['psnr'], epoch)
            writer.add_scalar('val/ssim', val_metrics['ssim'], epoch)

            # Save best model
            is_best = val_metrics['ssim'] > best_ssim
            if is_best:
                best_ssim = val_metrics['ssim']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_ssim': best_ssim,
                    'config': {k: v for k, v in vars(type(cfg)).items()
                               if not k.startswith('_') and k.isupper()},
                }, os.path.join(weights_dir, 'best_model.pt'))

            elapsed = time.time() - t_start
            best_marker = " ★ BEST" if is_best else ""
            print(f"Epoch [{epoch + 1:>3}/{cfg.EPOCHS}] | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val PSNR: {val_metrics['psnr']:.2f} dB | "
                  f"Val SSIM: {val_metrics['ssim']:.4f}{best_marker} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"{elapsed:.1f}s")
        else:
            elapsed = time.time() - t_start
            print(f"Epoch [{epoch + 1:>3}/{cfg.EPOCHS}] | "
                  f"Loss: {train_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"{elapsed:.1f}s")

        # Save checkpoint periodically
        if (epoch + 1) % cfg.SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_ssim': best_ssim,
            }, os.path.join(weights_dir, f'checkpoint_epoch{epoch + 1}.pt'))
            print(f"  → Checkpoint saved: checkpoint_epoch{epoch + 1}.pt")

    # ── Final Save ─────────────────────────────────────────────────────
    torch.save({
        'epoch': cfg.EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_ssim': best_ssim,
    }, os.path.join(weights_dir, 'final_model.pt'))

    # Copy best model to project root weights/
    project_weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
    os.makedirs(project_weights, exist_ok=True)
    best_src = os.path.join(weights_dir, 'best_model.pt')
    best_dst = os.path.join(project_weights, 'best_model.pt')
    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)
        print(f"\n✓ Best model copied to {best_dst}")

    writer.close()

    print(f"\n{'═' * 60}")
    print(f"  Training complete!")
    print(f"  Best SSIM: {best_ssim:.4f}")
    print(f"  Weights: {weights_dir}")
    print(f"{'═' * 60}\n")


if __name__ == '__main__':
    main()
