"""
Configuration for SwinIR-based Semiconductor Image Restoration.
All hyperparameters in one place for easy tuning.

Optimized for ~5 hours training on Apple Silicon MPS (16GB RAM).
"""


class Config:
    """Central configuration for model, training, and loss parameters."""

    # ── Model Architecture ──────────────────────────────────────────────
    #  Medium-light: 4 RSTB blocks, 96-dim embedding (~3.34M params)
    IN_CHANNELS = 1              # Grayscale images
    EMBED_DIM = 96               # Swin Transformer embedding dimension
    DEPTHS = [6, 6, 6, 6]       # Number of STL layers per RSTB block
    NUM_HEADS = [6, 6, 6, 6]    # Attention heads per RSTB block
    WINDOW_SIZE = 8              # Shifted-window attention window size
    MLP_RATIO = 4.0              # MLP hidden dim = embed_dim * mlp_ratio
    UPSCALE = 2                  # Super-resolution upscale factor
    IMG_RANGE = 1.0              # Input normalization range
    RESI_CONNECTION = '1conv'    # Residual connection type

    # ── Training ────────────────────────────────────────────────────────
    #  batch=8 → 400 steps/epoch → ~8.5 min/epoch → ~30 epochs in 5 hours
    BATCH_SIZE = 8               # Larger batch = more stable gradients
    PATCH_SIZE = 64              # LR patch size; GT patch = 128
    EPOCHS = 30                  # Fits in ~5 hours on Apple Silicon MPS
    LEARNING_RATE = 2e-4         # Initial learning rate
    MIN_LR = 1e-6                # Cosine annealing minimum LR
    WEIGHT_DECAY = 1e-4          # AdamW weight decay
    WARMUP_EPOCHS = 2            # Linear warmup epochs
    SEED = 42                    # Random seed
    NUM_WORKERS = 2              # DataLoader workers (2 for Mac)
    GRAD_CLIP = 1.0              # Gradient clipping max norm
    VAL_SPLIT = 0.1              # 10% for validation

    # ── Loss Weights ────────────────────────────────────────────────────
    PIXEL_WEIGHT = 1.0           # Charbonnier pixel loss
    PERCEPTUAL_WEIGHT = 0.1      # VGG perceptual loss (disabled with --no_perceptual)
    FFT_WEIGHT = 0.1             # Frequency-domain (FFT) loss
    EDGE_WEIGHT = 0.05           # Sobel edge-aware loss

    # ── Checkpointing & Logging ─────────────────────────────────────────
    VAL_EVERY = 5                # Validate every N epochs
    SAVE_EVERY = 5               # Save checkpoint every 5 epochs (frequent for short training)
    LOG_EVERY = 50               # Log training metrics every N batches

    # ── Data Augmentation ───────────────────────────────────────────────
    AUGMENT_FLIP = True          # Random horizontal/vertical flips
    AUGMENT_ROTATE = True        # Random 90° rotations
    AUGMENT_MIXUP = False        # Mixup augmentation
    AUGMENT_INTENSITY = True     # Random intensity scaling ±10%

    def __repr__(self):
        attrs = {k: v for k, v in vars(type(self)).items()
                 if not k.startswith('_') and k.isupper()}
        lines = [f"  {k} = {v}" for k, v in sorted(attrs.items())]
        return "Config(\n" + "\n".join(lines) + "\n)"
