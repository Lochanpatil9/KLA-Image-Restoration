"""
SwinIR: Image Restoration Using Swin Transformer
─────────────────────────────────────────────────
Adapted for grayscale semiconductor inspection image restoration.
Handles joint denoising (speckle + Gaussian) and 2× super-resolution.

Architecture:
  1. Shallow Feature Extraction (3×3 Conv)
  2. Deep Feature Extraction (RSTB blocks with Swin Transformer Layers)
  3. Image Reconstruction (PixelShuffle 2× + 3×3 Conv)

Reference: Liang et al., "SwinIR: Image Restoration Using Swin Transformer", ICCVW 2021
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
#  Utility modules
# ═══════════════════════════════════════════════════════════════════════

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic depth — drop entire residual branches during training."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """MLP block used inside each Swin Transformer layer."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Window partition / reverse
# ═══════════════════════════════════════════════════════════════════════

def window_partition(x, window_size: int):
    """Partition feature map into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """Reverse window partition back to feature map.

    Args:
        windows: (num_windows * B, window_size, window_size, C)
        window_size: int
        H, W: original spatial dimensions
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ═══════════════════════════════════════════════════════════════════════
#  Window-based Multi-head Self-Attention (W-MSA / SW-MSA)
# ═══════════════════════════════════════════════════════════════════════

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias.

    Supports both regular and shifted window partitions via the attention mask.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table: (2*Wh-1) * (2*Ww-1), num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C) where N = window_size^2
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B_, num_heads, N, head_dim)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, num_heads, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply attention mask for shifted windows
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Swin Transformer Block
# ═══════════════════════════════════════════════════════════════════════

class SwinTransformerBlock(nn.Module):
    """A single Swin Transformer block: W-MSA or SW-MSA + FFN.

    Alternating blocks use shift_size=0 (regular window) and
    shift_size=window_size//2 (shifted window) to enable cross-window connections.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4.0, qkv_bias=True, drop=0.0,
                 attn_drop=0.0, drop_path_val=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Adjust window/shift if resolution is smaller than window
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size),
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_val) if drop_path_val > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        # Pre-compute attention mask for shifted window partition
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def _compute_attn_mask(self, H, W, device):
        """Dynamically compute attention mask for given spatial dimensions."""
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1), device=device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
            return attn_mask
        return None

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Determine correct attention mask: use pre-computed if resolution
        # matches, otherwise compute dynamically (handles arbitrary sizes)
        if (H, W) == tuple(self.input_resolution):
            attn_mask = self.attn_mask
        else:
            attn_mask = self._compute_attn_mask(H, W, x.device)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows back
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Residual + FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ═══════════════════════════════════════════════════════════════════════
#  Basic Layer (a group of Swin Transformer Blocks)
# ═══════════════════════════════════════════════════════════════════════

class BasicLayer(nn.Module):
    """A group of Swin Transformer blocks with alternating window shifts."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path_val=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Patch Embed / Unembed
# ═══════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """Image to Patch Embedding (or feature to sequence)."""

    def __init__(self, img_size=None, patch_size=1, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        if img_size is not None:
            img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
            patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.img_size = img_size
            self.patches_resolution = patches_resolution
            self.num_patches = patches_resolution[0] * patches_resolution[1]
        else:
            self.img_size = None
            self.patches_resolution = None
            self.num_patches = None

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if in_chans > 0:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=patch_size)
        else:
            self.proj = None

        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Sequence back to feature map."""

    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


# ═══════════════════════════════════════════════════════════════════════
#  RSTB: Residual Swin Transformer Block
# ═══════════════════════════════════════════════════════════════════════

class RSTB(nn.Module):
    """Residual Swin Transformer Block.

    A BasicLayer (group of STL blocks) wrapped in a residual connection
    with a convolution for feature refinement.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, resi_connection='1conv'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim, input_resolution=input_resolution,
            depth=depth, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(img_size=None, patch_size=1, in_chans=0,
                                       embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def forward(self, x, x_size):
        return self.patch_embed(
            self.conv(
                self.patch_unembed(self.residual_group(x, x_size), x_size)
            )
        ) + x


# ═══════════════════════════════════════════════════════════════════════
#  Upsampling
# ═══════════════════════════════════════════════════════════════════════

class Upsample(nn.Sequential):
    """PixelShuffle-based upsampling module.

    Supports 2× and 4× (power-of-2) and 3× scale factors.
    """

    def __init__(self, scale, num_feat):
        layers = []
        if (scale & (scale - 1)) == 0:  # power of 2
            for _ in range(int(math.log2(scale))):
                layers.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            layers.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Scale {scale} not supported. Use power-of-2 or 3.")
        super().__init__(*layers)


# ═══════════════════════════════════════════════════════════════════════
#  SwinIR: Full Model
# ═══════════════════════════════════════════════════════════════════════

class SwinIR(nn.Module):
    """SwinIR for joint denoising and super-resolution.

    Takes a degraded grayscale image and outputs a clean, upscaled image.

    Architecture:
        Shallow Feature Extraction → Deep Feature Extraction (RSTB) →
        High-Quality Image Reconstruction (PixelShuffle)

    Args:
        img_size (int): Input image size for training (used for position bias).
        patch_size (int): Patch embedding size (default 1 = no patching).
        in_chans (int): Number of input channels (1 for grayscale).
        embed_dim (int): Transformer embedding dimension.
        depths (list[int]): Number of STL blocks in each RSTB.
        num_heads (list[int]): Number of attention heads in each RSTB.
        window_size (int): Window size for shifted-window attention.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        qkv_bias (bool): Add learnable bias to Q, K, V projections.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        upscale (int): Upsampling scale factor.
        img_range (float): Image value range for normalization.
        resi_connection (str): Residual connection type: '1conv' or '3conv'.
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=1,
                 embed_dim=180, depths=(6, 6, 6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6, 6, 6), window_size=8,
                 mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 upscale=2, img_range=1.0, resi_connection='1conv',
                 **kwargs):
        super().__init__()

        self.img_range = img_range
        self.upscale = upscale
        self.window_size = window_size
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64  # intermediate feature channels for reconstruction

        # ── Mean for channel-wise normalization ─────────────────────────
        self.mean = nn.Parameter(torch.zeros(1, 1, 1, 1), requires_grad=False)

        # ── 1. Shallow Feature Extraction ───────────────────────────────
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ── 2. Deep Feature Extraction ──────────────────────────────────
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # Patch embed (for converting conv features → sequence)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Patch unembed (sequence → conv features)
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build RSTB blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                resi_connection=resi_connection)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # ── 3. High-Quality Image Reconstruction ───────────────────────
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
            nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # ── Initialize weights ─────────────────────────────────────────
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        """Pad input to be divisible by window_size."""
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        """Deep feature extraction via RSTB stack."""
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # Normalize
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # Shallow features
        x = self.conv_first(x)

        # Deep features with global residual
        x = self.conv_after_body(self.forward_features(x)) + x

        # Reconstruction
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)

        # Denormalize
        x = x / self.img_range + self.mean

        return x[:, :, :H * self.upscale, :W * self.upscale]


# ═══════════════════════════════════════════════════════════════════════
#  Factory function
# ═══════════════════════════════════════════════════════════════════════

def build_swinir(config=None):
    """Build SwinIR model from config object.

    Args:
        config: Config object with model parameters. If None, uses defaults.

    Returns:
        SwinIR model instance.
    """
    if config is None:
        from config import Config
        config = Config()

    model = SwinIR(
        img_size=config.PATCH_SIZE,
        patch_size=1,
        in_chans=config.IN_CHANNELS,
        embed_dim=config.EMBED_DIM,
        depths=config.DEPTHS,
        num_heads=config.NUM_HEADS,
        window_size=config.WINDOW_SIZE,
        mlp_ratio=config.MLP_RATIO,
        upscale=config.UPSCALE,
        img_range=config.IMG_RANGE,
        resi_connection=config.RESI_CONNECTION,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SwinIR created — {n_params / 1e6:.2f}M trainable parameters")

    return model


if __name__ == '__main__':
    # Quick sanity check
    model = build_swinir()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
    assert y.shape == (1, 1, 128, 128), f"Expected (1,1,128,128), got {y.shape}"
    print("✓ SwinIR forward pass OK")
