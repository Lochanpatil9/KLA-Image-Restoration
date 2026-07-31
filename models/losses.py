"""
Custom loss functions for semiconductor image restoration.
─────────────────────────────────────────────────────────────
Combined loss: Charbonnier (pixel) + Perceptual (VGG) + FFT (frequency) + Edge (Sobel)

Design rationale:
  - Charbonnier: Robust to intensity outliers from speckle noise (better than L1/MSE)
  - Perceptual: Captures structural similarity via VGG features
  - FFT: Preserves frequency content — critical for periodic semiconductor patterns
  - Edge: Preserves sharp edges needed for defect detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1 variant).

    L = sqrt((pred - target)^2 + eps^2)

    Advantages over plain L1:
    - Differentiable everywhere (including at 0)
    - Robust to outliers caused by speckle noise
    - Encourages sharper outputs than MSE
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps_sq = eps * eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return loss.mean()


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 feature maps.

    Compares intermediate feature representations rather than raw pixels,
    capturing structural and textural similarity.

    Uses layers: relu1_2, relu2_2, relu3_4, relu4_4
    Grayscale images are expanded to 3 channels for VGG compatibility.
    """

    def __init__(self, layer_weights=None):
        super().__init__()
        import torchvision.models as models

        # Load pretrained VGG19 — try new API first, fall back to legacy
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        except (AttributeError, TypeError):
            vgg = models.vgg19(pretrained=True).features

        # Extract feature layers
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])    # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18])  # relu3_4
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27]) # relu4_4

        # Freeze VGG — we only use it for feature extraction
        for param in self.parameters():
            param.requires_grad = False

        self.weights = layer_weights or [0.25, 0.25, 0.25, 0.25]

        # VGG normalization (ImageNet stats)
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        """Normalize for VGG: expand grayscale to 3ch, apply ImageNet stats."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.vgg_mean) / self.vgg_std

    def forward(self, pred, target):
        pred = self._normalize(pred)
        target = self._normalize(target)

        f1_pred = self.slice1(pred)
        f2_pred = self.slice2(f1_pred)
        f3_pred = self.slice3(f2_pred)
        f4_pred = self.slice4(f3_pred)

        with torch.no_grad():
            f1_target = self.slice1(target)
            f2_target = self.slice2(f1_target)
            f3_target = self.slice3(f2_target)
            f4_target = self.slice4(f3_target)

        loss = (self.weights[0] * F.l1_loss(f1_pred, f1_target) +
                self.weights[1] * F.l1_loss(f2_pred, f2_target) +
                self.weights[2] * F.l1_loss(f3_pred, f3_target) +
                self.weights[3] * F.l1_loss(f4_pred, f4_target))
        return loss


class FFTLoss(nn.Module):
    """Frequency-domain loss using 2D FFT.

    Compares magnitude and phase spectra of predicted and target images.
    Critical for semiconductor images which have strong periodic patterns
    (e.g., regular array structures, gratings).
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # Magnitude loss (amplitude spectrum)
        mag_loss = F.l1_loss(pred_fft.abs(), target_fft.abs())

        # Phase loss
        phase_loss = F.l1_loss(
            torch.angle(pred_fft),
            torch.angle(target_fft)
        )

        return self.loss_weight * (mag_loss + phase_loss)


class EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel filters.

    Applies Sobel operators in X and Y directions to extract edges,
    then computes L1 loss between predicted and target edge maps.
    Ensures that the model preserves sharp boundaries critical for
    semiconductor defect detection.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                 [0,  0,  0],
                                 [1,  2,  1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        # Extract edges
        pred_edge_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_edge_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target, self.sobel_y, padding=1)

        loss = F.l1_loss(pred_edge_x, target_edge_x) + F.l1_loss(pred_edge_y, target_edge_y)
        return loss


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss (1 - SSIM).

    Can optionally be added to the combined loss for direct SSIM optimization.
    """

    def __init__(self, window_size=11, channel=1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)

    def _create_window(self, window_size, channel):
        """Create Gaussian window for SSIM computation."""
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size)
        return window

    def forward(self, pred, target):
        window = self.window.to(pred.device).type_as(pred)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2,
                             groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2,
                             groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2,
                           groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """Combined loss for semiconductor image restoration.

    L_total = λ₁·L_pixel + λ₂·L_perceptual + λ₃·L_fft + λ₄·L_edge

    Returns total loss and a dictionary of individual loss components
    for logging/monitoring.
    """

    def __init__(self, pixel_weight=1.0, perceptual_weight=0.1,
                 fft_weight=0.1, edge_weight=0.05, use_perceptual=True):
        super().__init__()
        self.pixel_loss = CharbonnierLoss()
        self.fft_loss = FFTLoss()
        self.edge_loss = EdgeLoss()

        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.fft_weight = fft_weight
        self.edge_weight = edge_weight

        # Perceptual loss requires VGG download — make it optional
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()

    def forward(self, pred, target):
        loss_pixel = self.pixel_loss(pred, target)
        loss_fft = self.fft_loss(pred, target)
        loss_edge = self.edge_loss(pred, target)

        total = (self.pixel_weight * loss_pixel +
                 self.fft_weight * loss_fft +
                 self.edge_weight * loss_edge)

        loss_dict = {
            'pixel': loss_pixel.item(),
            'fft': loss_fft.item(),
            'edge': loss_edge.item(),
        }

        if self.use_perceptual:
            loss_perceptual = self.perceptual_loss(pred, target)
            total = total + self.perceptual_weight * loss_perceptual
            loss_dict['perceptual'] = loss_perceptual.item()

        loss_dict['total'] = total.item()

        return total, loss_dict
