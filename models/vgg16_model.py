import torch
import torch.nn as nn
from torchvision import models

from utils.config import Config


class VGG16Retinopathy(nn.Module):
    """
    VGG16 for Diabetic Retinopathy grading (5 classes).

    Freeze / Train strategy
    ────────────────────────────────────────────────────────────
    VGG16 feature extractor (features[0..30]):
        Block 1 (0–4)  : FROZEN — edges, corners
        Block 2 (5–9)  : FROZEN — textures
        Block 3 (10–16): FROZEN — patterns
        Block 4 (17–23): FROZEN — complex shapes
        Block 5 (24–30): TRAINABLE — high-level, fine-tuned
                         for retinopathy-specific patterns

    Added between backbone and head:
        BatchNorm1d(25088) — stabilises feature scale
        Dropout(0.4)       — regularisation before the head

    Custom classifier head (replaces VGG16's original):
        FC(25088 → 2048) → BatchNorm → ReLU → Dropout(0.5)
        FC(2048  →    5)
    ────────────────────────────────────────────────────────────
    Changes vs original VGG16:
        1. Blocks 1–4 frozen permanently
        2. Block 5 trainable from epoch 1 (no delayed unfreeze)
        3. BatchNorm + Dropout injected after flatten
        4. Head slimmed: 25088→2048→5 (vs original 25088→4096→4096→1000)
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, dropout=Config.DROPOUT):
        super().__init__()

        # ── Load pretrained backbone ──────────────────────────
        backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = backbone.features   # conv blocks 1–5
        self.avgpool  = backbone.avgpool    # AdaptiveAvgPool2d(7,7)

        # ── Freeze Blocks 1–4 (features[0:24]) ───────────────
        for layer in self.features[:24]:
            for param in layer.parameters():
                param.requires_grad = False

        # ── Keep Block 5 (features[24:]) trainable ───────────
        for layer in self.features[24:]:
            for param in layer.parameters():
                param.requires_grad = True

        # ── NEW: bridge layers between backbone and head ──────
        # Stabilise the 25088-dim feature vector before the FC head
        self.bridge = nn.Sequential(
            nn.BatchNorm1d(512 * 7 * 7),   # normalise across feature dims
            nn.Dropout(p=0.4),             # light dropout before head
        )

        # ── Custom head — slimmer than original VGG16 head ───
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)          # conv blocks 1–5
        x = self.avgpool(x)           # 512×7×7
        x = torch.flatten(x, 1)       # 25088
        x = self.bridge(x)            # BatchNorm + Dropout
        x = self.classifier(x)        # 5 logits
        return x

    def count_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        return total, trainable, frozen

    def param_summary(self):
        total, trainable, frozen = self.count_params()
        print(f"  Total params     : {total:>12,}")
        print(f"  Trainable params : {trainable:>12,}")
        print(f"  Frozen params    : {frozen:>12,}")

    def freeze_summary(self):
        """Print layer-by-layer trainable status."""
        print(f"  {'Layer':<45} {'Trainable':>10}")
        print("  " + "-" * 57)
        for name, param in self.named_parameters():
            status = "YES" if param.requires_grad else "frozen"
            print(f"  {name:<45} {status:>10}")