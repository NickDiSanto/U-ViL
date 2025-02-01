import torch
import torch.nn as nn
from vision_lstm import ViLLayer
from torchinfo import summary

class ConvStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 4, 1)  # BCDHW -> BDHWC
        x = self.norm1(x)
        x = x.permute(0, 4, 1, 2, 3)  # BDHWC -> BCDHW
        x = self.act(x)

        x = self.conv2(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm2(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.act(x)
        return x

class PatchExpanding(nn.Module):
    def __init__(self, dim: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.dim = dim
        self.expand = nn.Linear(dim, scale_factor * scale_factor * scale_factor * dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # BCDHW -> BDHWC
        x = self.expand(x)
        x = x.reshape(B, D, H, W, self.scale_factor, self.scale_factor, self.scale_factor, self.dim)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(B, self.dim, D * self.scale_factor, H * self.scale_factor, W * self.scale_factor)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x

class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UViL(nn.Module):
    def __init__(self,
                 in_channels: int,
                 base_channels: int,
                 num_classes: int,
                 vil_blocks_per_stage: int = 2):
        super().__init__()
        # Conv Stem
        self.stem = ConvStem(in_channels, base_channels)

        # Encoder Path
        self.encoder_stages = nn.ModuleList()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        for i in range(4):
            stage = nn.Sequential(*[
                ViLLayer(channels[i])
                for _ in range(vil_blocks_per_stage)
            ])
            if i < 3:  # No downsampling for last stage
                stage.append(
                    nn.Conv3d(channels[i], channels[i + 1],
                              kernel_size=2, stride=2)
                )
            self.encoder_stages.append(stage)

        # Decoder Path
        self.decoder_stages = nn.ModuleList()
        decoder_channels = channels[::-1]  # Reverse channel list

        for i in range(3):  # 3 decoder stages
            stage = nn.Sequential(
                PatchExpanding(decoder_channels[i]),
                *[ViLLayer(decoder_channels[i + 1])
                  for _ in range(vil_blocks_per_stage)]
            )
            self.decoder_stages.append(stage)

            # Segmentation Head
            self.seg_head = SegmentationHead(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial stem
        x = self.stem(x)

        # Store encoder outputs for skip connections
        encoder_features = []

        # Encoder path
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_features.append(x)

        # Decoder path with skip connections
        for i, stage in enumerate(self.decoder_stages):
            skip = encoder_features[-(i + 2)]  # Skip connection from encoder
            x = stage(x)
            x = x + skip  # Skip connection addition

        # Segmentation head
        out = self.seg_head(x)

        return out


if __name__ == "__main__":
    model = UViL(in_channels=1, base_channels=16, num_classes=2)
    summary(model, input_size=(1, 32, 32, 32), device='cpu')