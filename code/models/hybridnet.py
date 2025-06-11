import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import ResidualBlock, DenseBlock, CBAM

class HybridBranch(nn.Module):
    def __init__(self, in_channels, base_channels=24, num_blocks=2, growth_rate=24, num_layers=2):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels, base_channels)
              for _ in range(num_blocks)]
        )

        self.dense_block = DenseBlock(
            in_channels=base_channels,
            growth_rate=growth_rate,
            num_layers=num_layers
        )

        self.out_channels = base_channels + growth_rate * num_layers
        self.cbam = CBAM(self.out_channels)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.dense_block(x)
        x = self.cbam(x)
        return x

class MultiScalePyramid(nn.Module):
    def __init__(self, scales, in_channels):
        super().__init__()
        self.scales = sorted(scales, reverse=True)
        self.down_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            ) for _ in range(len(scales) - 1)
        ])

        self.up_convs = nn.ModuleList()
        for i in range(len(scales) - 1):
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU()
                )
            )

        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        features = [x]
        for down_conv in self.down_convs:
            features.append(down_conv(features[-1]))

        x = features[-1]
        for i in reversed(range(len(self.up_convs))):
            target_size = features[i].shape[-2:]
            x = F.interpolate(x, size=target_size, mode='bilinear')
            x = torch.cat([x, features[i]], dim=1)
            x = self.up_convs[i](x)

        return self.fusion_conv(x)

class HybridNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.branch_rgb = HybridBranch(in_channels=3, base_channels=24, num_layers=2)
        self.branch_heatmap = HybridBranch(in_channels=1, base_channels=24, num_layers=2)
        self.pyramid = MultiScalePyramid(scales=args.pyramid_scales, in_channels=144)
        self.head = nn.Sequential(
            nn.Conv2d(144, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 1, 1)
        )

    def forward(self, x):
        rgb_feat = self.branch_rgb(x[:, :3])
        heat_feat = self.branch_heatmap(x[:, 3:])
        fused_feat = torch.cat([rgb_feat, heat_feat], dim=1)
        pyramid_feat = self.pyramid(fused_feat)
        return self.head(pyramid_feat) 