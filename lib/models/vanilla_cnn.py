# /content/CvT/lib/models/vanilla_cnn.py

import torch
import torch.nn as nn
from .builder import MODEL_REGISTRY # Corrected import

# --- Basic CNN Block ---
class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Identity mapping for residual connection: handles dimension mismatch (stride/channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x) # Apply downsample to identity first
        out = self.conv(x)
        return out + identity # Residual connection

# --- Vanilla CNN Model ---
@MODEL_REGISTRY.register_module
class VanillaCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        num_classes = config.MODEL.NUM_CLASSES
        depths = config.CNN_SPEC.DEPTHS
        channels = config.CNN_SPEC.CHANNELS
        
        # Initial input stage: 7x7 Conv, BatchNorm, ReLU, and MaxPool (typical ResNet stem)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        in_c = channels[0]
        self.stages = nn.ModuleList()
        
        # Build sequential stages
        for i in range(len(depths)):
            out_c = channels[i + 1]
            num_blocks = depths[i]
            
            # First block in a stage handles downsampling (if i > 0)
            blocks = [BasicConvBlock(in_c, out_c, stride=2 if i > 0 else 1)]
            in_c = out_c
            
            # Remaining blocks in the stage
            for _ in range(1, num_blocks):
                blocks.append(BasicConvBlock(in_c, out_c, stride=1))
                
            self.stages.append(nn.Sequential(*blocks))

        # Classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(channels[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Flatten for the linear head
        x = self.head(x)
        return x
