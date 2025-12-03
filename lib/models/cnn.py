# lib/models/vanilla_cnn.py

import torch
import torch.nn as nn
from .builder import MODEL_REGISTRY

# A simple block using Conv, BatchNorm, and ReLU (like a minimal ResNet block)
class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Optional: identity mapping for downsampling, required for residual connections
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity # Add residual connection

@MODEL_REGISTRY.register_module
class VanillaCNN(nn.Module):
    """
    A minimal CNN model (ResNet-like structure) that uses the CvT framework.
    It takes configuration for total depth and channels but ignores CvT-specific parameters.
    """
    def __init__(self, config):
        super().__init__()
        
        # Extract general CNN parameters from the config
        num_classes = config.MODEL.NUM_CLASSES
        depths = config.CNN_SPEC.DEPTHS
        channels = config.CNN_SPEC.CHANNELS
        
        # Initial input stage (7x7 Conv)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        in_c = channels[0]
        self.stages = nn.ModuleList()
        
        # Build sequential stages (similar to ResNet stages)
        for i in range(len(depths)):
            out_c = channels[i + 1]
            num_blocks = depths[i]
            
            # First block in a stage may have stride 2 for downsampling
            blocks = [BasicConvBlock(in_c, out_c, stride=2 if i > 0 else 1)]
            in_c = out_c
            
            # Remaining blocks in the stage have stride 1
            for _ in range(1, num_blocks):
                blocks.append(BasicConvBlock(in_c, out_c, stride=1))
                
            self.stages.append(nn.Sequential(*blocks))

        # Classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(channels[-1], num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        # Basic weight initialization (optional, but good practice)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x is assumed to be (B, 3, H, W)
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
