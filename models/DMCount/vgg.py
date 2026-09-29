import torch
import torch.nn as nn
from torchvision import models


class VGG19_DMCount(nn.Module):
    def __init__(self, pretrained=False):
        super(VGG19_DMCount, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT if pretrained else None)

        # Features extractor (conv1_1 to conv4_4)
        self.features = vgg.features[:27]

        # Regressor head for density map generation
        self.reg_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.reg_head(x)
        return x, None