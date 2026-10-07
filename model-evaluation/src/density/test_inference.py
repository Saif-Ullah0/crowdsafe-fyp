import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Define CSRNet Architecture
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.frontend = self._make_layers(self.frontend_feat)
        self.backend  = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        d_rate = 2 if dilation else 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

# Initialize Model
model = CSRNet()
model.eval()

# Preprocessing Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test on one generated sample image
img_path = 'datasets/ShanghaiTech/part_A_final/train_data/images/IMG_1.jpg'
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0)

# Perform CPU Forward Pass
with torch.no_grad():
    output_density = model(img_tensor)
    predicted_count = output_density.sum().item()

print("=" * 45)
print("CSRNet Model Architecture Loaded Successfully!")
print(f"Input Image Tensor Shape : {img_tensor.shape}")
print(f"Output Density Map Shape : {output_density.shape}")
print(f"Untrained Model Estimated Count: {predicted_count:.2f}")
print("=" * 45)