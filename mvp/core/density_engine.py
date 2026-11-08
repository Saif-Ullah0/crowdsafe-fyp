"""
Plug-and-Play Density Estimation Engine for CrowdSafe MVP
Provides a modular abstraction layer supporting:
1. FastSimModel: Lightweight procedural density simulator (<5ms latency)
2. MCNNModel: Multi-Column CNN for multi-scale perspective crowd counting
3. CSRNetModel: Dilated convolutional architecture for congested scenes
"""
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BaseDensityModel(ABC):
    """Abstract Base Class for crowd density estimation models."""

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Runs density estimation inference on an input frame.
        
        Args:
            frame_bgr: np.ndarray of shape (H, W, 3)
            
        Returns:
            Dict containing:
                - 'total_count': float (estimated head count)
                - 'density_map': np.ndarray (2D density distribution)
                - 'latency_ms': float (inference time in milliseconds)
                - 'model_name': str
        """
        pass


class FastSimModel(BaseDensityModel):
    """
    High-speed procedural density engine for local testing and real-time development.
    Extracts high-contrast regions (people/heads) and applies Gaussian spatial smoothing.
    Runs with ~2ms latency on CPU without requiring deep learning weights.
    """
    def __init__(self):
        self.model_name = "FastSim-Engine"

    def predict(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        start = time.time()
        import cv2
        h, w = frame_bgr.shape[:2]
        out_h, out_w = h // 4, w // 4

        # Convert to grayscale and detect human-scale spatial contrast
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
        
        # Adaptive contrast and Laplacian edge detection
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Apply Gaussian blur to create continuous spatial density field
        density_field = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 4.0)
        
        # Scale to realistic headcount values
        sum_val = np.sum(density_field)
        if sum_val > 0:
            # Calibrate to realistic crowd headcount range (e.g., 20 - 150 people)
            head_factor = float(np.count_nonzero(edges > 0)) / 22.0
            density_map = (density_field / sum_val) * max(head_factor, 15.0)
        else:
            density_map = np.zeros((out_h, out_w), dtype=np.float32)

        total_count = float(np.sum(density_map))
        latency = (time.time() - start) * 1000.0

        return {
            "total_count": round(total_count, 1),
            "density_map": density_map,
            "latency_ms": round(latency, 2),
            "model_name": self.model_name
        }


if TORCH_AVAILABLE:
    class Conv2dBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, same_padding: bool = True):
            super().__init__()
            padding = int((kernel_size - 1) / 2) if same_padding else 0
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.conv(x))


    class MCNNNet(nn.Module):
        """
        Multi-Column Convolutional Neural Network (MCNN)
        Scale-invariant branches for large, medium, and small heads.
        """
        def __init__(self):
            super().__init__()
            # Branch 1 (Large filters: 9x9, 7x7)
            self.branch1 = nn.Sequential(
                Conv2dBlock(3, 16, 9), nn.MaxPool2d(2),
                Conv2dBlock(16, 32, 7), nn.MaxPool2d(2),
                Conv2dBlock(32, 16, 7), Conv2dBlock(16, 8, 7)
            )
            # Branch 2 (Medium filters: 7x7, 5x5)
            self.branch2 = nn.Sequential(
                Conv2dBlock(3, 20, 7), nn.MaxPool2d(2),
                Conv2dBlock(20, 40, 5), nn.MaxPool2d(2),
                Conv2dBlock(40, 20, 5), Conv2dBlock(20, 10, 5)
            )
            # Branch 3 (Small filters: 5x5, 3x3)
            self.branch3 = nn.Sequential(
                Conv2dBlock(3, 24, 5), nn.MaxPool2d(2),
                Conv2dBlock(24, 48, 3), nn.MaxPool2d(2),
                Conv2dBlock(48, 24, 3), Conv2dBlock(24, 12, 3)
            )
            self.fuse = nn.Sequential(Conv2dBlock(30, 1, 1))

        def forward(self, x):
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            x_cat = torch.cat((x1, x2, x3), 1)
            return self.fuse(x_cat)


    class MCNNModel(BaseDensityModel):
        """MCNN inference wrapper with PyTorch runtime."""
        def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
            self.model_name = "MCNN (Multi-Column CNN)"
            self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
            self.model = MCNNNet().to(self.device)
            self.fallback = FastSimModel()
            self.has_weights = False

            if weights_path and os.path.exists(weights_path):
                try:
                    try:
                        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                    except TypeError:
                        checkpoint = torch.load(weights_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        self.model.load_state_dict(checkpoint["state_dict"])
                    elif isinstance(checkpoint, dict):
                        self.model.load_state_dict(checkpoint)
                    self.model.eval()
                    self.has_weights = True
                    print(f"[MCNNModel] Successfully loaded weights: {weights_path}")
                except Exception as e:
                    print(f"[MCNNModel] Error loading checkpoint ({e}). Running in simulation mode.")
            else:
                print(f"[MCNNModel] No weights found at '{weights_path}'. Using FastSim backend.")

        def predict(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
            if not self.has_weights:
                res = self.fallback.predict(frame_bgr)
                res["model_name"] = "MCNN (FastSim Fallback)"
                return res

            start = time.time()
            img = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0  # BGR to RGB
            # Normalize with ImageNet standard
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                out = self.model(tensor)
                density_map = out.squeeze().cpu().numpy()
                density_map = np.maximum(density_map, 0.0)
                total_count = float(np.sum(density_map))

            latency = (time.time() - start) * 1000.0
            return {
                "total_count": round(total_count, 1),
                "density_map": density_map,
                "latency_ms": round(latency, 2),
                "model_name": self.model_name
            }


    class CSRNetNet(nn.Module):
        """
        CSRNet: Dilated Convolutional Neural Network for Congested Scene Recognition.
        Frontend based on VGG-16 first 10 conv layers + dilated conv backend.
        """
        def __init__(self):
            super().__init__()
            self.frontend = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)
            )
            self.backend = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True)
            )
            self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        def forward(self, x):
            x = self.frontend(x)
            x = self.backend(x)
            x = self.output_layer(x)
            return x


    class CSRNetModel(BaseDensityModel):
        """CSRNet dilated convolutional model wrapper."""
        def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
            self.model_name = "CSRNet (Dilated CNN)"
            self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
            self.model = CSRNetNet().to(self.device)
            self.fallback = FastSimModel()
            self.has_weights = False

            if weights_path and os.path.exists(weights_path):
                try:
                    try:
                        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                    except TypeError:
                        checkpoint = torch.load(weights_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        self.model.load_state_dict(checkpoint["state_dict"])
                    elif isinstance(checkpoint, dict):
                        self.model.load_state_dict(checkpoint)
                    self.model.eval()
                    self.has_weights = True
                    print(f"[CSRNetModel] Successfully loaded weights: {weights_path}")
                except Exception as e:
                    print(f"[CSRNetModel] Error loading checkpoint ({e}). Running in simulation mode.")
            else:
                print(f"[CSRNetModel] No weights found at '{weights_path}'. Using FastSim backend.")

        def predict(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
            if not self.has_weights:
                res = self.fallback.predict(frame_bgr)
                res["model_name"] = "CSRNet (FastSim Fallback)"
                return res

            start = time.time()
            img = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0  # BGR to RGB
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                out = self.model(tensor)
                density_map = out.squeeze().cpu().numpy()
                density_map = np.maximum(density_map, 0.0)
                total_count = float(np.sum(density_map))

            latency = (time.time() - start) * 1000.0
            return {
                "total_count": round(total_count, 1),
                "density_map": density_map,
                "latency_ms": round(latency, 2),
                "model_name": self.model_name
            }


def get_density_engine(
    architecture: str = "fast_sim",
    weights_path: Optional[str] = None,
    device: str = "cpu"
) -> BaseDensityModel:
    """
    Factory function for plug-and-play density estimation models.
    
    Args:
        architecture: 'fast_sim', 'mcnn', or 'csrnet'
        weights_path: Path to .pth weight file (optional)
        device: 'cpu' or 'cuda'
        
    Returns:
        Instance of BaseDensityModel
    """
    arch = architecture.lower().strip()
    if arch == "fast_sim" or not TORCH_AVAILABLE:
        return FastSimModel()
    elif arch == "mcnn":
        return MCNNModel(weights_path=weights_path, device=device)
    elif arch == "csrnet":
        return CSRNetModel(weights_path=weights_path, device=device)
    else:
        print(f"[get_density_engine] Unknown architecture '{architecture}'. Defaulting to FastSim.")
        return FastSimModel()
