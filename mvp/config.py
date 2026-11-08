"""
CrowdSafe MVP Configuration File
Defines default system parameters, safety thresholds, ROI zones, model paths,
streaming controls, and webhook notification settings.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Safety & Density Thresholds (People per m²)
# Based on Fruin's Level of Service & International Crowd Dynamics
# ---------------------------------------------------------
DENSITY_LEVELS = {
    "LOW": {
        "max": 0.1,  # <--- Changed from 2.0
        "color": (0, 255, 0),
        "label": "Normal Flow",
        "description": "Free movement, normal walking speed"
    },
    "MODERATE": {
        "max": 0.2,  # <--- Changed from 3.5
        "color": (0, 255, 255),
        "label": "Caution / High Volume",
        "description": "Constrained movement, walking speed restricted"
    },
    "HIGH": {
        "max": 0.25, # <--- Changed from 5.0
        "color": (0, 165, 255),
        "label": "Warning / Bottleneck",
        "description": "Crowd compression begins, involuntary contact"
    },
    "CRITICAL": {
        "max": float("inf"),
        "color": (0, 0, 255),
        "label": "Critical / Crush Risk",
        "description": "Extreme compressive asphyxiation / stampede risk"
    }
}
# ---------------------------------------------------------
# Default Zone / Region of Interest (ROI) Definitions
# Normalized coordinates (x1, y1, x2, y2) in range [0.0, 1.0]
# ---------------------------------------------------------
DEFAULT_ZONES = [
    {
        "id": "zone_entry",
        "name": "Entrance Gateway",
        "bbox": (0.05, 0.1, 0.45, 0.9),  # (x_min, y_min, x_max, y_max)
        "estimated_area_m2": 25.0
    },
    {
        "id": "zone_corridor",
        "name": "Central Concourse",
        "bbox": (0.45, 0.1, 0.85, 0.9),
        "estimated_area_m2": 35.0
    },
    {
        "id": "zone_exit",
        "name": "Emergency Exit Route",
        "bbox": (0.85, 0.2, 0.98, 0.8),
        "estimated_area_m2": 15.0
    }
]

# ---------------------------------------------------------
# Streaming & DVR Ingestion Settings
# ---------------------------------------------------------
@dataclass
class StreamConfig:
    source: str = "sample"          # 'sample', directory path, video file, or camera index
    target_fps: float = 1.0        # Default 1 FPS CCTV/DVR polling rate
    loop: bool = True              # Loop image sequences or video clips
    max_frames: int = None         # None = unlimited, or int limit
    frame_size: Tuple[int, int] = (480, 640)  # (H, W)

# ---------------------------------------------------------
# Model & Inference Settings
# ---------------------------------------------------------
@dataclass
class ModelConfig:
    architecture: str = "fast_sim"  # Options: 'mcnn', 'csrnet', 'fast_sim'
    weights_path: str = str(BASE_DIR.parent / "models" / "MCNN" / "mcnn_mall_best.pth")
    device: str = "cpu"            # 'cpu' or 'cuda'
    input_size: Tuple[int, int] = (480, 640)  # (H, W)

# ---------------------------------------------------------
# Alert & Webhook Dispatch Settings
# ---------------------------------------------------------
@dataclass
class AlertConfig:
    consecutive_frames_trigger: int = 3      # Standard hysteresis for HIGH/CRITICAL
    instant_critical_threshold: float = 5.5  # Instant trigger (bypass hysteresis) for flash surges
    alert_log_file: Path = OUTPUT_DIR / "alerts.jsonl"
    enable_console_logging: bool = True
    webhook_url: str = "http://localhost:3000/api/webhook"  # <--- UPDATE THIS LINE
    webhook_timeout: float = 2.0            # Seconds before non-blocking timeout
# ---------------------------------------------------------
# Visualization & GUI Settings
# ---------------------------------------------------------
@dataclass
class VisualizerConfig:
    window_name: str = "CrowdSafe MVP — CCTV Sentinel"
    show_heatmap: bool = True       # Overlay translucent density heatmap
    show_zones: bool = True         # Show zone bounding boxes and labels
    show_dots: bool = True          # Show detected cluster peak dots
    show_hud: bool = True           # Show top tactical HUD safety banner
    heatmap_alpha: float = 0.45     # Alpha blending weight for heatmap
    headless: bool = False          # Run headless without opening cv2 GUI window

# ---------------------------------------------------------
# Master Application Configuration
# ---------------------------------------------------------
@dataclass
class AppConfig:
    stream: StreamConfig = field(default_factory=StreamConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    zones: List[Dict] = field(default_factory=lambda: DEFAULT_ZONES)
    output_dir: Path = OUTPUT_DIR
