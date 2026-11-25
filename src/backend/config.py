"""
CrowdSafe MVP Configuration File
Defines default system parameters, safety thresholds, ROI zones, model paths,
streaming controls, and webhook notification settings.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

BASE_DIR = Path(__file__).resolve().parent

def find_repo_root(start_path: Path) -> Path:
    cur = start_path.resolve()
    for p in [cur] + list(cur.parents):
        if (p / ".git").exists() or (p / "DataSet").exists():
            return p
    return cur.parent

REPO_ROOT = find_repo_root(BASE_DIR)
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Floor Plan Themes (Design 4 Variants)
# ---------------------------------------------------------
FLOOR_PLAN_LIGHT = REPO_ROOT / "design_4_clean_base.png"
if not FLOOR_PLAN_LIGHT.exists():
    _asset_light = BASE_DIR / "assets" / "design_4_clean_base.png"
    if _asset_light.exists():
        FLOOR_PLAN_LIGHT = _asset_light

FLOOR_PLAN_DARK = REPO_ROOT / "design_4_tactical_gamemap.png"
if not FLOOR_PLAN_DARK.exists():
    _asset_dark = BASE_DIR / "assets" / "design_4_tactical_gamemap.png"
    if _asset_dark.exists():
        FLOOR_PLAN_DARK = _asset_dark

DEFAULT_FLOOR_PLAN = REPO_ROOT / "design_4_transit_interchange.png"
if not DEFAULT_FLOOR_PLAN.exists():
    _asset_plan = BASE_DIR / "assets" / "design_4_transit_interchange.png"
    if _asset_plan.exists():
        DEFAULT_FLOOR_PLAN = _asset_plan

@dataclass
class StreamConfig:
    source: str = "sample"          # 'sample', directory path, video file, or camera index
    target_fps: float = 1.0        # Default 1 FPS CCTV/DVR polling rate
    loop: bool = True              # Loop image sequences or video clips
    max_frames: Optional[int] = None  # None = unlimited, or int limit
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
    webhook_url: Optional[str] = "http://localhost:3000/api/webhook"  # Next.js telemetry dashboard endpoint
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
