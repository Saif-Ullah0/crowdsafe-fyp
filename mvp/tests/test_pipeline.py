"""
Unit & Integration Test Suite for CrowdSafe MVP
Tests DVR Ingestion, Density Models, Spatial Analysis, Alerting, and Visualizer.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import json
import numpy as np
import pytest

from config import AppConfig, DENSITY_LEVELS
from core.dvr_streamer import DVRStreamer
from core.density_engine import get_density_engine, FastSimModel
from core.spatial_analyzer import SpatialAnalyzer
from core.alert_system import AlertSystem
from core.visualizer import Visualizer


def test_dvr_streamer_synthetic():
    """Verify DVR streamer generates valid synthetic CCTV frames."""
    streamer = DVRStreamer(source="sample", target_fps=10.0, max_frames=3, frame_size=(480, 640))
    success, frame, idx = streamer.read(pace=False)
    
    assert success is True
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert idx == 1
    streamer.close()


def test_fast_sim_density_engine():
    """Verify FastSim procedural model predicts density and count cleanly."""
    model = FastSimModel()
    dummy_frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    # Add a mock high-contrast person patch
    dummy_frame[100:150, 100:150] = 220

    res = model.predict(dummy_frame)
    assert "density_map" in res
    assert "total_count" in res
    assert "latency_ms" in res
    assert isinstance(res["density_map"], np.ndarray)
    assert res["density_map"].ndim == 2
    assert res["total_count"] > 0
    assert res["latency_ms"] >= 0


def test_model_factory_plug_and_play():
    """Verify model factory instantiates requested architectures without crashes."""
    sim_engine = get_density_engine("fast_sim")
    assert sim_engine is not None

    mcnn_engine = get_density_engine("mcnn", weights_path="non_existent.pth")
    assert mcnn_engine is not None

    csr_engine = get_density_engine("csrnet", weights_path="non_existent.pth")
    assert csr_engine is not None


def test_spatial_analyzer_risk_levels():
    """Verify risk classification adheres strictly to configured Level of Service standards."""
    analyzer = SpatialAnalyzer()
    low_val = DENSITY_LEVELS["LOW"]["max"] / 2.0
    mod_val = (DENSITY_LEVELS["LOW"]["max"] + DENSITY_LEVELS["MODERATE"]["max"]) / 2.0
    high_val = (DENSITY_LEVELS["MODERATE"]["max"] + DENSITY_LEVELS["HIGH"]["max"]) / 2.0
    crit_val = DENSITY_LEVELS["HIGH"]["max"] + 1.0

    assert analyzer._classify_risk(low_val) == "LOW"
    assert analyzer._classify_risk(mod_val) == "MODERATE"
    assert analyzer._classify_risk(high_val) == "HIGH"
    assert analyzer._classify_risk(crit_val) == "CRITICAL"


def test_spatial_analysis_integration():
    """Verify spatial analyzer correctly segments zones and flags peak density."""
    analyzer = SpatialAnalyzer()
    density_map = np.zeros((120, 160), dtype=np.float32)
    # Inject heavy density in the Entry Gateway (first zone: bbox 0.05 to 0.45)
    density_map[12:108, 8:72] = 2.0  # high density patch

    analysis = analyzer.analyze(density_map, frame_shape=(480, 640))
    assert "zones" in analysis
    assert len(analysis["zones"]) == 3
    assert "overall_risk" in analysis
    assert "peak_density_m2" in analysis
    assert analysis["peak_density_m2"] > 0.0


def test_alert_hysteresis():
    """Verify alert system only triggers after K consecutive frames, and resets on safe count."""
    config = AppConfig()
    config.alert.consecutive_frames_trigger = 3
    alert_sys = AlertSystem(config=config.alert)

    # Frame with HIGH risk
    zone_high = [{
        "id": "zone_corridor",
        "name": "Central Concourse",
        "risk_level": "HIGH",
        "density_per_m2": 4.5,
        "count": 60.0
    }]
    # Frame with LOW risk
    zone_low = [{
        "id": "zone_corridor",
        "name": "Central Concourse",
        "risk_level": "LOW",
        "density_per_m2": 1.0,
        "count": 10.0
    }]

    # Frame 1: Should not trigger yet (counter = 1)
    a1 = alert_sys.evaluate(zone_high)
    assert len(a1) == 0

    # Frame 2: Should not trigger yet (counter = 2)
    a2 = alert_sys.evaluate(zone_high)
    assert len(a2) == 0

    # Frame 3: Reaches K=3, should trigger!
    a3 = alert_sys.evaluate(zone_high)
    assert len(a3) == 1
    assert a3[0]["alert_level"] == "HIGH"
    assert a3[0]["zone_id"] == "zone_corridor"

    # Frame 4: Reset back to safe
    a4 = alert_sys.evaluate(zone_low)
    assert len(a4) == 0
    assert alert_sys.zone_counters["zone_corridor"] == 0
    alert_sys.shutdown()


def test_alert_instant_flash_spike():
    """Verify that a sudden extreme surge bypasses hysteresis and triggers immediately."""
    config = AppConfig()
    config.alert.consecutive_frames_trigger = 5
    config.alert.instant_critical_threshold = 5.5
    alert_sys = AlertSystem(config=config.alert)

    zone_spike = [{
        "id": "zone_exit",
        "name": "Emergency Exit Route",
        "risk_level": "CRITICAL",
        "density_per_m2": 6.8,  # > 5.5 threshold
        "count": 90.0
    }]

    # Frame 1: Should immediately fire due to is_flash_spike=True
    alerts = alert_sys.evaluate(zone_spike, is_flash_spike=True)
    assert len(alerts) == 1
    assert alerts[0]["trigger_type"] == "INSTANT_FLASH_SPIKE"
    alert_sys.shutdown()


def test_visualizer_render_composite():
    """Verify visualizer composites all overlay layers into a valid output frame."""
    config = AppConfig()
    config.visualizer.headless = True
    vis = Visualizer(config=config.visualizer)

    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    density_map = np.ones((120, 160), dtype=np.float32)
    spatial_analysis = {
        "zones": [
            {"name": "Zone A", "bbox_norm": (0.1, 0.1, 0.5, 0.5), "risk_level": "LOW", "density_per_m2": 1.2, "count": 20.0}
        ],
        "overall_risk": "LOW",
        "peak_density_m2": 1.2,
        "congestion_clusters": [],
        "peak_points": [(150, 150)]
    }

    annotated = vis.render(frame, density_map, spatial_analysis, fps=1.0, latency_ms=5.0)
    assert annotated is not None
    assert annotated.shape == (480, 640, 3)
    vis.close()


def test_csrnet_density_engine():
    """Verify CSRNet model instantiates and processes frames gracefully."""
    csr = get_density_engine("csrnet", weights_path=None)
    dummy_frame = np.full((480, 640, 3), 60, dtype=np.uint8)
    res = csr.predict(dummy_frame)
    assert "density_map" in res
    assert "total_count" in res
    assert "CSRNet" in res["model_name"]


def test_cli_args_parsing(monkeypatch):
    """Verify that CLI flags including --visualize and --weights parse without error."""
    from app import parse_args
    test_args = ["app.py", "--source", "sample", "--visualize", "--weights", "model.pth", "--max-frames", "5"]
    monkeypatch.setattr(sys, "argv", test_args)
    parsed = parse_args()
    assert parsed.visualize is True
    assert parsed.weights == "model.pth"
    assert parsed.max_frames == 5
