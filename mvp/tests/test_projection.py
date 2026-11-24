"""
Unit Tests for Camera-to-Floor-Plan Projection & Modular Detection Pipeline
"""
import sys
import os
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

# Ensure mvp/ is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.projection import PlanarProjector
from core.detection_source import (
    PersonDetection,
    AnnotationDetectionSource,
    ModelDetectionSource
)
from core.density_engine import FastSimModel


def test_homography_projection_accuracy():
    """Verify forward and inverse perspective projective transformation accuracy."""
    # Define known synthetic camera coordinates
    src_pts = [(0.0, 0.0), (1920.0, 0.0), (1920.0, 1080.0), (0.0, 1080.0)]
    # Define mapped floor coordinates (e.g. tilted trapezoid on floor plan)
    dst_pts = [(500.0, 300.0), (1200.0, 350.0), (1500.0, 900.0), (200.0, 850.0)]

    proj = PlanarProjector(camera_shape=(1080, 1920), floor_shape=(1200, 1600))
    ok = proj.compute_from_points(src_pts, dst_pts)
    assert ok is True
    assert proj.is_calibrated is True

    # Test projection of each of the 4 calibration points
    for (sx, sy), (dx, dy) in zip(src_pts, dst_pts):
        proj_pt = proj.project_point(sx, sy)
        assert proj_pt is not None
        px, py = proj_pt
        assert pytest.approx(px, abs=1e-3) == dx
        assert pytest.approx(py, abs=1e-3) == dy

    # Test vectorized batch projection
    batch_res = proj.project_points(src_pts)
    assert batch_res.shape == (4, 2)
    for i, (dx, dy) in enumerate(dst_pts):
        assert pytest.approx(batch_res[i][0], abs=1e-3) == dx
        assert pytest.approx(batch_res[i][1], abs=1e-3) == dy


def test_calibration_save_and_load_roundtrip():
    """Verify serialization and deserialization of homography calibration JSON."""
    src_pts = [(10.0, 20.0), (1900.0, 25.0), (1910.0, 1070.0), (15.0, 1060.0)]
    dst_pts = [(100.0, 150.0), (800.0, 160.0), (950.0, 600.0), (50.0, 580.0)]

    proj = PlanarProjector(
        camera_shape=(1080, 1920),
        floor_shape=(2000, 3000),
        metadata={"camera_id": "CAM_40"}
    )
    proj.compute_from_points(src_pts, dst_pts)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        proj.save_calibration(tmp_path)
        assert tmp_path.is_file()

        loaded_proj = PlanarProjector.load_calibration(tmp_path)
        assert loaded_proj.is_calibrated is True
        assert loaded_proj.camera_shape == (1080, 1920)
        assert loaded_proj.floor_shape == (2000, 3000)
        assert loaded_proj.metadata.get("camera_id") == "CAM_40"

        # Verify H matrices match
        np.testing.assert_allclose(proj.H, loaded_proj.H, atol=1e-6)

        # Verify test point projection matches
        p1 = proj.project_point(960.0, 540.0)
        p2 = loaded_proj.project_point(960.0, 540.0)
        assert pytest.approx(p1[0], abs=1e-4) == p2[0]
        assert pytest.approx(p1[1], abs=1e-4) == p2[1]
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_annotation_detection_source_parsing():
    """Verify parsing VIA JSON format and computing ground-contact feet points."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_path = Path(tmp_dir)
        # Create a mock 001.json in VIA format
        sample_data = {
            "001.jpg123": {
                "filename": "001.jpg",
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "rect",
                            "x": 100,
                            "y": 200,
                            "width": 40,
                            "height": 80
                        }
                    },
                    {
                        "shape_attributes": {
                            "name": "point",
                            "cx": 350,
                            "cy": 500
                        }
                    }
                ]
            }
        }
        with open(dir_path / "001.json", "w") as f:
            json.dump(sample_data, f)

        source = AnnotationDetectionSource(dir_path)
        dets = source.get_detections(frame_idx=1)

        assert len(dets) == 2

        # First detection: rect (100, 200, 40, 80)
        d1 = dets[0]
        assert d1.bbox == (100.0, 200.0, 40.0, 80.0)
        # feet point: x + w/2 = 120, y + h = 280
        assert d1.feet_point == (120.0, 280.0)

        # Second detection: point (cx=350, cy=500)
        d2 = dets[1]
        assert d2.feet_point == (350.0, 500.0)


def test_dataset_40_live_annotation_read():
    """Verify live read of actual DataSet/40 if present in workspace."""
    dataset_40 = BASE_DIR.parent / "DataSet" / "40"
    if dataset_40.is_dir():
        source = AnnotationDetectionSource(dataset_40)
        dets = source.get_detections(frame_idx=1)
        assert len(dets) > 0
        for d in dets:
            assert len(d.bbox) == 4
            assert len(d.feet_point) == 2


def test_camera_quad_generation():
    """Verify get_projected_camera_quad returns 4 valid vertices on floor."""
    src_pts = [(0.0, 0.0), (640.0, 0.0), (640.0, 480.0), (0.0, 480.0)]
    dst_pts = [(100.0, 100.0), (500.0, 100.0), (600.0, 400.0), (50.0, 400.0)]

    proj = PlanarProjector(camera_shape=(480, 640), floor_shape=(1000, 1000))
    proj.compute_from_points(src_pts, dst_pts)

    quad = proj.get_projected_camera_quad()
    assert quad is not None
    assert quad.shape == (4, 2)
    # Check bounds
    assert np.all(quad >= 0)


def test_model_detection_source_adapter():
    """Verify ModelDetectionSource wraps density engine smoothly."""
    engine = FastSimModel()
    dummy_frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    dummy_frame[100:140, 100:140] = 230  # bright patch

    source = ModelDetectionSource(density_engine=engine)
    dets = source.get_detections(frame_idx=1, frame_bgr=dummy_frame)
    assert isinstance(dets, list)


def test_display_scale_homography_and_throughput():
    """Verify display-scale homography (H_disp = S @ H) preserves geometry and runs at >= 30 FPS."""
    import time
    import cv2

    src_pts = [(0.0, 0.0), (1920.0, 0.0), (1920.0, 1080.0), (0.0, 1080.0)]
    dst_pts = [(1000.0, 800.0), (3000.0, 800.0), (3800.0, 3500.0), (200.0, 3500.0)]

    orig_floor_w, orig_floor_h = 5336, 5488
    disp_w, disp_h = 680, 680
    scale_x = disp_w / orig_floor_w
    scale_y = disp_h / orig_floor_h

    proj = PlanarProjector(camera_shape=(1080, 1920), floor_shape=(orig_floor_h, orig_floor_w))
    proj.compute_from_points(src_pts, dst_pts)

    S = np.array([
        [scale_x, 0.0, 0.0],
        [0.0, scale_y, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    H_disp = S @ proj.H

    # Test that projecting (sx, sy) through H and then scaling matches projecting through H_disp
    test_cam_pts = np.array([
        [0.0, 0.0],
        [960.0, 540.0],
        [1920.0, 1080.0]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Via original H then scale
    pts_orig = cv2.perspectiveTransform(test_cam_pts, proj.H).reshape(-1, 2)
    pts_orig_scaled = pts_orig * np.array([scale_x, scale_y])

    # Via H_disp directly
    pts_disp = cv2.perspectiveTransform(test_cam_pts, H_disp).reshape(-1, 2)

    np.testing.assert_allclose(pts_disp, pts_orig_scaled, rtol=1e-5, atol=1e-4)

    # Benchmark display-scale warp throughput (must comfortably exceed 30 FPS, target >= 60 FPS)
    dummy_frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    # Warmup
    for _ in range(5):
        _ = cv2.warpPerspective(dummy_frame, H_disp, (disp_w, disp_h))

    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = cv2.warpPerspective(dummy_frame, H_disp, (disp_w, disp_h))
    dt = time.perf_counter() - t0
    fps = iters / dt

    # Must achieve at least 30 FPS
    assert fps >= 30.0, f"Expected warp throughput >= 30 FPS, got {fps:.1f} FPS"


def test_transform_state_to_and_from_pins():
    """Verify TransformState mathematical roundtrip between parametric state and 4 corner pins."""
    from calibrate import TransformState

    w0, h0 = 500.0, 300.0
    floor_w, floor_h = 5000.0, 5000.0

    # Test varying rotations, scales, and keystone flare factors
    for deg in [0.0, 45.0, 90.0, 180.0, 270.0, 315.0]:
        for scale in [0.5, 1.0, 1.8]:
            for keystone in [0.8, 1.0, 1.25]:
                state = TransformState(
                    cx=2500.0,
                    cy=2500.0,
                    rotation_deg=deg,
                    scale=scale,
                    aspect=1.0,
                    keystone=keystone,
                    w0=w0,
                    h0=h0
                )
                pins = state.to_corner_pins(floor_w, floor_h)
                assert len(pins) == 4

                reconstructed = TransformState.from_corner_pins(pins, w0, h0)
                assert pytest.approx(reconstructed.rotation_deg, abs=1e-2) == deg
                assert pytest.approx(reconstructed.scale, abs=1e-2) == scale
                assert pytest.approx(reconstructed.keystone, abs=1e-2) == keystone
                assert pytest.approx(reconstructed.cx, abs=1.0) == 2500.0
                assert pytest.approx(reconstructed.cy, abs=1.0) == 2500.0


def test_ui_button_hit_testing():
    """Verify UIButton bounds check and hit testing for dedicated Window 2."""
    from calibrate import UIButton

    btn = UIButton(rect=(100, 200, 80, 40), text="Test", action="TEST_ACT")
    assert btn.contains(120, 220) is True
    assert btn.contains(100, 200) is True
    assert btn.contains(180, 240) is True
    assert btn.contains(99, 220) is False
    assert btn.contains(181, 220) is False
    assert btn.contains(120, 199) is False
    assert btn.contains(120, 241) is False


def test_canva_figma_studio_hit_testing_and_rendering():
    """Verify Figma/Canva bounding box handle hit testing and dynamic pointer icon rendering."""
    from calibrate import CanvaFigmaCalibrationStudio, DynamicPointerIcon

    floor_dummy = np.full((1200, 1600, 3), 200, dtype=np.uint8)
    studio = CanvaFigmaCalibrationStudio(
        floor_image=floor_dummy,
        source_dir=BASE_DIR.parent / "DataSet" / "40"
    )

    # 1. Render test
    frame = studio.render()
    assert frame is not None
    assert frame.shape == (studio.disp_h, studio.disp_w, 3)

    geo = studio._get_canvas_geometry()
    handles = studio._compute_screen_handles(geo)

    rot_knob = handles["rot_knob"]
    disp_pins = handles["disp_pins"]
    center = handles["center"]

    # 2. Hit-test rotation lollipop
    hit_type, _, p_type = studio._hit_test(rot_knob[0], rot_knob[1], geo)
    assert hit_type == "ROT_HANDLE"
    assert p_type == "ROTATE"

    # 3. Hit-test corner scale handle (TL corner)
    tl = disp_pins[0]
    hit_type, h_idx, p_type = studio._hit_test(tl[0], tl[1], geo)
    assert hit_type == "CORNER_SCALE"
    assert h_idx == 0
    assert p_type == "RESIZE_DIAG_1"

    # 4. Hit-test body move (at center)
    hit_type, _, p_type = studio._hit_test(center[0], center[1], geo)
    assert hit_type == "BODY"
    assert p_type == "MOVE"

    # 5. Hit-test ribbon button
    btn_rect = studio.ribbon_buttons[0].rect
    hit_type, _, _ = studio._hit_test(btn_rect[0] + 5, btn_rect[1] + 5, geo)
    assert hit_type == "RIBBON_BTN"

    # 6. Test all dynamic pointer types render without exceptions
    test_canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    for p in ["MOVE", "ROTATE", "RESIZE_DIAG_1", "RESIZE_DIAG_2", "DISTORT_PIN", "POINTER"]:
        DynamicPointerIcon.draw(test_canvas, 100, 100, p, tooltip="Test 45°")


def test_universal_target_resolution_and_custom_folders():
    """Verify universal target resolver handles single image files, custom folders, and datasets."""
    from calibrate import resolve_universal_targets, CalibrationTarget

    # 1. Test dataset series string (e.g. '40')
    targets_40, idx_40 = resolve_universal_targets("40")
    assert len(targets_40) >= 1
    assert targets_40[idx_40].name == "40"
    assert targets_40[idx_40].image_path.is_file()

    # 2. Test single image file input
    sample_img = BASE_DIR.parent / "DataSet" / "40" / "001.jpg"
    if sample_img.is_file():
        targets_file, idx_file = resolve_universal_targets(str(sample_img))
        assert len(targets_file) >= 1
        assert targets_file[idx_file].is_file is True
        assert targets_file[idx_file].name == "001.jpg"
        assert targets_file[idx_file].calib_path.name == "001_calibration.json"

    # 3. Test arbitrary custom directory with temporary images
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_cam = Path(tmp_dir) / "custom_hallway_cam"
        tmp_cam.mkdir()
        # Create a dummy image
        import cv2
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_cam / "cam_frame.png"), dummy_img)

        targets_custom, idx_custom = resolve_universal_targets(str(tmp_cam))
        assert len(targets_custom) == 1
        assert targets_custom[idx_custom].name == "custom_hallway_cam"
        assert targets_custom[idx_custom].image_path == (tmp_cam / "cam_frame.png")


def test_floor_plan_theme_assets_and_config():
    """Verify Light CAD and Dark Tactical floor plan assets exist and match architectural scale."""
    from config import FLOOR_PLAN_LIGHT, FLOOR_PLAN_DARK
    import cv2

    assert Path(FLOOR_PLAN_LIGHT).is_file(), f"Light floor plan missing: {FLOOR_PLAN_LIGHT}"
    assert Path(FLOOR_PLAN_DARK).is_file(), f"Dark floor plan missing: {FLOOR_PLAN_DARK}"

    img_light = cv2.imread(str(FLOOR_PLAN_LIGHT))
    img_dark = cv2.imread(str(FLOOR_PLAN_DARK))

    assert img_light is not None, "Failed to decode FLOOR_PLAN_LIGHT"
    assert img_dark is not None, "Failed to decode FLOOR_PLAN_DARK"
    assert img_light.shape == img_dark.shape, f"Theme dimension mismatch: {img_light.shape} vs {img_dark.shape}"
    assert img_light.shape[:2] == (4170, 3912), f"Unexpected dimensions: {img_light.shape}"


def test_projector_dual_canvas_pre_scaling_and_toggle():
    """Verify ProjectorApp pre-scales dual canvases and handles instant runtime theme switching."""
    from projector_app import ProjectorApp
    import cv2

    series_40 = BASE_DIR.parent / "DataSet" / "40"
    if not series_40.exists():
        pytest.skip("DataSet/40 not present for integration test")

    app = ProjectorApp(source_dir=series_40, headless=True, theme="dark")
    assert app.theme == "dark"
    assert app.floor_dark_disp.shape == (680, 680, 3)
    assert app.floor_light_disp.shape == (680, 680, 3)

    # Active canvas should point to floor_dark_disp
    assert np.array_equal(app.floor_img_disp, app.floor_dark_disp)

    # Toggle to Light CAD theme
    app.toggle_theme()
    assert app.theme == "light"
    assert np.array_equal(app.floor_img_disp, app.floor_light_disp)

    # Toggle back to Dark Tactical theme
    app.toggle_theme()
    assert app.theme == "dark"
    assert np.array_equal(app.floor_img_disp, app.floor_dark_disp)

    # Simulate mouse click on the theme button in HUD
    bx1, by1, bx2, by2 = app.theme_btn_rect
    click_x = (bx1 + bx2) // 2
    click_y = (by1 + by2) // 2
    app._on_floor_mouse_event(cv2.EVENT_LBUTTONDOWN, click_x, click_y, 0, None)
    assert app.theme == "light"

    # Render Window 2 (Projection + Dots) under Light theme
    frame = cv2.imread(app.image_files[0])
    canvas_light = app.render_floor_projection_with_dots(frame, [], 1)
    assert canvas_light.shape == (680, 680, 3)

    # Switch to Dark and render again
    app.toggle_theme()
    canvas_dark = app.render_floor_projection_with_dots(frame, [], 1)
    assert canvas_dark.shape == (680, 680, 3)


def test_projector_gui_window_initialization():
    """Verify ProjectorApp GUI window names and attributes are properly initialized."""
    from projector_app import ProjectorApp
    import cv2

    series_40 = BASE_DIR.parent / "DataSet" / "40"
    if not series_40.exists():
        pytest.skip("DataSet/40 not present for integration test")

    app = ProjectorApp(source_dir=series_40, headless=False, theme="dark")
    assert app.headless is False
    assert hasattr(app, "win_floor_proj")
    assert hasattr(app, "win_floor_dots")
    assert hasattr(app, "win_cam_video")
    cv2.destroyAllWindows()






