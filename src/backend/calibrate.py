#!/usr/bin/env python3
"""
CrowdSafe MVP — Figma & Canva-Style Universal Visual Calibration Studio (Desktop Only)
Interactively calibrates camera feeds into the 2D CAD Floor Plan.

Supports:
  1. Universal Input: Any single image (.jpg, .png, .jpeg, .webp, .bmp), any custom folder, or parent dataset.
  2. In-App File/Folder Browser: Click [ 🖼️ Image ] or [ 📁 Folder ] in the top ribbon to load any file on disk.
  3. Direct Canvas Manipulation: Drag & drop camera feed, resize corners, rotate with top lollipop handle.
  4. Dynamic Pointer Icons: Visual cursor morphs directly on canvas (Move ✥, Rotate ↻, Resize ⤢/⤡, Distort ✦).
  5. Zero Shortcut Friction: Everything is directly manipulable via mouse, with a clean top application ribbon.
"""
import sys
import os
import argparse
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

# Ensure cv2/qt/fonts directory exists to suppress Qt font database warning on Linux
try:
    _qt_font_dir = Path(cv2.__file__).resolve().parent / "qt" / "fonts"
    _qt_font_dir.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Ensure mvp/ is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import DEFAULT_FLOOR_PLAN, REPO_ROOT, BASE_DIR
from core.projection import PlanarProjector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class CalibrationTarget:
    """Represents a calibration target: either a directory containing frames or a single image file."""
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.is_file = self.path.is_file()

        if self.is_file:
            self.name = self.path.name
            self.image_path = self.path
            self.folder_path = self.path.parent
            self.calib_path = self.path.parent / f"{self.path.stem}_calibration.json"
            self.fallback_calib_path = self.path.parent / "calibration.json"
            self.frame_count = 1
        else:
            self.name = self.path.name
            self.folder_path = self.path
            imgs = sorted([
                f for f in self.path.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ])
            if not imgs:
                raise ValueError(f"No image files found in folder: {self.path}")
            self.image_path = imgs[0]
            self.calib_path = self.path / "calibration.json"
            self.fallback_calib_path = self.calib_path
            self.frame_count = len(imgs)

    @property
    def existing_calib_file(self) -> Optional[Path]:
        if self.calib_path.is_file():
            return self.calib_path
        if self.fallback_calib_path.is_file():
            return self.fallback_calib_path
        return None


def find_images_in_dir(d: Path) -> List[Path]:
    """Finds all image files directly inside directory d."""
    if not d.is_dir():
        return []
    return sorted([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])


def resolve_series_path(source_or_series: str) -> Path:
    """Resolves series input into an absolute Path to a folder or image file."""
    p = Path(source_or_series)
    if p.exists():
        return p.resolve()

    cand1 = REPO_ROOT / "DataSet" / source_or_series
    if cand1.exists():
        return cand1.resolve()

    cand2 = BASE_DIR / "DataSet" / source_or_series
    if cand2.exists():
        return cand2.resolve()

    cand3 = Path("DataSet") / source_or_series
    if cand3.exists():
        return cand3.resolve()

    cand4 = REPO_ROOT / source_or_series
    if cand4.exists():
        return cand4.resolve()

    return p.resolve()


def resolve_universal_targets(source_or_series: Optional[str] = None) -> Tuple[List[CalibrationTarget], int]:
    """
    Universally resolves any input:
      - Single image file -> list of sibling images with selected image active
      - Folder with images -> folder target + sibling folders if present
      - Parent folder with subfolders -> list of all subfolders with images
      - Series name (e.g. '40') -> searches DataSet/ or local folders
      - None / default -> DataSet/40
    """
    if not source_or_series:
        source_or_series = "DataSet/40"

    p = resolve_series_path(source_or_series)

    # 1. Direct single image file
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
        sibling_imgs = find_images_in_dir(p.parent)
        targets = [CalibrationTarget(f) for f in sibling_imgs]
        idx = next((i for i, t in enumerate(targets) if t.path == p), 0)
        return targets, idx

    # 2. Directory handling
    if p.is_dir():
        direct_imgs = find_images_in_dir(p)
        if direct_imgs:
            # Check sibling folders containing images
            parent = p.parent
            sibling_dirs = [d for d in sorted(parent.iterdir()) if d.is_dir() and find_images_in_dir(d)]
            if p in sibling_dirs and len(sibling_dirs) > 1:
                targets = [CalibrationTarget(d) for d in sibling_dirs]
                idx = next((i for i, t in enumerate(targets) if t.path == p), 0)
                return targets, idx
            return [CalibrationTarget(p)], 0

        # Parent directory containing subfolders with images (e.g. test_data or DataSet)
        sub_dirs = [d for d in sorted(p.iterdir()) if d.is_dir() and find_images_in_dir(d)]
        if sub_dirs:
            targets = [CalibrationTarget(d) for d in sub_dirs]
            return targets, 0

    # Fallback to DataSet/40 if available
    fallback_dir = REPO_ROOT / "DataSet" / "40"
    if fallback_dir.is_dir():
        return resolve_universal_targets(str(fallback_dir))

    raise ValueError(f"Could not resolve any valid image or folder from: {source_or_series}")


def discover_all_series(source_path: Path) -> List[Path]:
    """Compatibility wrapper returning list of directory paths."""
    targets, _ = resolve_universal_targets(str(source_path))
    return [t.folder_path for t in targets]


def list_available_series():
    """Lists all detected camera series folders and their calibration status."""
    targets, _ = resolve_universal_targets("DataSet/40")

    print("\n" + "=" * 65)
    print(" Available Camera Targets in Dataset:")
    print("=" * 65)
    for t in targets:
        calib_tag = "✅ Calibrated" if t.existing_calib_file else "⚠️  Uncalibrated"
        unit = "file" if t.is_file else "frames"
        print(f"  • Target [{t.name:20s}] : {t.frame_count:3d} {unit:<6s} | {calib_tag}")
    print("=" * 65 + "\n")


@dataclass
class TransformState:
    """Parametric representation of camera quad placement on floor plan."""
    cx: float
    cy: float
    rotation_deg: float = 0.0
    scale: float = 1.0
    aspect: float = 1.0
    keystone: float = 1.0
    alpha: float = 0.50
    w0: float = 500.0
    h0: float = 300.0

    def to_corner_pins(self, floor_w: float, floor_h: float) -> List[List[float]]:
        """Converts parametric state to 4 corner points [TL, TR, BR, BL] on floor plan."""
        w = max(20.0, self.w0 * self.scale)
        h = max(20.0, self.h0 * self.scale * self.aspect)
        k = max(0.2, min(5.0, self.keystone))

        # Local camera frame points:
        # Sightline is forward (-Y), so TL and TR are top (-h), BR and BL are bottom (+h)
        local_pts = np.array([
            [-w * k, -h],  # 0: TL
            [+w * k, -h],  # 1: TR
            [+w / k, +h],  # 2: BR
            [-w / k, +h],  # 3: BL
        ], dtype=np.float64)

        rad = np.radians(self.rotation_deg)
        cos_t, sin_t = np.cos(rad), np.sin(rad)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        rotated = np.dot(local_pts, R.T) + np.array([self.cx, self.cy])

        clipped = []
        for pt in rotated:
            clipped.append([
                float(np.clip(pt[0], 0, floor_w)),
                float(np.clip(pt[1], 0, floor_h))
            ])
        return clipped

    @classmethod
    def from_corner_pins(
        cls,
        pins: List[List[float]],
        w0: float,
        h0: float,
        alpha: float = 0.50
    ) -> "TransformState":
        """Estimates parametric state from 4 corner points."""
        pts = np.array(pins, dtype=np.float64)
        cx, cy = np.mean(pts, axis=0)

        top_mid = (pts[0] + pts[1]) / 2.0
        bot_mid = (pts[2] + pts[3]) / 2.0
        v = top_mid - bot_mid

        deg = float(np.degrees(np.arctan2(v[0], -v[1])) % 360.0)

        h_dist = float(np.linalg.norm(v))
        h = max(10.0, h_dist / 2.0)

        top_w = float(np.linalg.norm(pts[1] - pts[0]))
        bot_w = float(np.linalg.norm(pts[2] - pts[3]))

        w = max(10.0, 0.5 * np.sqrt(max(1.0, top_w * bot_w)))
        keystone = float(np.sqrt(max(0.01, top_w / max(1.0, bot_w))))

        scale = float(w / max(1.0, w0))
        aspect = float((h / max(1.0, h0)) / max(0.01, scale))

        return cls(
            cx=float(cx),
            cy=float(cy),
            rotation_deg=deg,
            scale=scale,
            aspect=aspect,
            keystone=keystone,
            alpha=alpha,
            w0=w0,
            h0=h0
        )


class TopRibbonButton:
    """Clickable UI button inside the top application ribbon."""
    def __init__(
        self,
        rect: Tuple[int, int, int, int],  # x, y, w, h
        text: str,
        action: str,
        bg_color: Tuple[int, int, int] = (45, 45, 55),
        hover_color: Tuple[int, int, int] = (65, 65, 80),
        active_color: Tuple[int, int, int] = (24, 160, 251),  # Figma blue
        text_color: Tuple[int, int, int] = (240, 240, 245),
        border_color: Tuple[int, int, int] = (75, 75, 90),
        font_scale: float = 0.40,
        is_toggle: bool = False
    ):
        self.rect = rect
        self.text = text
        self.action = action
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.active_color = active_color
        self.text_color = text_color
        self.border_color = border_color
        self.font_scale = font_scale
        self.is_toggle = is_toggle

    def contains(self, x: int, y: int) -> bool:
        bx, by, bw, bh = self.rect
        return bx <= x <= bx + bw and by <= y <= by + bh

    def draw(self, canvas: np.ndarray, is_hover: bool = False, is_active: bool = False):
        bx, by, bw, bh = self.rect
        if is_active:
            color = self.active_color
            border = (255, 255, 255)
            txt_col = (255, 255, 255)
        elif is_hover:
            color = self.hover_color
            border = (120, 120, 150)
            txt_col = (255, 255, 255)
        else:
            color = self.bg_color
            border = self.border_color
            txt_col = self.text_color

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border, 1, lineType=cv2.LINE_AA)

        (tw, th), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
        tx = bx + max(2, (bw - tw) // 2)
        ty = by + max(th, (bh + th) // 2)
        cv2.putText(canvas, self.text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, txt_col, 1, cv2.LINE_AA)


# Compatibility alias
UIButton = TopRibbonButton


class DynamicPointerIcon:
    """Renders Canva / Figma-style dynamic pointer icons directly at mouse position."""

    @staticmethod
    def draw(canvas: np.ndarray, x: int, y: int, pointer_type: str, tooltip: str = ""):
        h, w = canvas.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        if pointer_type == "MOVE":
            cv2.line(canvas, (x - 12, y), (x + 12, y), (0, 0, 0), 4, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x, y - 12), (x, y + 12), (0, 0, 0), 4, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x - 12, y), (x + 12, y), (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x, y - 12), (x, y + 12), (255, 255, 255), 2, lineType=cv2.LINE_AA)
            for pt in [(x - 12, y), (x + 12, y), (x, y - 12), (x, y + 12)]:
                cv2.circle(canvas, pt, 2, (24, 160, 251), -1, lineType=cv2.LINE_AA)

        elif pointer_type == "ROTATE":
            center = (x, y)
            radius = 11
            cv2.ellipse(canvas, center, (radius, radius), 0, 30, 310, (0, 0, 0), 4, lineType=cv2.LINE_AA)
            cv2.ellipse(canvas, center, (radius, radius), 0, 30, 310, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            arrow_pt = (x + 10, y - 4)
            cv2.circle(canvas, arrow_pt, 3, (24, 160, 251), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, center, 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        elif pointer_type == "RESIZE_DIAG_1":
            p1 = (x - 9, y - 9)
            p2 = (x + 9, y + 9)
            cv2.line(canvas, p1, p2, (0, 0, 0), 4, lineType=cv2.LINE_AA)
            cv2.line(canvas, p1, p2, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, (24, 160, 251), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p2, 3, (24, 160, 251), -1, lineType=cv2.LINE_AA)

        elif pointer_type == "RESIZE_DIAG_2":
            p1 = (x + 9, y - 9)
            p2 = (x - 9, y + 9)
            cv2.line(canvas, p1, p2, (0, 0, 0), 4, lineType=cv2.LINE_AA)
            cv2.line(canvas, p1, p2, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, (24, 160, 251), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p2, 3, (24, 160, 251), -1, lineType=cv2.LINE_AA)

        elif pointer_type == "DISTORT_PIN":
            cv2.circle(canvas, (x, y), 8, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 6, (0, 245, 255), 2, lineType=cv2.LINE_AA)
            cv2.drawMarker(canvas, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

        elif pointer_type == "POINTER":
            pts = np.array([[x, y], [x, y + 16], [x + 4, y + 12], [x + 9, y + 16], [x + 11, y + 14], [x + 6, y + 10], [x + 12, y + 10]], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.fillPoly(canvas, [pts], (255, 255, 255), lineType=cv2.LINE_AA)

        if tooltip:
            (tw, th), _ = cv2.getTextSize(tooltip, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            bx, by = x + 16, y - 10
            cv2.rectangle(canvas, (bx - 4, by - th - 4), (bx + tw + 6, by + 6), (20, 20, 28), -1)
            cv2.rectangle(canvas, (bx - 4, by - th - 4), (bx + tw + 6, by + 6), (80, 80, 110), 1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, tooltip, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 230, 100), 1, cv2.LINE_AA)


class CanvaFigmaCalibrationStudio:
    """
    Canva & Figma-Style Universal Creative Calibration Studio.
    Calibrates any image or folder with direct drag-and-drop, corner scaling, and rotation.
    """

    def __init__(
        self,
        floor_image: np.ndarray,
        source_dir: Optional[Path] = None,
        series_list: Optional[List[Path]] = None,
        initial_series_name: Optional[str] = None
    ):
        self.floor_img_orig = floor_image
        self.floor_h, self.floor_w = self.floor_img_orig.shape[:2]

        # Universal resolution of targets
        raw_source = str(source_dir) if source_dir else "DataSet/40"
        self.targets, self.current_target_idx = resolve_universal_targets(raw_source)

        if initial_series_name:
            for i, t in enumerate(self.targets):
                if t.name == initial_series_name:
                    self.current_target_idx = i
                    break

        self.active_target: CalibrationTarget = self.targets[self.current_target_idx]
        self.target_name = self.active_target.name
        self.series_name = self.target_name
        self.output_path = self.active_target.calib_path

        # Window settings
        self.win_name = "CrowdSafe Studio - Visual Floor Plan Calibration"
        self.disp_w = 1280
        self.disp_h = 820
        self.ribbon_h = 56
        self.canvas_h = self.disp_h - self.ribbon_h

        # Modes: "TRANSFORM" (Move, Scale, Rotate) | "DISTORT" (Perspective Corner Pins)
        self.tool_mode = "TRANSFORM"
        self.zoom_mode: bool = False
        self.zoom_factor: float = 2.5

        # Base dimensions
        self.w0 = self.floor_w * 0.14
        self.h0 = self.floor_h * 0.10

        # State
        cx, cy = self.floor_w / 2.0, self.floor_h / 2.0
        self.state = TransformState(
            cx=cx, cy=cy, rotation_deg=0.0, scale=1.0, aspect=1.0, keystone=1.0, alpha=0.55,
            w0=self.w0, h0=self.h0
        )
        self.corner_pins: List[List[float]] = self.state.to_corner_pins(self.floor_w, self.floor_h)

        # Mouse & Pointer State
        self.mouse_x: int = -100
        self.mouse_y: int = -100
        self.active_pointer_type: str = "POINTER"
        self.pointer_tooltip: str = ""

        # Dragging State
        self.drag_mode: Optional[str] = None
        self.active_handle_idx: int = -1
        self.drag_start_floor: Tuple[float, float] = (0.0, 0.0)
        self.drag_start_disp: Tuple[int, int] = (0, 0)
        self.start_state_cx: float = 0.0
        self.start_state_cy: float = 0.0
        self.start_state_scale: float = 1.0
        self.start_state_rot: float = 0.0
        self.start_dist_to_center: float = 1.0
        self.start_pins: List[List[float]] = []

        # Top Ribbon Buttons
        self.ribbon_buttons: List[TopRibbonButton] = []
        self.hover_button_action: Optional[str] = None
        self._build_ribbon_buttons()

        # Toast notification
        self.toast_msg: str = ""
        self.toast_time: float = 0.0
        self.has_unsaved_changes: bool = False

        # Load active camera target
        self._load_active_target()

    def _load_active_target(self):
        """Loads active camera frame and calibration."""
        self.active_target = self.targets[self.current_target_idx]
        self.target_name = self.active_target.name
        self.series_name = self.target_name
        self.output_path = self.active_target.calib_path

        self.cam_img_orig = cv2.imread(str(self.active_target.image_path))
        if self.cam_img_orig is None:
            raise ValueError(f"Failed to read image from {self.active_target.image_path}")
        self.cam_h, self.cam_w = self.cam_img_orig.shape[:2]

        self.projector = PlanarProjector(
            camera_shape=(self.cam_h, self.cam_w),
            floor_shape=(self.floor_h, self.floor_w)
        )

        cam_aspect = (self.cam_h / max(1, self.cam_w))
        self.h0 = self.w0 * cam_aspect
        self.state.h0 = self.h0

        self._load_existing_calibration()
        self._recompute_corner_homography()
        self.has_unsaved_changes = False

    def _load_existing_calibration(self):
        """Loads previous calibration if available."""
        calib_file = self.active_target.existing_calib_file
        if calib_file and calib_file.is_file():
            try:
                proj = PlanarProjector.load_calibration(calib_file)
                if proj.is_calibrated and len(proj.dst_points) == 4:
                    self.corner_pins = [list(p) for p in proj.dst_points]
                    self.projector = proj
                    self.state = TransformState.from_corner_pins(
                        self.corner_pins, self.w0, self.h0, alpha=self.state.alpha
                    )
                    self.set_toast(f"Loaded calibration for [{self.target_name}]")
            except Exception as e:
                print(f"[Studio] Note: Could not parse existing calibration ({e})")
        else:
            cx, cy = self.floor_w / 2.0, self.floor_h / 2.0
            self.state.cx = cx
            self.state.cy = cy
            self.state.rotation_deg = 0.0
            self.state.scale = 1.0
            self.state.aspect = 1.0
            self.state.keystone = 1.0
            self.corner_pins = self.state.to_corner_pins(self.floor_w, self.floor_h)

    def _recompute_corner_homography(self):
        """Computes homography H from the 4 corner handles."""
        cam_corners = [
            (0.0, 0.0),
            (float(self.cam_w), 0.0),
            (float(self.cam_w), float(self.cam_h)),
            (0.0, float(self.cam_h))
        ]
        floor_corners = [tuple(p) for p in self.corner_pins]
        self.projector.compute_from_points(cam_corners, floor_corners)

    def switch_series(self, direction: int):
        """Switches to next (+1) or previous (-1) target."""
        if len(self.targets) <= 1:
            self.set_toast(f"Only 1 target: [{self.target_name}]")
            return

        self.current_target_idx = (self.current_target_idx + direction) % len(self.targets)
        self._load_active_target()
        calib_tag = "✅ Calibrated" if self.active_target.existing_calib_file else "⚠️ Uncalibrated"
        self.set_toast(f"Active: [{self.target_name}] ({self.current_target_idx + 1}/{len(self.targets)}) — {calib_tag}")

    def load_new_path(self, p: Path):
        """Loads an arbitrary newly selected image file or directory into the studio."""
        try:
            targets, idx = resolve_universal_targets(str(p))
            self.targets = targets
            self.current_target_idx = idx
            self._load_active_target()
            calib_tag = "✅ Calibrated" if self.active_target.existing_calib_file else "⚠️ Uncalibrated"
            self.set_toast(f"Opened: [{self.target_name}] ({self.current_target_idx + 1}/{len(self.targets)}) — {calib_tag}")
            print(f"🔄 [Studio] Opened new target: {self.active_target.path} ({calib_tag})")
        except Exception as e:
            print(f"[Studio] Error loading target: {e}")
            self.set_toast(f"Error: {e}")

    def open_image_picker(self):
        """Opens native file dialog to select any image file."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askopenfilename(
                title="Select Camera Image to Calibrate",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
                    ("All files", "*.*")
                ]
            )
            root.destroy()
            if selected:
                self.load_new_path(Path(selected))
        except Exception as e:
            print(f"[Studio] Error opening file dialog: {e}")
            self.set_toast(f"Error: {e}")

    def open_folder_picker(self):
        """Opens native directory dialog to select any camera series folder."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(title="Select Camera Series Folder")
            root.destroy()
            if selected:
                self.load_new_path(Path(selected))
        except Exception as e:
            print(f"[Studio] Error opening folder dialog: {e}")
            self.set_toast(f"Error: {e}")

    def set_toast(self, msg: str):
        self.toast_msg = msg
        self.toast_time = time.time()

    # -------------------------------------------------------------------------
    # Top Ribbon Configuration
    # -------------------------------------------------------------------------
    def _build_ribbon_buttons(self):
        """Builds Canva/Figma-style top application ribbon buttons."""
        self.ribbon_buttons = []

        # 1. In-App Open File / Folder Buttons
        self.ribbon_buttons.append(TopRibbonButton(
            rect=(175, 11, 80, 34), text="🖼️ File", action="OPEN_IMAGE",
            bg_color=(36, 44, 56), hover_color=(50, 68, 90), font_scale=0.38
        ))
        self.ribbon_buttons.append(TopRibbonButton(
            rect=(260, 11, 80, 34), text="📁 Folder", action="OPEN_FOLDER",
            bg_color=(36, 44, 56), hover_color=(50, 68, 90), font_scale=0.38
        ))

        # 2. Carousel Switcher
        self.ribbon_buttons.append(TopRibbonButton(
            rect=(348, 11, 30, 34), text="◀", action="PREV_CAM",
            bg_color=(38, 42, 52), hover_color=(55, 65, 80), font_scale=0.45
        ))
        self.ribbon_buttons.append(TopRibbonButton(
            rect=(515, 11, 30, 34), text="▶", action="NEXT_CAM",
            bg_color=(38, 42, 52), hover_color=(55, 65, 80), font_scale=0.45
        ))

        # 3. Tool Switcher Pills
        self.ribbon_buttons.append(TopRibbonButton(
            rect=(555, 11, 125, 34), text="👆 Transform", action="MODE_TRANSFORM",
            bg_color=(35, 38, 48), active_color=(24, 160, 251), font_scale=0.40, is_toggle=True
        ))
        self.ribbon_buttons.append(TopRibbonButton(
            rect=(686, 11, 125, 34), text="📐 Distort Pin", action="MODE_DISTORT",
            bg_color=(35, 38, 48), active_color=(200, 100, 255), font_scale=0.40, is_toggle=True
        ))
        self.ribbon_buttons.append(TopRibbonButton(
            rect=(817, 11, 85, 34), text="🔍 Zoom", action="TOGGLE_ZOOM",
            bg_color=(35, 38, 48), active_color=(255, 180, 0), font_scale=0.40, is_toggle=True
        ))

        # 4. Opacity Controls
        self.ribbon_buttons.append(TopRibbonButton(
            rect=(908, 11, 26, 34), text="-", action="OPACITY_-",
            bg_color=(38, 40, 50), hover_color=(60, 60, 75), font_scale=0.48
        ))
        self.ribbon_buttons.append(TopRibbonButton(
            rect=(978, 11, 26, 34), text="+", action="OPACITY_+",
            bg_color=(38, 40, 50), hover_color=(60, 60, 75), font_scale=0.48
        ))

        # 5. Reset & Save
        self.ribbon_buttons.append(TopRibbonButton(
            rect=(1012, 11, 78, 34), text="🔄 Reset", action="RESET",
            bg_color=(45, 45, 55), hover_color=(65, 65, 75), font_scale=0.40
        ))
        self.ribbon_buttons.append(TopRibbonButton(
            rect=(1098, 10, 170, 36), text="💾 Save Calibration", action="SAVE",
            bg_color=(16, 140, 80), hover_color=(22, 175, 100), active_color=(16, 185, 95),
            text_color=(255, 255, 255), border_color=(40, 210, 110), font_scale=0.44
        ))

    # -------------------------------------------------------------------------
    # Viewport & Coordinate Geometry
    # -------------------------------------------------------------------------
    def _get_canvas_geometry(self) -> Dict[str, Any]:
        """Calculates scale and crop window for floor plan canvas below ribbon."""
        aspect = self.floor_w / self.floor_h
        avail_w = self.disp_w
        avail_h = self.canvas_h

        if aspect > (avail_w / avail_h):
            view_w = avail_w
            view_h = int(avail_w / aspect)
        else:
            view_h = avail_h
            view_w = int(avail_h * aspect)

        view_x0 = (avail_w - view_w) // 2
        view_y0 = self.ribbon_h + (avail_h - view_h) // 2

        if self.zoom_mode:
            centroid = np.mean(self.corner_pins, axis=0)
            crop_w = int(self.floor_w / self.zoom_factor)
            crop_h = int(self.floor_h / self.zoom_factor)
            crop_x0 = int(np.clip(centroid[0] - crop_w / 2.0, 0, self.floor_w - crop_w))
            crop_y0 = int(np.clip(centroid[1] - crop_h / 2.0, 0, self.floor_h - crop_h))
            return {
                "view_x0": view_x0, "view_y0": view_y0, "view_w": view_w, "view_h": view_h,
                "crop_x0": crop_x0, "crop_y0": crop_y0, "crop_w": crop_w, "crop_h": crop_h,
                "scale_x": view_w / crop_w, "scale_y": view_h / crop_h
            }
        else:
            return {
                "view_x0": view_x0, "view_y0": view_y0, "view_w": view_w, "view_h": view_h,
                "crop_x0": 0, "crop_y0": 0, "crop_w": self.floor_w, "crop_h": self.floor_h,
                "scale_x": view_w / self.floor_w, "scale_y": view_h / self.floor_h
            }

    def floor_to_disp(self, fx: float, fy: float, geo: Dict[str, Any]) -> Tuple[int, int]:
        dx = int(geo["view_x0"] + (fx - geo["crop_x0"]) * geo["scale_x"])
        dy = int(geo["view_y0"] + (fy - geo["crop_y0"]) * geo["scale_y"])
        return dx, dy

    def disp_to_floor(self, dx: int, dy: int, geo: Dict[str, Any]) -> Tuple[float, float]:
        fx = geo["crop_x0"] + ((dx - geo["view_x0"]) / max(1e-6, geo["scale_x"]))
        fy = geo["crop_y0"] + ((dy - geo["view_y0"]) / max(1e-6, geo["scale_y"]))
        return fx, fy

    # -------------------------------------------------------------------------
    # Bounding Box & Handle Hit-Testing
    # -------------------------------------------------------------------------
    def _compute_screen_handles(self, geo: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates on-screen display coordinates for bounding box handles."""
        disp_pins = np.array([
            self.floor_to_disp(px, py, geo) for px, py in self.corner_pins
        ], dtype=np.int32)

        centroid = np.mean(self.corner_pins, axis=0)
        disp_cx, disp_cy = self.floor_to_disp(centroid[0], centroid[1], geo)

        top_mid = (disp_pins[0] + disp_pins[1]) / 2.0
        bot_mid = (disp_pins[2] + disp_pins[3]) / 2.0
        v_sight = top_mid - bot_mid
        v_len = np.linalg.norm(v_sight)
        if v_len > 1e-3:
            v_dir = v_sight / v_len
        else:
            v_dir = np.array([0.0, -1.0])

        rot_knob_disp = top_mid + v_dir * 36.0

        return {
            "disp_pins": disp_pins,
            "center": (disp_cx, disp_cy),
            "top_mid": (int(top_mid[0]), int(top_mid[1])),
            "rot_knob": (int(rot_knob_disp[0]), int(rot_knob_disp[1]))
        }

    def _hit_test(self, mx: int, my: int, geo: Dict[str, Any]) -> Tuple[str, int, str]:
        if my < self.ribbon_h:
            for btn in self.ribbon_buttons:
                if btn.contains(mx, my):
                    return "RIBBON_BTN", -1, "POINTER"
            return "RIBBON_BG", -1, "POINTER"

        handles = self._compute_screen_handles(geo)
        disp_pins = handles["disp_pins"]
        rot_knob = handles["rot_knob"]

        if self.tool_mode == "DISTORT":
            for i, (dx, dy) in enumerate(disp_pins):
                if np.hypot(mx - dx, my - dy) <= 16:
                    return "CORNER_PIN", i, "DISTORT_PIN"
            if cv2.pointPolygonTest(disp_pins, (float(mx), float(my)), False) >= 0:
                return "BODY", -1, "MOVE"
            return "CANVAS", -1, "POINTER"

        if np.hypot(mx - rot_knob[0], my - rot_knob[1]) <= 14:
            return "ROT_HANDLE", -1, "ROTATE"

        for i, (dx, dy) in enumerate(disp_pins):
            dist = np.hypot(mx - dx, my - dy)
            if dist <= 12:
                p_type = "RESIZE_DIAG_1" if (i in [0, 2]) else "RESIZE_DIAG_2"
                return "CORNER_SCALE", i, p_type
            elif 13 <= dist <= 30:
                return "CORNER_ROT", i, "ROTATE"

        if cv2.pointPolygonTest(disp_pins, (float(mx), float(my)), False) >= 0:
            return "BODY", -1, "MOVE"

        return "CANVAS", -1, "POINTER"

    # -------------------------------------------------------------------------
    # Mouse Interaction Dispatcher
    # -------------------------------------------------------------------------
    def _mouse_callback(self, event, x, y, flags, param):
        geo = self._get_canvas_geometry()
        self.mouse_x, self.mouse_y = x, y

        hit_type, handle_idx, pointer_type = self._hit_test(x, y, geo)
        self.active_pointer_type = pointer_type

        if y < self.ribbon_h:
            hovered = None
            for btn in self.ribbon_buttons:
                if btn.contains(x, y):
                    hovered = btn.action
                    break
            self.hover_button_action = hovered
        else:
            self.hover_button_action = None

        fx, fy = self.disp_to_floor(x, y, geo)

        if event == cv2.EVENT_LBUTTONDOWN:
            if y < self.ribbon_h:
                for btn in self.ribbon_buttons:
                    if btn.contains(x, y):
                        self._handle_action(btn.action)
                        return

            handles = self._compute_screen_handles(geo)
            disp_cx, disp_cy = handles["center"]

            if hit_type == "BODY":
                self.drag_mode = "BODY"
                self.drag_start_floor = (fx, fy)
                self.start_state_cx = self.state.cx
                self.start_state_cy = self.state.cy
                self.start_pins = [list(p) for p in self.corner_pins]
                self.pointer_tooltip = f"Moving ({int(self.state.cx)}, {int(self.state.cy)})"

            elif hit_type == "CORNER_SCALE":
                self.drag_mode = "CORNER_SCALE"
                self.active_handle_idx = handle_idx
                self.drag_start_disp = (x, y)
                self.start_dist_to_center = max(10.0, float(np.hypot(x - disp_cx, y - disp_cy)))
                self.start_state_scale = self.state.scale
                self.pointer_tooltip = f"Scale: {self.state.scale * 100:.0f}%"

            elif hit_type in ["ROT_HANDLE", "CORNER_ROT"]:
                self.drag_mode = "ROTATE"
                self.start_state_rot = self.state.rotation_deg
                self.start_dist_to_center = float(np.degrees(np.arctan2(y - disp_cy, x - disp_cx)))
                self.pointer_tooltip = f"∠ {self.state.rotation_deg:.1f}°"

            elif hit_type == "CORNER_PIN":
                self.drag_mode = "CORNER_PIN"
                self.active_handle_idx = handle_idx
                self.pointer_tooltip = f"Pin #{handle_idx + 1}"

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_mode == "BODY":
                dx = fx - self.drag_start_floor[0]
                dy = fy - self.drag_start_floor[1]

                if self.tool_mode == "TRANSFORM":
                    self.state.cx = np.clip(self.start_state_cx + dx, 0, self.floor_w)
                    self.state.cy = np.clip(self.start_state_cy + dy, 0, self.floor_h)
                    self.corner_pins = self.state.to_corner_pins(self.floor_w, self.floor_h)
                else:
                    new_pins = []
                    for p in self.start_pins:
                        new_pins.append([
                            float(np.clip(p[0] + dx, 0, self.floor_w)),
                            float(np.clip(p[1] + dy, 0, self.floor_h))
                        ])
                    self.corner_pins = new_pins
                    self.state.cx = float(np.mean([p[0] for p in new_pins]))
                    self.state.cy = float(np.mean([p[1] for p in new_pins]))

                self._recompute_corner_homography()
                self.has_unsaved_changes = True
                self.pointer_tooltip = f"Pos: ({int(self.state.cx)}, {int(self.state.cy)})"

            elif self.drag_mode == "CORNER_SCALE":
                handles = self._compute_screen_handles(geo)
                disp_cx, disp_cy = handles["center"]
                curr_dist = max(10.0, float(np.hypot(x - disp_cx, y - disp_cy)))
                scale_ratio = curr_dist / max(1.0, self.start_dist_to_center)

                self.state.scale = float(np.clip(self.start_state_scale * scale_ratio, 0.20, 3.50))
                self.corner_pins = self.state.to_corner_pins(self.floor_w, self.floor_h)
                self._recompute_corner_homography()
                self.has_unsaved_changes = True
                self.pointer_tooltip = f"Scale: {self.state.scale * 100:.0f}%"

            elif self.drag_mode == "ROTATE":
                handles = self._compute_screen_handles(geo)
                disp_cx, disp_cy = handles["center"]
                curr_angle = float(np.degrees(np.arctan2(y - disp_cy, x - disp_cx)))
                d_theta = curr_angle - self.start_dist_to_center

                self.state.rotation_deg = float((self.start_state_rot + d_theta) % 360.0)
                self.corner_pins = self.state.to_corner_pins(self.floor_w, self.floor_h)
                self._recompute_corner_homography()
                self.has_unsaved_changes = True
                self.pointer_tooltip = f"∠ {self.state.rotation_deg:.1f}°"

            elif self.drag_mode == "CORNER_PIN" and self.active_handle_idx >= 0:
                self.corner_pins[self.active_handle_idx] = [
                    float(np.clip(fx, 0, self.floor_w)),
                    float(np.clip(fy, 0, self.floor_h))
                ]
                self.state = TransformState.from_corner_pins(
                    self.corner_pins, self.w0, self.h0, alpha=self.state.alpha
                )
                self._recompute_corner_homography()
                self.has_unsaved_changes = True
                self.pointer_tooltip = f"Pin #{self.active_handle_idx + 1}"

            else:
                self.pointer_tooltip = ""

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_mode = None
            self.active_handle_idx = -1
            self.pointer_tooltip = ""

    # -------------------------------------------------------------------------
    # Action Handlers
    # -------------------------------------------------------------------------
    def _handle_action(self, action: str):
        if action == "OPEN_IMAGE":
            self.open_image_picker()
        elif action == "OPEN_FOLDER":
            self.open_folder_picker()
        elif action == "PREV_CAM":
            self.switch_series(-1)
        elif action == "NEXT_CAM":
            self.switch_series(1)
        elif action == "MODE_TRANSFORM":
            self.tool_mode = "TRANSFORM"
            self.set_toast("Tool: Transform (Move, Corner Scale, Top Rotate)")
        elif action == "MODE_DISTORT":
            self.tool_mode = "DISTORT"
            self.set_toast("Tool: Perspective Distort (Drag Corner Pins)")
        elif action == "TOGGLE_ZOOM":
            self.zoom_mode = not self.zoom_mode
            self.set_toast(f"Focus Zoom: {'ON (2.5x)' if self.zoom_mode else 'OFF'}")
        elif action == "OPACITY_-":
            self.state.alpha = max(0.10, self.state.alpha - 0.10)
            self.set_toast(f"Opacity: {self.state.alpha * 100:.0f}%")
        elif action == "OPACITY_+":
            self.state.alpha = min(0.95, self.state.alpha + 0.10)
            self.set_toast(f"Opacity: {self.state.alpha * 100:.0f}%")
        elif action == "RESET":
            self.state.cx = self.floor_w / 2.0
            self.state.cy = self.floor_h / 2.0
            self.state.rotation_deg = 0.0
            self.state.scale = 1.0
            self.state.aspect = 1.0
            self.state.keystone = 1.0
            self.corner_pins = self.state.to_corner_pins(self.floor_w, self.floor_h)
            self._recompute_corner_homography()
            self.has_unsaved_changes = True
            self.set_toast("Reset handles to default concourse center")
        elif action == "SAVE":
            self.save()

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------
    def render(self) -> np.ndarray:
        """Renders the Canva/Figma unified studio canvas."""
        canvas = np.full((self.disp_h, self.disp_w, 3), 26, dtype=np.uint8)
        geo = self._get_canvas_geometry()

        # 1. Render Floor Plan Concourse
        if self.zoom_mode:
            cropped = self.floor_img_orig[
                geo["crop_y0"]:geo["crop_y0"] + geo["crop_h"],
                geo["crop_x0"]:geo["crop_x0"] + geo["crop_w"]
            ]
            floor_resized = cv2.resize(cropped, (geo["view_w"], geo["view_h"]), interpolation=cv2.INTER_LINEAR)
        else:
            floor_resized = cv2.resize(self.floor_img_orig, (geo["view_w"], geo["view_h"]), interpolation=cv2.INTER_LINEAR)

        vx0, vy0 = geo["view_x0"], geo["view_y0"]
        vw, vh = geo["view_w"], geo["view_h"]
        canvas[vy0:vy0 + vh, vx0:vx0 + vw] = floor_resized

        # 2. Warp Camera Feed inside Floor Viewport
        handles = self._compute_screen_handles(geo)
        disp_pins = handles["disp_pins"]

        if self.projector.is_calibrated:
            try:
                cam_corners = np.array([
                    [0.0, 0.0],
                    [float(self.cam_w), 0.0],
                    [float(self.cam_w), float(self.cam_h)],
                    [0.0, float(self.cam_h)]
                ], dtype=np.float32)
                H_disp, _ = cv2.findHomography(cam_corners, disp_pins.astype(np.float32))

                if H_disp is not None:
                    warped = cv2.warpPerspective(
                        self.cam_img_orig, H_disp, (self.disp_w, self.disp_h), flags=cv2.INTER_LINEAR
                    )
                    mask = np.zeros((self.disp_h, self.disp_w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, disp_pins, 255)

                    mask_bool = (mask > 0) & (np.arange(self.disp_h)[:, None] >= self.ribbon_h)
                    alpha = self.state.alpha
                    canvas[mask_bool] = cv2.addWeighted(
                        canvas, 1.0 - alpha, warped, alpha, 0
                    )[mask_bool]
            except Exception:
                pass

        # 3. Render Figma-Style Bounding Box & Handles
        cv2.polylines(canvas, [disp_pins], isClosed=True, color=(251, 160, 24), thickness=2, lineType=cv2.LINE_AA)

        top_mid = handles["top_mid"]
        disp_cx, disp_cy = handles["center"]
        cv2.arrowedLine(
            canvas, (disp_cx, disp_cy), (top_mid[0], top_mid[1]),
            (0, 255, 120), 2, line_type=cv2.LINE_AA, tipLength=0.22
        )

        if self.tool_mode == "TRANSFORM":
            rot_knob = handles["rot_knob"]
            cv2.line(canvas, (top_mid[0], top_mid[1]), (rot_knob[0], rot_knob[1]), (251, 160, 24), 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, rot_knob, 7, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, rot_knob, 7, (251, 160, 24), 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, rot_knob, 2, (251, 160, 24), -1, lineType=cv2.LINE_AA)

            for dx, dy in disp_pins:
                cv2.rectangle(canvas, (dx - 5, dy - 5), (dx + 5, dy + 5), (255, 255, 255), -1)
                cv2.rectangle(canvas, (dx - 5, dy - 5), (dx + 5, dy + 5), (251, 160, 24), 2, lineType=cv2.LINE_AA)

        else:
            pin_cols = [(0, 255, 255), (255, 160, 0), (0, 200, 0), (200, 0, 255)]
            pin_names = ["TL", "TR", "BR", "BL"]
            for i, (dx, dy) in enumerate(disp_pins):
                cv2.circle(canvas, (dx, dy), 8, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (dx, dy), 6, pin_cols[i], -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (dx, dy), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.putText(canvas, pin_names[i], (dx + 10, dy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Top Application Ribbon
        cv2.rectangle(canvas, (0, 0), (self.disp_w, self.ribbon_h), (28, 28, 36), -1)
        cv2.line(canvas, (0, self.ribbon_h), (self.disp_w, self.ribbon_h), (50, 50, 62), 1, cv2.LINE_AA)

        cv2.circle(canvas, (24, 28), 6, (24, 160, 251), -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "CrowdSafe Studio", (36, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (250, 250, 255), 2, cv2.LINE_AA)

        # Active Target Label in Ribbon
        display_label = self.target_name
        if len(display_label) > 14:
            display_label = display_label[:12] + ".."
        tag_str = f"[{display_label}] {self.current_target_idx + 1}/{len(self.targets)}"
        cv2.putText(canvas, tag_str, (384, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1, cv2.LINE_AA)

        for btn in self.ribbon_buttons:
            is_hover = (self.hover_button_action == btn.action)
            is_active = False
            if btn.action == "MODE_TRANSFORM" and self.tool_mode == "TRANSFORM":
                is_active = True
            elif btn.action == "MODE_DISTORT" and self.tool_mode == "DISTORT":
                is_active = True
            elif btn.action == "TOGGLE_ZOOM" and self.zoom_mode:
                is_active = True

            btn.draw(canvas, is_hover=is_hover, is_active=is_active)

        cv2.putText(canvas, f"{self.state.alpha * 100:.0f}%", (938, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 220), 1, cv2.LINE_AA)

        # 5. Live Inspector Badges (Floating Bottom Left Pill)
        prop_y = self.disp_h - 22
        calib_tag = "✅ Calibrated" if self.active_target.existing_calib_file else "⚠️ Uncalibrated"

        cv2.rectangle(canvas, (15, self.disp_h - 40), (660, self.disp_h - 8), (20, 20, 26), -1)
        cv2.rectangle(canvas, (15, self.disp_h - 40), (660, self.disp_h - 8), (55, 55, 70), 1, lineType=cv2.LINE_AA)

        stat_txt = f"[{self.target_name}]  |  X:{int(self.state.cx)} Y:{int(self.state.cy)}  |  Scale:{self.state.scale*100:.0f}%  |  ∠{self.state.rotation_deg:.1f}°  |  {calib_tag}"
        cv2.putText(canvas, stat_txt, (25, prop_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 240), 1, cv2.LINE_AA)

        # 6. Toast Banner
        now = time.time()
        if (now - self.toast_time) < 2.8 and self.toast_msg:
            (tw, _), _ = cv2.getTextSize(self.toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            tx = self.disp_w - tw - 35
            cv2.rectangle(canvas, (tx - 12, self.disp_h - 40), (self.disp_w - 15, self.disp_h - 8), (15, 60, 35), -1)
            cv2.rectangle(canvas, (tx - 12, self.disp_h - 40), (self.disp_w - 15, self.disp_h - 8), (40, 180, 80), 1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, self.toast_msg, (tx, prop_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 255, 160), 1, cv2.LINE_AA)

        # 7. Dynamic Pointer Icon
        DynamicPointerIcon.draw(canvas, self.mouse_x, self.mouse_y, self.active_pointer_type, self.pointer_tooltip)

        return canvas

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def save(self) -> bool:
        """Saves calibrated homography matrix and coordinates to disk."""
        if not self.projector.is_calibrated:
            self.set_toast("Error: Calibration is degenerate!")
            return False

        centroid = np.mean(self.corner_pins, axis=0)
        self.projector.metadata = {
            "series_name": self.target_name,
            "source_path": str(self.active_target.path),
            "is_single_image": self.active_target.is_file,
            "calibration_mode": "CANVA_FIGMA_STUDIO",
            "num_points": len(self.projector.src_points),
            "floor_centroid": [float(centroid[0]), float(centroid[1])],
            "rotation_deg": float(self.state.rotation_deg),
            "scale": float(self.state.scale),
            "keystone": float(self.state.keystone),
            "aspect": float(self.state.aspect),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.projector.save_calibration(self.output_path)

        if self.active_target.is_file and not self.active_target.fallback_calib_path.exists():
            try:
                self.projector.save_calibration(self.active_target.fallback_calib_path)
            except Exception:
                pass

        self.has_unsaved_changes = False
        self.set_toast(f"✅ Saved Calibration for [{self.target_name}]!")
        print(f"✅ [Studio] Calibration saved to: {self.output_path}")
        return True

    # -------------------------------------------------------------------------
    # Main Event Loop
    # -------------------------------------------------------------------------
    def run(self):
        """Runs the Figma/Canva creative studio."""
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, self.disp_w, self.disp_h)

        init_frame = self.render()
        cv2.imshow(self.win_name, init_frame)
        for _ in range(5):
            cv2.waitKey(10)

        try:
            cv2.setMouseCallback(self.win_name, self._mouse_callback)
        except cv2.error:
            cv2.waitKey(50)
            cv2.setMouseCallback(self.win_name, self._mouse_callback)

        print("\n" + "=" * 68)
        print(" CrowdSafe Studio - Canva & Figma-Style Visual Calibration Active")
        print("=" * 68)
        print(" Universal Controls:")
        print("   🖱️ [ 🖼️ File ] / [ 📁 Folder ]: Open any image or custom directory on disk")
        print("   🖱️ Drag Inside Feed         : Freely move & drop camera feed anywhere")
        print("   🖱️ Drag White Corner Handles: Enlarge / Shrink coverage")
        print("   🖱️ Drag Top Lollipop Handle : Rotate camera feed (or drag outside corners)")
        print("   🖱️ Mode Pills               : Switch between Transform & Distort Pin")
        print("   🖱️ Navigation Pills         : [◀] Previous | Next [▶] across all loaded targets")
        print("   🖱️ 1-Click Save             : Click [💾 Save Calibration] button or press 's'")
        print("   'q' or Esc                  : Exit studio")
        print("=" * 68 + "\n")

        while True:
            try:
                if cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            frame = self.render()
            try:
                cv2.imshow(self.win_name, frame)
            except cv2.error:
                break

            key = cv2.waitKey(16)
            if key == -1:
                continue

            k8 = key & 0xFF
            if k8 in [ord('q'), 27]:
                break
            elif k8 == ord('s'):
                self.save()
            elif k8 in [ord('n'), ord(']')]:
                self.switch_series(1)
            elif k8 in [ord('p'), ord('[')]:
                self.switch_series(-1)
            elif k8 == ord('z'):
                self._handle_action("TOGGLE_ZOOM")

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# Compatibility alias
CalibrationGUI = CanvaFigmaCalibrationStudio


def parse_args():
    parser = argparse.ArgumentParser(description="CrowdSafe Canva/Figma-Style Universal Visual Calibration Studio")
    parser.add_argument("--series", type=str, default=None, help="Camera series number, custom folder, or image file")
    parser.add_argument("--source", type=str, default=None, help="Camera series folder, custom directory, or image path")
    parser.add_argument("--floor-plan", type=str, default=str(DEFAULT_FLOOR_PLAN), help="Path to 2D floor plan image")
    parser.add_argument("--list", action="store_true", help="List all available camera targets")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        list_available_series()
        return

    target_input = args.series or args.source or "DataSet/40"

    floor_plan_path = Path(args.floor_plan)
    if not floor_plan_path.is_file():
        floor_plan_path = DEFAULT_FLOOR_PLAN

    floor_img = cv2.imread(str(floor_plan_path))
    if floor_img is None:
        print(f"[Studio] Error: Failed to load floor plan image: {floor_plan_path}")
        sys.exit(1)

    studio = CanvaFigmaCalibrationStudio(
        floor_image=floor_img,
        source_dir=Path(target_input)
    )
    studio.run()


if __name__ == "__main__":
    main()
