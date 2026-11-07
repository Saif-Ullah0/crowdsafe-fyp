"""
High-Freedom OpenCV Visualizer & Overlay Compositor for CrowdSafe MVP
Renders customizable multi-layer visual filters:
1. Base CCTV video feed
2. Alpha-blended continuous Jet density heatmap (Toggle: 'h')
3. Zone bounding boxes & Fruin risk color tags (Toggle: 'b')
4. Head detection dots & congestion cluster contours (Toggle: 'd')
5. Tactical Safety HUD banner & flashing alert warnings (Toggle: 'z')
"""
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np
from config import VisualizerConfig, DENSITY_LEVELS, OUTPUT_DIR

class Visualizer:
    """Multi-layer OpenCV overlay compositor with interactive keyboard controls."""

    def __init__(self, config: Optional[VisualizerConfig] = None):
        self.config = config if config is not None else VisualizerConfig()
        self.show_heatmap = self.config.show_heatmap
        self.show_zones = self.config.show_zones
        self.show_dots = self.config.show_dots
        self.show_hud = self.config.show_hud
        self.heatmap_alpha = self.config.heatmap_alpha
        self.window_name = self.config.window_name
        self.headless = self.config.headless
        self.is_paused = False

        if not self.headless:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1024, 768)

    def render(
        self,
        frame_bgr: np.ndarray,
        density_map: np.ndarray,
        spatial_analysis: Dict[str, Any],
        fps: float = 1.0,
        latency_ms: float = 0.0,
        active_alert: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Composes all enabled visual layers onto the base frame.

        Returns:
            Annotated BGR composite frame.
        """
        h, w = frame_bgr.shape[:2]
        canvas = frame_bgr.copy()

        # -------------------------------------------------------------
        # Layer 1: Alpha-blended Density Heatmap (Jet Colormap)
        # -------------------------------------------------------------
        if self.show_heatmap and density_map is not None:
            canvas = self._apply_heatmap_overlay(canvas, density_map)

        # -------------------------------------------------------------
        # Layer 2: Detection Dots (Peak Centroids) & Cluster Contours
        # -------------------------------------------------------------
        if self.show_dots:
            # Draw cluster bounding boxes
            for cluster in spatial_analysis.get("congestion_clusters", []):
                cx1, cy1, cx2, cy2 = cluster["pixel_coords"]
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (255, 0, 255), 1, cv2.LINE_AA)
            
            # Draw individual density peak dots
            for px, py in spatial_analysis.get("peak_points", []):
                cv2.circle(canvas, (px, py), 4, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 5, (0, 0, 0), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # Layer 3: Zone Bounding Boxes & Fruin Risk Labels
        # -------------------------------------------------------------
        if self.show_zones:
            for zone in spatial_analysis.get("zones", []):
                x1_n, y1_n, x2_n, y2_n = zone["bbox_norm"]
                zx1, zy1 = int(x1_n * w), int(y1_n * h)
                zx2, zy2 = int(x2_n * w), int(y2_n * h)

                risk = zone["risk_level"]
                color = DENSITY_LEVELS[risk]["color"]

                # Draw zone bounding box
                cv2.rectangle(canvas, (zx1, zy1), (zx2, zy2), color, 2)

                # Draw semi-transparent header tag
                tag_text = f"{zone['name']} | {zone['density_per_m2']} p/m2 [{risk}]"
                self._draw_text_tag(canvas, tag_text, (zx1, max(zy1 - 10, 20)), color)

        # -------------------------------------------------------------
        # Layer 4: Tactical HUD & Alert Warning Banner
        # -------------------------------------------------------------
        if self.show_hud:
            canvas = self._draw_tactical_hud(canvas, spatial_analysis, fps, latency_ms, active_alert)

        return canvas

    def _apply_heatmap_overlay(self, frame: np.ndarray, density_map: np.ndarray) -> np.ndarray:
        """Normalizes density map, applies Jet colormap, and blends with alpha."""
        h, w = frame.shape[:2]
        max_v = float(np.max(density_map))
        if max_v > 0:
            norm = (density_map / max_v * 255.0).astype(np.uint8)
        else:
            norm = np.zeros(density_map.shape[:2], dtype=np.uint8)

        color_heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        resized_heatmap = cv2.resize(color_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        blended = cv2.addWeighted(frame, 1.0 - self.heatmap_alpha, resized_heatmap, self.heatmap_alpha, 0)
        return blended

    def _draw_text_tag(self, img: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int]):
        """Renders text with dark background pill for high contrast."""
        x, y = pos
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        # Background rect
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + baseline), (25, 25, 25), -1)
        # Left color indicator bar
        cv2.rectangle(img, (x, y - th - 4), (x + 3, y + baseline), color, -1)
        # Text string
        cv2.putText(img, text, (x + 6, y - 2), font, scale, (240, 240, 240), thickness, cv2.LINE_AA)

    def _draw_tactical_hud(
        self,
        canvas: np.ndarray,
        spatial_analysis: Dict[str, Any],
        fps: float,
        latency_ms: float,
        active_alert: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Renders top status dashboard and bottom alert notification banner."""
        h, w = canvas.shape[:2]
        
        # 1. Top HUD Bar
        hud_bar = canvas[:50, :].copy()
        dark_overlay = np.zeros_like(hud_bar)
        cv2.addWeighted(hud_bar, 0.25, dark_overlay, 0.75, 0, hud_bar)
        canvas[:50, :] = hud_bar

        overall_risk = spatial_analysis.get("overall_risk", "LOW")
        risk_color = DENSITY_LEVELS[overall_risk]["color"]
        total_people = sum(z["count"] for z in spatial_analysis.get("zones", []))
        peak_density = spatial_analysis.get("peak_density_m2", 0.0)

        # Status indicators
        cv2.putText(canvas, "CROWDSAFE SENTINEL", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        metrics_str = f"Count: {total_people:.0f}  |  Peak: {peak_density:.1f} p/m2  |  Rate: {fps:.1f} FPS ({latency_ms:.0f}ms)"
        cv2.putText(canvas, metrics_str, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # System Risk Badge
        badge_text = f"RISK: {overall_risk}"
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        badge_x = w - bw - 25
        cv2.rectangle(canvas, (badge_x - 8, 10), (badge_x + bw + 8, 40), risk_color, -1)
        badge_text_color = (0, 0, 0) if overall_risk in ["LOW", "MODERATE"] else (255, 255, 255)
        cv2.putText(canvas, badge_text, (badge_x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, badge_text_color, 2, cv2.LINE_AA)

        # 2. Bottom Alert Banner (Flashing when alert is active)
        if active_alert or overall_risk in ["HIGH", "CRITICAL"]:
            banner_h = 36
            banner_roi = canvas[h - banner_h:, :].copy()
            b_color = (0, 0, 200) if overall_risk == "CRITICAL" else (0, 140, 255)
            cv2.rectangle(canvas, (0, h - banner_h), (w, h), b_color, -1)
            
            alert_msg = (
                f"🚨 CRITICAL ALERT: {active_alert.get('zone_name', 'Bottleneck')} exceeds safe threshold! Divert crowd flow."
                if active_alert else
                f"⚠️ CAUTION: High crowd concentration ({peak_density:.1f} p/m2) in monitored sector."
            )
            cv2.putText(canvas, alert_msg, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

    def handle_key_events(self, frame_annotated: np.ndarray, wait_ms: int = 1) -> bool:
        """
        Processes interactive keyboard shortcuts.
        Returns:
            False if user requested exit ('q' or ESC), True otherwise.
        """
        if self.headless:
            return True

        cv2.imshow(self.window_name, frame_annotated)
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in [ord('q'), 27]:  # 'q' or ESC
            return False
        elif key == ord('h'):
            self.show_heatmap = not self.show_heatmap
            print(f"[Visualizer] Heatmap overlay: {'ON' if self.show_heatmap else 'OFF'}")
        elif key == ord('b'):
            self.show_zones = not self.show_zones
            print(f"[Visualizer] Zone bounding boxes: {'ON' if self.show_zones else 'OFF'}")
        elif key == ord('d'):
            self.show_dots = not self.show_dots
            print(f"[Visualizer] Head detection dots: {'ON' if self.show_dots else 'OFF'}")
        elif key == ord('z'):
            self.show_hud = not self.show_hud
            print(f"[Visualizer] Safety HUD banner: {'ON' if self.show_hud else 'OFF'}")
        elif key == ord('s'):
            # Save snapshot
            snap_path = OUTPUT_DIR / f"snapshot_{int(cv2.getTickCount())}.png"
            cv2.imwrite(str(snap_path), frame_annotated)
            print(f"📸 [Visualizer] Saved preview snapshot to: {snap_path}")
        elif key == ord(' '):  # Spacebar Pause / Resume
            self.is_paused = not self.is_paused
            print(f"[Visualizer] {'PAUSED' if self.is_paused else 'RESUMED'}")
            while self.is_paused:
                pk = cv2.waitKey(100) & 0xFF
                if pk == ord(' '):
                    self.is_paused = False
                    print("[Visualizer] RESUMED")
                elif pk in [ord('q'), 27]:
                    return False

        return True

    def close(self):
        """Closes all OpenCV display windows."""
        if not self.headless:
            cv2.destroyAllWindows()
