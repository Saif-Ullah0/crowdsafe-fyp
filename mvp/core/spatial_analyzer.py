"""
Spatial Risk & Zone Analyzer for CrowdSafe MVP
Calculates crowd density per square meter across designated venue ROIs and
extracts congestion clusters and local density peak points.
"""
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from config import DENSITY_LEVELS, DEFAULT_ZONES

class SpatialAnalyzer:
    """Analyzes 2D density maps to compute zone-based risk and detect physical congestion hotspots."""

    def __init__(self, zones: List[Dict] = None, instant_critical_threshold: float = 5.5):
        self.zones = zones if zones is not None else DEFAULT_ZONES
        self.instant_critical_threshold = instant_critical_threshold

    def analyze(self, density_map: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Analyzes the density map across defined zones and identifies congestion clusters.

        Args:
            density_map: 2D numpy array of estimated density
            frame_shape: (height, width) of the original camera frame

        Returns:
            Dict containing:
                - 'zones': List of zone metrics (id, name, count, area, density_per_m2, risk_level)
                - 'overall_risk': Highest risk level ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')
                - 'is_flash_spike': Boolean indicating if an instant flash surge occurred
                - 'peak_density_m2': Maximum density observed in any single zone
                - 'congestion_clusters': List of bounding boxes for high-density clusters
                - 'peak_points': List of (x, y) coordinates of density peaks (head dots)
        """
        frame_h, frame_w = frame_shape[:2]
        map_h, map_w = density_map.shape[:2]
        
        zone_results = []
        overall_risk = "LOW"
        peak_density = 0.0
        is_flash_spike = False

        for zone in self.zones:
            x1_norm, y1_norm, x2_norm, y2_norm = zone["bbox"]
            
            # Map normalized coordinates to density map indices
            x1 = max(0, int(x1_norm * map_w))
            y1 = max(0, int(y1_norm * map_h))
            x2 = min(map_w, int(x2_norm * map_w))
            y2 = min(map_h, int(y2_norm * map_h))

            # Integrate density within zone sub-grid
            zone_density_patch = density_map[y1:y2, x1:x2]
            zone_count = float(np.sum(zone_density_patch)) if zone_density_patch.size > 0 else 0.0
            
            area_m2 = zone.get("estimated_area_m2", 25.0)
            density_per_m2 = round(zone_count / max(area_m2, 1.0), 2)
            
            if density_per_m2 > peak_density:
                peak_density = density_per_m2

            if density_per_m2 >= self.instant_critical_threshold:
                is_flash_spike = True

            risk_level = self._classify_risk(density_per_m2)
            overall_risk = self._escalate_risk(overall_risk, risk_level)

            zone_results.append({
                "id": zone["id"],
                "name": zone["name"],
                "bbox_norm": zone["bbox"],
                "count": round(zone_count, 1),
                "area_m2": area_m2,
                "density_per_m2": density_per_m2,
                "risk_level": risk_level
            })

        # Cluster and Peak Detection (for bounding boxes and head dots)
        clusters, peak_points = self._detect_clusters_and_peaks(density_map, frame_w, frame_h)

        return {
            "zones": zone_results,
            "overall_risk": overall_risk,
            "is_flash_spike": is_flash_spike,
            "peak_density_m2": peak_density,
            "congestion_clusters": clusters,
            "peak_points": peak_points
        }

    def _classify_risk(self, density_per_m2: float) -> str:
        """Determines risk level according to Fruin's Level of Service standards."""
        if density_per_m2 < DENSITY_LEVELS["LOW"]["max"]:
            return "LOW"
        elif density_per_m2 < DENSITY_LEVELS["MODERATE"]["max"]:
            return "MODERATE"
        elif density_per_m2 < DENSITY_LEVELS["HIGH"]["max"]:
            return "HIGH"
        else:
            return "CRITICAL"

    def _escalate_risk(self, current: str, candidate: str) -> str:
        """Returns the higher risk level between current and candidate."""
        order = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        return candidate if order.index(candidate) > order.index(current) else current

    def _detect_clusters_and_peaks(
        self,
        density_map: np.ndarray,
        frame_w: int,
        frame_h: int
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
        """
        Extracts high-density cluster bounding boxes and local peak centroids (dots).
        """
        map_h, map_w = density_map.shape[:2]
        max_val = float(np.max(density_map))
        if max_val <= 0:
            return [], []

        norm_map = (density_map / max_val * 255.0).astype(np.uint8)
        
        # Threshold top 30% densest areas to find congestion blobs
        _, thresh = cv2.threshold(norm_map, 175, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        clusters = []
        scale_x = frame_w / max(map_w, 1)
        scale_y = frame_h / max(map_h, 1)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 4:  # Filter out single-pixel noise
                bx, by, bw, bh = cv2.boundingRect(cnt)
                clusters.append({
                    "bbox_norm": (bx / map_w, by / map_h, (bx + bw) / map_w, (by + bh) / map_h),
                    "pixel_coords": (int(bx * scale_x), int(by * scale_y), int((bx + bw) * scale_x), int((by + bh) * scale_y)),
                    "area": float(area)
                })

        # Local Maxima / Peak Detection for discrete Head Dots
        # Find points where density is a local peak
        peak_points = []
        blurred = cv2.GaussianBlur(norm_map, (5, 5), 0)
        # Dilate to find local maxima
        dilated = cv2.dilate(blurred, np.ones((5, 5), np.uint8))
        local_peaks = (blurred == dilated) & (blurred > 70)
        peak_coords = np.argwhere(local_peaks)

        for py, px in peak_coords:
            frame_px = int(px * scale_x)
            frame_py = int(py * scale_y)
            peak_points.append((frame_px, frame_py))

        return clusters, peak_points
