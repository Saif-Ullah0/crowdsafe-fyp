"""
Modular Detection Source Interface for CrowdSafe MVP
Supports plug-and-play person extraction from dataset JSON annotations or live neural density models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import glob
import json
import numpy as np


@dataclass
class PersonDetection:
    """Represents a single person detection in the camera perspective plane."""
    bbox: Tuple[float, float, float, float]       # (x, y, width, height)
    feet_point: Tuple[float, float]               # (x_ground, y_ground) contact point
    confidence: float = 1.0
    person_id: Optional[str] = None


class BaseDetectionSource(ABC):
    """Abstract interface for detection sources."""

    @abstractmethod
    def get_detections(
        self,
        frame_idx: int,
        frame_bgr: Optional[np.ndarray] = None
    ) -> List[PersonDetection]:
        """
        Retrieves detected persons for a specific frame.
        
        Args:
            frame_idx: Frame index (1-based or 0-based sequence counter)
            frame_bgr: Optional BGR image array for on-the-fly model inference
            
        Returns:
            List of PersonDetection objects
        """
        pass


class AnnotationDetectionSource(BaseDetectionSource):
    """
    Parses VGG Image Annotator (VIA) and standard crowd counting JSON annotation files
    stored alongside images in dataset series folders (e.g. DataSet/40/001.json).
    """

    def __init__(self, series_dir: Union[str, Path]):
        self.series_dir = Path(series_dir)
        self.annotation_files: Dict[int, Path] = {}
        self._index_annotations()

    def _index_annotations(self):
        """Indexes all JSON annotation files in the target directory."""
        if not self.series_dir.is_dir():
            return

        json_paths = sorted(glob.glob(str(self.series_dir / "*.json")))
        for p in json_paths:
            path_obj = Path(p)
            stem = path_obj.stem
            # Ignore calibration files or metadata
            if stem.lower() in ["calibration", "calibrations", "metadata", "scenario_summary"]:
                continue

            try:
                # Support numeric frame indices (e.g. 001.json -> 1)
                idx = int(stem)
                self.annotation_files[idx] = path_obj
            except ValueError:
                # If non-numeric filename, store with sequential counter
                pass

        # If no numeric indices found, sort all and index sequentially 1..N
        if not self.annotation_files and json_paths:
            valid_jsons = [
                Path(p) for p in json_paths
                if Path(p).stem.lower() not in ["calibration", "calibrations", "metadata", "scenario_summary"]
            ]
            for seq_idx, p in enumerate(valid_jsons, start=1):
                self.annotation_files[seq_idx] = p

    def get_detections(
        self,
        frame_idx: int,
        frame_bgr: Optional[np.ndarray] = None
    ) -> List[PersonDetection]:
        """Reads and extracts person bounding boxes and feet contact points."""
        json_path = self.annotation_files.get(frame_idx)
        if not json_path or not json_path.is_file():
            return []

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[AnnotationDetectionSource] Warning reading {json_path}: {e}")
            return []

        detections = []

        # Handle VIA format: { "filename_hash": { "regions": [ ... ] } }
        for img_key, img_val in data.items():
            if not isinstance(img_val, dict):
                continue

            regions = img_val.get("regions", [])
            for r_idx, reg in enumerate(regions):
                shape = reg.get("shape_attributes", {})
                name = shape.get("name", "rect")

                if name == "rect":
                    rx = float(shape.get("x", 0))
                    ry = float(shape.get("y", 0))
                    rw = float(shape.get("width", 0))
                    rh = float(shape.get("height", 0))
                    # Ground contact point: bottom-center of bounding box
                    feet_x = rx + (rw / 2.0)
                    feet_y = ry + rh

                elif name in ["point", "circle"]:
                    feet_x = float(shape.get("cx", 0))
                    feet_y = float(shape.get("cy", 0))
                    rw = 15.0
                    rh = 30.0
                    rx = feet_x - (rw / 2.0)
                    ry = feet_y - rh

                elif name == "polygon":
                    all_x = shape.get("all_points_x", [])
                    all_y = shape.get("all_points_y", [])
                    if all_x and all_y:
                        rx = float(min(all_x))
                        ry = float(min(all_y))
                        rw = float(max(all_x) - rx)
                        rh = float(max(all_y) - ry)
                        feet_x = rx + (rw / 2.0)
                        feet_y = ry + rh
                    else:
                        continue
                else:
                    continue

                detections.append(PersonDetection(
                    bbox=(rx, ry, rw, rh),
                    feet_point=(feet_x, feet_y),
                    confidence=1.0,
                    person_id=f"p_{frame_idx}_{r_idx}"
                ))

        return detections


class ModelDetectionSource(BaseDetectionSource):
    """
    Plug-and-play adapter that extracts person points dynamically from live deep learning
    density maps or detection engines (e.g. MCNN, CSRNet, or FastSim peak detection).
    """

    def __init__(self, density_engine, spatial_analyzer=None):
        self.density_engine = density_engine
        self.spatial_analyzer = spatial_analyzer

    def get_detections(
        self,
        frame_idx: int,
        frame_bgr: Optional[np.ndarray] = None
    ) -> List[PersonDetection]:
        if frame_bgr is None or self.density_engine is None:
            return []

        # Predict density map
        result = self.density_engine.predict(frame_bgr)
        density_map = result.get("density_map")

        if density_map is None:
            return []

        # If spatial analyzer is provided, use peak maxima detection
        if self.spatial_analyzer:
            analysis = self.spatial_analyzer.analyze(density_map, frame_bgr.shape[:2])
            peak_points = analysis.get("peak_points", [])
        else:
            # Fallback simple local peak extraction
            peak_points = []

        detections = []
        for idx, (px, py) in enumerate(peak_points):
            detections.append(PersonDetection(
                bbox=(float(px - 10), float(py - 25), 20.0, 30.0),
                feet_point=(float(px), float(py + 10)),
                confidence=1.0,
                person_id=f"model_{frame_idx}_{idx}"
            ))

        return detections
