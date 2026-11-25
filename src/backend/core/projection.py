"""
Planar Homography & Floor Plan Projection Engine for CrowdSafe MVP
Provides projective transformations between camera perspective frames and 2D architectural floor plans.
"""
from typing import Tuple, List, Optional, Dict, Any, Union
from pathlib import Path
import json
import numpy as np
import cv2


class PlanarProjector:
    """
    Manages 3x3 projective homography transformations mapping camera coordinates (u, v)
    to physical 2D floor plan pixel coordinates (X, Y).
    """

    def __init__(
        self,
        H: Optional[np.ndarray] = None,
        camera_shape: Optional[Tuple[int, int]] = None,
        floor_shape: Optional[Tuple[int, int]] = None,
        src_points: Optional[List[Tuple[float, float]]] = None,
        dst_points: Optional[List[Tuple[float, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.H = np.asarray(H, dtype=np.float64) if H is not None else None
        self.H_inv = np.linalg.pinv(self.H) if self.H is not None else None
        self.camera_shape = tuple(camera_shape) if camera_shape else None  # (H, W)
        self.floor_shape = tuple(floor_shape) if floor_shape else None      # (H, W)
        self.src_points = list(src_points) if src_points else []
        self.dst_points = list(dst_points) if dst_points else []
        self.metadata = metadata or {}

    @property
    def is_calibrated(self) -> bool:
        """Returns True if a valid non-singular homography matrix is configured."""
        if self.H is None or self.H.shape != (3, 3):
            return False
        return not np.isclose(np.linalg.det(self.H), 0.0)

    def compute_from_points(
        self,
        src_points: List[Tuple[float, float]],
        dst_points: List[Tuple[float, float]],
        method: int = 0
    ) -> bool:
        """
        Computes the 3x3 homography matrix H from matching coordinate pairs.
        
        Args:
            src_points: Points on camera frame [(u1, v1), (u2, v2), ...]
            dst_points: Corresponding points on floor plan [(X1, Y1), (X2, Y2), ...]
            method: cv2 algorithm (0 = least-squares for 4 points, cv2.RANSAC for N > 4)
            
        Returns:
            True if computation succeeded and resulted in a non-singular matrix.
        """
        if len(src_points) < 4 or len(dst_points) < 4 or len(src_points) != len(dst_points):
            return False

        src_arr = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)
        dst_arr = np.array(dst_points, dtype=np.float32).reshape(-1, 1, 2)

        algo = cv2.RANSAC if len(src_points) > 4 else method
        H_mat, mask = cv2.findHomography(src_arr, dst_arr, algo, 5.0)

        if H_mat is not None and not np.isclose(np.linalg.det(H_mat), 0.0):
            self.H = H_mat.astype(np.float64)
            self.H_inv = np.linalg.pinv(self.H)
            self.src_points = [tuple(p) for p in src_points]
            self.dst_points = [tuple(p) for p in dst_points]
            return True
        return False

    def project_point(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """
        Projects a single camera image coordinate (u, v) onto the floor plan (X, Y).
        
        Args:
            u: Horizontal pixel on camera image (X-axis)
            v: Vertical pixel on camera image (Y-axis)
            
        Returns:
            (X_floor, Y_floor) tuple or None if point is at projective infinity.
        """
        if not self.is_calibrated:
            return None

        p = np.array([u, v, 1.0], dtype=np.float64)
        p_proj = self.H @ p
        w = p_proj[2]

        if np.isclose(w, 0.0):
            return None

        return float(p_proj[0] / w), float(p_proj[1] / w)

    def project_points(self, points: Union[List[Tuple[float, float]], np.ndarray]) -> np.ndarray:
        """
        Vectorized batch projection of multiple camera points onto the floor plan.
        
        Args:
            points: Array or list of shape (N, 2)
            
        Returns:
            Array of shape (N, 2) containing projected (X_floor, Y_floor) coordinates.
        """
        if not self.is_calibrated or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)

        pts_arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts_arr, self.H)
        return projected.reshape(-1, 2)

    def warp_camera_to_floor(
        self,
        camera_bgr: np.ndarray,
        floor_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Warps the entire camera frame onto the floor plan coordinate plane.
        
        Args:
            camera_bgr: BGR camera image of shape (H_cam, W_cam, 3)
            floor_shape: Target (H_floor, W_floor) or self.floor_shape
            
        Returns:
            Warped perspective image matching the floor plan canvas dimensions.
        """
        if not self.is_calibrated:
            raise ValueError("PlanarProjector is not calibrated with a valid H matrix.")

        h, w = floor_shape or self.floor_shape or (1080, 1920)
        return cv2.warpPerspective(camera_bgr, self.H, (w, h), flags=cv2.INTER_LINEAR)

    def get_projected_camera_quad(self) -> Optional[np.ndarray]:
        """
        Returns the 4 corner points of the camera view mapped onto the floor plan.
        Used for drawing the camera frustum / field-of-view boundary polygon.
        """
        if not self.is_calibrated or not self.camera_shape:
            return None

        h_cam, w_cam = self.camera_shape[:2]
        corners = np.array([
            [0.0, 0.0],
            [float(w_cam), 0.0],
            [float(w_cam), float(h_cam)],
            [0.0, float(h_cam)]
        ], dtype=np.float32)

        return self.project_points(corners)

    def save_calibration(self, filepath: Union[str, Path]):
        """
        Serializes calibration data (H matrix, control points, resolutions) to JSON.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "homography_matrix": self.H.tolist() if self.H is not None else None,
            "camera_shape": list(self.camera_shape) if self.camera_shape else None,
            "floor_shape": list(self.floor_shape) if self.floor_shape else None,
            "src_points": [list(p) for p in self.src_points],
            "dst_points": [list(p) for p in self.dst_points],
            "metadata": self.metadata
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_calibration(cls, filepath: Union[str, Path]) -> "PlanarProjector":
        """
        Loads calibration configuration from a JSON file.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Calibration file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        H_mat = np.array(data["homography_matrix"], dtype=np.float64) if data.get("homography_matrix") else None
        camera_shape = tuple(data["camera_shape"]) if data.get("camera_shape") else None
        floor_shape = tuple(data["floor_shape"]) if data.get("floor_shape") else None
        src_points = [tuple(p) for p in data.get("src_points", [])]
        dst_points = [tuple(p) for p in data.get("dst_points", [])]
        metadata = data.get("metadata", {})

        return cls(
            H=H_mat,
            camera_shape=camera_shape,
            floor_shape=floor_shape,
            src_points=src_points,
            dst_points=dst_points,
            metadata=metadata
        )
