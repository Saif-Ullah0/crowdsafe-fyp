"""
DVR & CCTV Ingestion Streamer for CrowdSafe MVP
Simulates commercial surveillance DVR hardware streaming at configurable frame rates (default: 1 FPS).
Supports:
1. Image sequence directories (e.g. Mall dataset frames)
2. Video files (.mp4, .avi, etc.)
3. Live webcam / USB camera devices
4. Dynamic synthetic CCTV simulator with moving pedestrians and realistic crowd surges
"""
import os
import glob
import time
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Tuple, Optional, List
import cv2
import numpy as np

class DVRStreamer:
    """Ingests frames from video, camera, image directory, or synthetic generator at target FPS."""

    def __init__(
        self,
        source: str = "sample",
        target_fps: float = 1.0,
        loop: bool = True,
        max_frames: Optional[int] = None,
        frame_size: Tuple[int, int] = (480, 640)
    ):
        self.source = str(source).strip()
        self.target_fps = max(float(target_fps), 0.01)
        self.interval = 1.0 / self.target_fps
        self.loop = loop
        self.max_frames = max_frames
        self.frame_size = frame_size  # (H, W)
        
        self.frame_idx = 0
        self.last_frame_time = 0.0
        self._mode = "synthetic"
        
        # Source handlers
        self.image_files: List[str] = []
        self.image_ptr = 0
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Synthetic simulator state
        self._sim_pedestrians = []
        
        self._initialize_source()

    def _initialize_source(self):
        """Detects source type and initializes appropriate reader."""
        if self.source.lower() == "sample":
            self._mode = "synthetic"
            self._init_synthetic_simulation()
            print(f"[DVRStreamer] Initialized Synthetic CCTV Simulator ({self.target_fps} FPS)")
            return

        # Check if source is a numeric camera index (e.g. 0, 1)
        if self.source.isdigit():
            self._mode = "webcam"
            cam_id = int(self.source)
            self.cap = cv2.VideoCapture(cam_id)
            if not self.cap.isOpened():
                print(f"[DVRStreamer] Warning: Failed to open webcam #{cam_id}. Falling back to synthetic.")
                self._mode = "synthetic"
                self._init_synthetic_simulation()
            else:
                print(f"[DVRStreamer] Initialized Live Webcam Stream (Device #{cam_id})")
            return

        src_path = Path(self.source)
        if src_path.is_dir():
            # Directory of images
            extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"]
            files = []
            for ext in extensions:
                files.extend(glob.glob(str(src_path / ext)))
            files.sort()
            if files:
                self._mode = "image_dir"
                self.image_files = files
                self.image_ptr = 0
                print(f"[DVRStreamer] Found {len(files)} frames in image directory: {src_path}")
            else:
                print(f"[DVRStreamer] Warning: No images found in {src_path}. Falling back to synthetic.")
                self._mode = "synthetic"
                self._init_synthetic_simulation()
            return

        if src_path.is_file():
            # Video file
            self._mode = "video_file"
            self.cap = cv2.VideoCapture(str(src_path))
            if not self.cap.isOpened():
                print(f"[DVRStreamer] Warning: Failed to open video file {src_path}. Falling back to synthetic.")
                self._mode = "synthetic"
                self._init_synthetic_simulation()
            else:
                print(f"[DVRStreamer] Initialized Video File Stream: {src_path.name}")
            return

        # Default fallback
        print(f"[DVRStreamer] Unknown source '{self.source}'. Running in synthetic simulation mode.")
        self._mode = "synthetic"
        self._init_synthetic_simulation()

    def _init_synthetic_simulation(self):
        """Initializes simulated pedestrian clusters moving across venue zones."""
        np.random.seed(42)
        h, w = self.frame_size
        num_pedestrians = 35
        self._sim_pedestrians = []
        for _ in range(num_pedestrians):
            self._sim_pedestrians.append({
                "x": float(np.random.randint(50, w - 50)),
                "y": float(np.random.randint(50, h - 50)),
                "vx": float(np.random.uniform(-4.0, 4.0)),
                "vy": float(np.random.uniform(-3.0, 3.0)),
                "radius": int(np.random.randint(6, 11)),
                "color": (
                    int(np.random.randint(180, 240)),
                    int(np.random.randint(180, 240)),
                    int(np.random.randint(180, 240))
                )
            })

    def read(self, pace: bool = True) -> Tuple[bool, Optional[np.ndarray], int]:
        """
        Reads the next frame in the sequence.
        
        Args:
            pace: If True, sleeps if necessary to maintain target FPS.
            
        Returns:
            Tuple: (success, frame_bgr, frame_idx)
        """
        if self.max_frames and self.frame_idx >= self.max_frames:
            return False, None, self.frame_idx

        # Enforce rate limiting to match target FPS
        if pace and self.last_frame_time > 0:
            elapsed = time.time() - self.last_frame_time
            sleep_duration = self.interval - elapsed
            if sleep_duration > 0:
                time.sleep(sleep_duration)

        frame = None
        success = False

        if self._mode == "image_dir":
            if self.image_ptr >= len(self.image_files):
                if self.loop:
                    self.image_ptr = 0
                else:
                    return False, None, self.frame_idx

            img_path = self.image_files[self.image_ptr]
            frame = cv2.imread(img_path)
            self.image_ptr += 1
            success = frame is not None

        elif self._mode in ["video_file", "webcam"]:
            ret, captured = self.cap.read()
            if not ret or captured is None:
                if self.loop and self._mode == "video_file":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, captured = self.cap.read()
                else:
                    return False, None, self.frame_idx
            frame = captured
            success = ret

        elif self._mode == "synthetic":
            frame = self._generate_synthetic_frame()
            success = True

        if success and frame is not None:
            self.frame_idx += 1
            self.last_frame_time = time.time()
            return True, frame, self.frame_idx

        return False, None, self.frame_idx

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generates realistic synthetic surveillance frame with moving crowds."""
        h, w = self.frame_size
        # Dark venue background with floor tile grid
        frame = np.full((h, w, 3), 42, dtype=np.uint8)
        
        # Draw subtle tile grid lines
        for y_line in range(0, h, 60):
            cv2.line(frame, (0, y_line), (w, y_line), (52, 52, 52), 1)
        for x_line in range(0, w, 60):
            cv2.line(frame, (x_line, 0), (x_line, h), (52, 52, 52), 1)

        # Dynamic crowd surge simulation cycle:
        # Every 12 frames, inject a surge in the Central Corridor to test alert triggers
        surge_intensity = max(0.0, math.sin(self.frame_idx * 0.25))

        # Update and draw pedestrians
        for ped in self._sim_pedestrians:
            ped["x"] += ped["vx"]
            ped["y"] += ped["vy"]
            
            # Bounce off boundaries
            if ped["x"] < 30 or ped["x"] > w - 30:
                ped["vx"] *= -1
            if ped["y"] < 30 or ped["y"] > h - 30:
                ped["vy"] *= -1

            # Render head dot & subtle body shadow
            cx, cy = int(ped["x"]), int(ped["y"])
            cv2.circle(frame, (cx, cy + 3), ped["radius"] + 2, (20, 20, 20), -1)
            cv2.circle(frame, (cx, cy), ped["radius"], ped["color"], -1)

        # Draw surge cluster when surge phase is active
        if surge_intensity > 0.4:
            surge_center_x, surge_center_y = int(w * 0.65), int(h * 0.5)
            num_surge_pedestrians = int(surge_intensity * 85)
            for s_idx in range(num_surge_pedestrians):
                offset_x = int(np.random.normal(0, 30))
                offset_y = int(np.random.normal(0, 30))
                scx = np.clip(surge_center_x + offset_x, 10, w - 10)
                scy = np.clip(surge_center_y + offset_y, 10, h - 10)
                cv2.circle(frame, (scx, scy), 8, (220, 225, 230), -1)
                cv2.circle(frame, (scx, scy), 3, (40, 40, 40), -1)

        # CCTV timestamp and camera watermark
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        watermark = f"CAM-01 [NORTH CONCOURSE]  |  {now_str}  |  DVR: 1.0 FPS"
        cv2.putText(frame, watermark, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        return frame

    def stream(self, pace: bool = True) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator yielding (frame_index, frame_bgr) continuously."""
        while True:
            success, frame, idx = self.read(pace=pace)
            if not success or frame is None:
                break
            yield idx, frame

    def close(self):
        """Releases underlying capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

