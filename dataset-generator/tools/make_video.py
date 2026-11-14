#!/usr/bin/env python3
"""
CrowdSafe - Surveillance Video Factory (make_video.py)
Encodes synthetic CCTV video sequences into standard MP4 format at 10 FPS:
- Authentic CCTV On-Screen Display (OSD): Camera Name, Timestamp, Blinking [REC],
  Live Density, Fruin Safety Status, Codec Watermark.
- Ground-Truth Debug Video Mode (Bounding boxes, Visible vs Occluded head markers).
- Multi-Camera Quad Split-Screen (Synchronized 2x2 matrix surveillance view).
- Direct ingestion into CrowdSafe MVP surveillance monitor.
"""

import os
import sys
import glob
import json
import math
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np


def draw_cctv_osd(
    frame: np.ndarray,
    cam_id: str,
    cam_name: str,
    frame_idx: int,
    fps: float,
    start_time: datetime,
    head_count: Optional[int] = None,
    density: Optional[float] = None,
    fruin_status: Optional[str] = "SAFE"
) -> np.ndarray:
    """Renders authentic surveillance CCTV on-screen display (OSD) overlays."""
    h, w = frame.shape[:2]
    out = frame.copy()

    # Time calculations
    curr_time = start_time + timedelta(seconds=(frame_idx / max(1.0, fps)))
    timestamp_str = curr_time.strftime("%Y-%m-%d  %H:%M:%S") + f".{(curr_time.microsecond // 100000):01d}"

    # Blinking REC dot: 1 Hz oscillation (0.5s ON, 0.5s OFF)
    rec_on = (int(frame_idx / fps * 2.0) % 2 == 0)

    # 1. Top Ribbon Semi-Transparent Shadow Bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (15, 15, 15), -1)
    cv2.rectangle(overlay, (0, h - 30), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # Font definitions
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = cv2.FONT_HERSHEY_DUPLEX

    # 2. Top-Left: Camera ID & Channel
    cam_header = f"[{cam_id.upper()}] {cam_name.upper()}"
    cv2.putText(out, cam_header, (14, 23), font_bold, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Top-Right: Blinking [● REC] & Live Clock
    time_x = w - 245
    cv2.putText(out, timestamp_str, (time_x, 23), font, 0.50, (235, 235, 235), 1, cv2.LINE_AA)

    # REC Indicator
    rec_x = time_x - 70
    if rec_on:
        cv2.circle(out, (rec_x, 18), 5, (40, 40, 240), -1)  # Red solid dot
        cv2.putText(out, "REC", (rec_x + 9, 23), font_bold, 0.45, (40, 40, 240), 1, cv2.LINE_AA)
    else:
        cv2.circle(out, (rec_x, 18), 5, (180, 180, 180), 1)  # Hollow dot
        cv2.putText(out, "REC", (rec_x + 9, 23), font_bold, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # 4. Bottom-Left: Live Crowd Metrics
    if head_count is not None:
        fruin_color = (80, 220, 80)  # Green
        if fruin_status == "CAUTION":
            fruin_color = (50, 200, 240)  # Yellow
        elif fruin_status == "WARNING":
            fruin_color = (40, 120, 255)  # Orange
        elif fruin_status == "CRITICAL_DANGER":
            fruin_color = (40, 40, 255)  # Red

        metric_str = f"COUNT: {head_count:02d}  |  DENS: {density:.2f} p/m2  |  STATUS: {fruin_status}"
        cv2.putText(out, metric_str, (14, h - 10), font_bold, 0.45, fruin_color, 1, cv2.LINE_AA)

    # 5. Bottom-Right: Codec Watermark
    codec_str = f"CCTV-IP  {w}x{h} @ {fps:.1f} FPS"
    cv2.putText(out, codec_str, (w - 195, h - 10), font, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    return out


def draw_tracking_overlay(frame: np.ndarray, annotation_data: Dict[str, Any]) -> np.ndarray:
    """Draws ground truth bounding boxes and visible/occluded head dots on frame."""
    out = frame.copy()

    # 1. Occluded Heads (Red hollow circles with cross)
    for occ in annotation_data.get("occluded_heads", []):
        px = int(round(occ.get("pixel_x", occ.get("x_pixel", 0))))
        py = int(round(occ.get("pixel_y", occ.get("y_pixel", 0))))
        cv2.circle(out, (px, py), 6, (0, 0, 230), 1, cv2.LINE_AA)
        cv2.line(out, (px - 4, py - 4), (px + 4, py + 4), (0, 0, 230), 1, cv2.LINE_AA)
        cv2.line(out, (px - 4, py + 4), (px + 4, py - 4), (0, 0, 230), 1, cv2.LINE_AA)

    # 2. Visible Heads (Green solid dots and bounding boxes)
    vis_list = annotation_data.get("head_coordinates", annotation_data.get("visible_heads", []))
    for vis in vis_list:
        px = int(round(vis.get("pixel_x", vis.get("x_pixel", 0))))
        py = int(round(vis.get("pixel_y", vis.get("y_pixel", 0))))
        aid = vis.get("agent_id", 0)
        bbox = vis.get("bbox", vis.get("bbox_xywh", None))

        # Draw green tracking center dot
        cv2.circle(out, (px, py), 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), 6, (0, 180, 0), 1, cv2.LINE_AA)

        # Draw bounding box if present
        if bbox and len(bbox) == 4:
            bx, by, bw, bh = [int(v) for v in bbox]
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
            cv2.putText(out, f"ID:{aid}", (bx, max(12, by - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    return out


def load_annotation_for_camera(anno_dir: str, cam_id: str, frame_idx: int) -> Dict[str, Any]:
    """Helper to locate and parse annotation record for a given camera and frame."""
    c_norm = cam_id.lower().replace("-", "_")
    candidates = [
        os.path.join(anno_dir, f"{c_norm}_frame_{frame_idx:04d}.json"),
        os.path.join(anno_dir, f"{cam_id.lower()}_frame_{frame_idx:04d}.json"),
        os.path.join(anno_dir, f"annotations_{frame_idx:04d}.json"),
        os.path.join(anno_dir, f"frame_{frame_idx:04d}.json"),
    ]
    a_file = next((f for f in candidates if os.path.exists(f)), None)
    if not a_file:
        return {}

    try:
        with open(a_file, "r") as f:
            data = json.load(f)
        if "cameras" in data:
            return data["cameras"].get(cam_id.upper(), {})
        return data
    except Exception:
        return {}


def encode_single_camera_video(
    frames_dir: str,
    anno_dir: Optional[str],
    output_mp4: str,
    cam_id: str,
    cam_name: str,
    fps: float = 10.0,
    with_osd: bool = True,
    with_tracking: bool = False
) -> str:
    """Encodes a single camera frame sequence into an MP4 video."""
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    # Inspect first frame for resolution
    first_img = cv2.imread(frame_files[0])
    h, w = first_img.shape[:2]

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
    writer = cv2.VideoWriter(output_mp4, fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_mp4}")

    start_sim_time = datetime(2026, 11, 14, 14, 0, 0)

    for idx, fpath in enumerate(frame_files):
        img = cv2.imread(fpath)
        if img is None:
            continue

        head_count = None
        density = 0.0
        fruin_status = "SAFE"

        # Check corresponding annotation
        if anno_dir:
            cam_data = load_annotation_for_camera(anno_dir, cam_id, idx)
            if cam_data:
                head_count = cam_data.get("total_visible_count", len(cam_data.get("head_coordinates", [])))

                # Highest density among covered zones
                zone_dict = cam_data.get("zone_densities", cam_data.get("zone_statistics", {}))
                for zname, zinfo in zone_dict.items():
                    zdens = zinfo.get("density_per_m2", 0.0)
                    if zdens > density:
                        density = zdens
                        fruin_status = zinfo.get("risk_level", "SAFE")

                if with_tracking:
                    img = draw_tracking_overlay(img, cam_data)

        if with_osd:
            img = draw_cctv_osd(
                img,
                cam_id=cam_id,
                cam_name=cam_name,
                frame_idx=idx,
                fps=fps,
                start_time=start_sim_time,
                head_count=head_count,
                density=density,
                fruin_status=fruin_status
            )

        writer.write(img)

    writer.release()
    print(f"[✓] Encoded {len(frame_files)} frames to video: {output_mp4}")
    return output_mp4


def encode_quad_split_screen(
    scenario_dir: str,
    output_mp4: str,
    camera_map: Dict[str, str],
    fps: float = 10.0,
    with_osd: bool = True
) -> str:
    """Combines 4 cameras into a synchronized 2x2 multi-camera split screen matrix."""
    cam_keys = list(camera_map.keys())[:4]
    cam_dirs = [os.path.join(scenario_dir, k.lower().replace("-", "_")) for k in cam_keys]

    # Find common frame count
    frame_lists = [sorted(glob.glob(os.path.join(cdir, "frame_*.png"))) for cdir in cam_dirs]
    min_len = min(len(fl) for fl in frame_lists)
    if min_len == 0:
        raise RuntimeError("No frames found across camera directories for quad split screen!")

    sample_img = cv2.imread(frame_lists[0][0])
    sh, sw = sample_img.shape[:2]
    quad_w = sw * 2
    quad_h = sh * 2

    anno_dir = os.path.join(scenario_dir, "annotations")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
    writer = cv2.VideoWriter(output_mp4, fourcc, fps, (quad_w, quad_h))

    start_sim_time = datetime(2026, 11, 14, 14, 0, 0)

    # 2x2 grid positions: (x, y, w, h)
    grid_positions = [
        (0, 0, sw, sh),       # Top-Left: CAM-01
        (sw, 0, sw, sh),      # Top-Right: CAM-02
        (0, sh, sw, sh),      # Bottom-Left: CAM-04
        (sw, sh, sw, sh)      # Bottom-Right: CAM-03
    ]

    for idx in range(min_len):
        quad_canvas = np.zeros((quad_h, quad_w, 3), dtype=np.uint8)

        for c_idx, (cid, cname) in enumerate(camera_map.items()):
            if c_idx >= 4:
                break
            fpath = frame_lists[c_idx][idx]
            img = cv2.imread(fpath)
            if img is None:
                img = np.zeros((sh, sw, 3), dtype=np.uint8)

            c_info = load_annotation_for_camera(anno_dir, cid, idx)
            head_count = c_info.get("total_visible_count", len(c_info.get("head_coordinates", [])))
            density = 0.0
            fruin = "SAFE"
            zone_dict = c_info.get("zone_densities", c_info.get("zone_statistics", {}))
            for zname, zinfo in zone_dict.items():
                if zinfo.get("density_per_m2", 0) > density:
                    density = zinfo.get("density_per_m2", 0)
                    fruin = zinfo.get("risk_level", "SAFE")

            if with_osd:
                img = draw_cctv_osd(
                    img,
                    cam_id=cid,
                    cam_name=cname,
                    frame_idx=idx,
                    fps=fps,
                    start_time=start_sim_time,
                    head_count=head_count,
                    density=density,
                    fruin_status=fruin
                )

            gx, gy, gw, gh = grid_positions[c_idx]
            quad_canvas[gy:gy+gh, gx:gx+gw] = img

        # Draw grid dividing borders
        cv2.line(quad_canvas, (sw, 0), (sw, quad_h), (20, 20, 20), 3)
        cv2.line(quad_canvas, (0, sh), (quad_w, sh), (20, 20, 20), 3)

        writer.write(quad_canvas)

    writer.release()
    print(f"[✓] Encoded {min_len} frames to Quad Split-Screen video: {output_mp4}")
    return output_mp4


def main():
    parser = argparse.ArgumentParser(description="Surveillance Video Factory (make_video.py)")
    parser.add_argument("--scenario-dir", required=True, help="Directory containing rendered camera frame subfolders")
    parser.add_argument("--camera", default="all", help="Camera ID (e.g. cam_01, cam_02), 'all', or 'quad'")
    parser.add_argument("--fps", type=float, default=10.0, help="Video framerate (default 10.0 FPS)")
    parser.add_argument("--tracking", action="store_true", help="Export ground-truth debug tracking overlay video")
    parser.add_argument("--no-osd", action="store_true", help="Disable CCTV on-screen display overlay")
    parser.add_argument("--output-dir", default=None, help="Output directory for MP4 videos")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    scenario_dir = os.path.abspath(args.scenario_dir)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else scenario_dir
    os.makedirs(out_dir, exist_ok=True)

    # Camera metadata mapping
    cam_meta = {
        "CAM-01": "West Corridor Surveillance",
        "CAM-02": "Food Court & Central Axis",
        "CAM-03": "East Corridor Surveillance",
        "CAM-04": "Main Entrance & Atrium"
    }

    anno_dir = os.path.join(scenario_dir, "annotations")
    with_osd = not args.no_osd

    if args.camera.lower() == "quad":
        out_quad = os.path.join(out_dir, "mall_quad_surveillance.mp4")
        encode_quad_split_screen(scenario_dir, out_quad, cam_meta, fps=args.fps, with_osd=with_osd)
        return

    # Process requested cameras
    if args.camera.lower() == "all":
        targets = list(cam_meta.keys())
    else:
        targets = [
            camera.strip().upper().replace("_", "-")
            for camera in args.camera.split(",")
            if camera.strip()
        ]

    for cid in targets:
        c_folder = cid.lower().replace("-", "_")
        c_path = os.path.join(scenario_dir, c_folder)
        if not os.path.exists(c_path):
            print(f"[!] Warning: Camera folder {c_path} does not exist, skipping.")
            continue

        cname = cam_meta.get(cid, cid)

        # Standard surveillance video
        surv_mp4 = os.path.join(out_dir, f"{c_folder}_surveillance.mp4")
        encode_single_camera_video(
            c_path, anno_dir, surv_mp4, cid, cname, fps=args.fps, with_osd=with_osd, with_tracking=False
        )

        # Ground-truth tracking debug video if requested
        if args.tracking:
            track_mp4 = os.path.join(out_dir, f"{c_folder}_tracking.mp4")
            encode_single_camera_video(
                c_path, anno_dir, track_mp4, cid, cname, fps=args.fps, with_osd=with_osd, with_tracking=True
            )

    # If all cameras were generated, also generate the Quad split screen view!
    if args.camera.lower() == "all":
        out_quad = os.path.join(out_dir, "mall_quad_surveillance.mp4")
        try:
            encode_quad_split_screen(scenario_dir, out_quad, cam_meta, fps=args.fps, with_osd=with_osd)
        except Exception as e:
            print(f"[!] Quad split screen note: {e}")


if __name__ == "__main__":
    main()
