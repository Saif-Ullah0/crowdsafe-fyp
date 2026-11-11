#!/usr/bin/env python3
"""
CrowdSafe Synthetic Simulation — Dataset Verification & Visual Inspection Tool
Overlays ground-truth annotations directly onto rendered frames:
- Green circles and bounding boxes on all visible heads
- Red markers on occluded agents (behind pillars/walls/people)
- Status HUD showing visible headcount, camera calibration, and per-zone Fruin densities
- Generates side-by-side comparison images for rapid visual quality audits
"""

import os
import sys
import glob
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Verify and visually inspect synthetic CCTV dataset.")
    parser.add_argument("--dataset-dir", required=True, help="Path to scenario output directory (e.g., dataset-generator/output/rush_hour)")
    parser.add_argument("--camera", default=None, help="Camera name to verify (e.g., 'cam_02' or 'CAM-02'). Defaults to first camera found.")
    parser.add_argument("--output-dir", default=None, help="Output directory for verified preview images")
    parser.add_argument("--num-frames", type=int, default=10, help="Maximum number of frames to verify (default: 10)")
    return parser.parse_args()


def find_camera_folder(dataset_dir: str, requested_cam: Optional[str] = None) -> Tuple[str, str]:
    """Locates the camera image directory and camera ID."""
    cam_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("cam_")]
    if not cam_dirs:
        raise FileNotFoundError(f"No camera directories ('cam_*') found in '{dataset_dir}'!")

    if requested_cam:
        norm_req = requested_cam.lower().replace("-", "_")
        for d in cam_dirs:
            if d == norm_req or d == f"cam_{norm_req}":
                cam_id = d.upper().replace("_", "-")
                return os.path.join(dataset_dir, d), cam_id
        raise FileNotFoundError(f"Requested camera '{requested_cam}' not found! Available: {cam_dirs}")

    # Default to first camera
    chosen_dir = sorted(cam_dirs)[0]
    cam_id = chosen_dir.upper().replace("_", "-")
    return os.path.join(dataset_dir, chosen_dir), cam_id


def overlay_ground_truth(
    image_bgr: np.ndarray,
    annotation: Dict[str, Any]
) -> np.ndarray:
    """Draws green dots on visible heads, red markers on occluded ones, and HUD overlay."""
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    visible_heads = annotation.get("head_coordinates", [])
    occluded_heads = annotation.get("occluded_heads", [])
    camera_id = annotation.get("camera_id", "UNKNOWN")
    frame_id = annotation.get("frame_id", 0)
    total_vis = annotation.get("total_visible_count", len(visible_heads))
    zones = annotation.get("zone_densities", {})

    # 1. Draw Occluded Heads (Red with 'X')
    for head in occluded_heads:
        px = int(round(head["x_pixel"]))
        py = int(round(head["y_pixel"]))
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(annotated, (px, py), 5, (0, 0, 220), 1, cv2.LINE_AA)
            cv2.line(annotated, (px - 3, py - 3), (px + 3, py + 3), (0, 0, 255), 1)
            cv2.line(annotated, (px - 3, py + 3), (px + 3, py - 3), (0, 0, 255), 1)

    # 2. Draw Visible Heads (Bright Green dot + perspective BBox)
    for head in visible_heads:
        px = int(round(head["x_pixel"]))
        py = int(round(head["y_pixel"]))
        aid = head.get("agent_id", "")
        bbox = head.get("bbox_xywh", [px - 5, py - 5, 10, 10])

        if 0 <= px < w and 0 <= py < h:
            # Solid green dot directly at head center
            cv2.circle(annotated, (px, py), 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(annotated, (px, py), 6, (0, 180, 0), 1, cv2.LINE_AA)

            # Draw bounding box
            bx, by, bw, bh = [int(v) for v in bbox]
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 120), 1, cv2.LINE_AA)

            # Draw tiny agent label
            cv2.putText(
                annotated,
                f"#{aid}",
                (bx, max(12, by - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

    # 3. Draw Semi-transparent HUD Header Bar
    hud_h = 42
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 24), -1)
    cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)

    # Status texts
    header_title = f"{camera_id} | FRAME {frame_id:04d} | GROUND TRUTH: {total_vis} HEADS"
    cv2.putText(annotated, header_title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Zone density status ticker
    zone_texts = []
    for zid, zinfo in zones.items():
        if zinfo.get("visible_count", 0) > 0:
            name = zinfo.get("name", zid)[:14]
            dens = zinfo.get("density_per_m2", 0.0)
            risk = zinfo.get("risk_level", "SAFE")
            zone_texts.append(f"{name}: {dens:.2f}/m2 [{risk}]")
    
    zone_line = " | ".join(zone_texts) if zone_texts else "All zones sparse (density < 0.1/m2)"
    cv2.putText(annotated, zone_line, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 230), 1, cv2.LINE_AA)

    # 4. Legend at bottom
    cv2.putText(annotated, "[GREEN: Visible Head Center]  [RED: Occluded by obstacle/agent]", (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    return annotated


def create_side_by_side(raw_img: np.ndarray, annotated_img: np.ndarray) -> np.ndarray:
    """Concatenates raw render and verified overlay side-by-side with divider."""
    h, w = raw_img.shape[:2]
    divider = np.zeros((h, 4, 3), dtype=np.uint8)
    divider[:, :] = (80, 80, 80)
    return np.hstack([raw_img, divider, annotated_img])


def main():
    args = parse_args()
    cam_dir, cam_id = find_camera_folder(args.dataset_dir, args.camera)
    anno_dir = os.path.join(args.dataset_dir, "annotations")

    if not os.path.exists(anno_dir):
        raise FileNotFoundError(f"Annotations directory not found at: {anno_dir}")

    output_dir = args.output_dir or os.path.join(args.dataset_dir, f"verified_{cam_id.lower().replace('-', '_')}")
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(cam_dir, "frame_*.png")))
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.png images found in '{cam_dir}'!")

    sample_frames = frame_paths[:args.num_frames]

    print("================================================================")
    print("      CROWDSAFE DATASET VERIFICATION & INSPECTION SUITE         ")
    print("================================================================")
    print(f"Dataset Directory: {args.dataset_dir}")
    print(f"Camera:            {cam_id}")
    print(f"Total Frames Found: {len(frame_paths)} (Verifying first {len(sample_frames)})")
    print(f"Output Preview:    {output_dir}")
    print("----------------------------------------------------------------")

    total_visible_verified = 0
    total_occluded_verified = 0
    all_within_bounds = True

    for f_idx, img_path in enumerate(sample_frames):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        frame_num = basename.split("_")[-1]

        # Match annotation JSON
        cam_prefix = cam_id.lower().replace("-", "_")
        anno_file = os.path.join(anno_dir, f"{cam_prefix}_{basename}.json")
        if not os.path.exists(anno_file):
            print(f"[Warning] Annotation file missing: {anno_file}")
            continue

        with open(anno_file, "r") as af:
            anno_data = json.load(af)

        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            continue

        annotated_bgr = overlay_ground_truth(raw_bgr, anno_data)
        side_by_side = create_side_by_side(raw_bgr, annotated_bgr)

        # Save verification outputs
        out_single = os.path.join(output_dir, f"verified_{basename}.png")
        out_comparison = os.path.join(output_dir, f"comparison_{basename}.png")

        cv2.imwrite(out_single, annotated_bgr)
        cv2.imwrite(out_comparison, side_by_side)

        vis_count = anno_data.get("total_visible_count", 0)
        occ_count = anno_data.get("total_occluded_count", 0)
        total_visible_verified += vis_count
        total_occluded_verified += occ_count

        # Check bounds
        h, w = raw_bgr.shape[:2]
        for h_entry in anno_data.get("head_coordinates", []):
            x = h_entry["x_pixel"]
            y = h_entry["y_pixel"]
            if not (0 <= x <= w and 0 <= y <= h):
                all_within_bounds = False

        print(f"Frame {frame_num} | Visible: {vis_count:2d} heads | Occluded: {occ_count:2d} heads | Saved: {out_comparison}")

    print("----------------------------------------------------------------")
    print("VERIFICATION AUDIT RESULTS:")
    print(f"  * Frames Audited:         {len(sample_frames)}")
    print(f"  * Total Visible Heads:    {total_visible_verified}")
    print(f"  * Total Occluded Heads:   {total_occluded_verified}")
    print(f"  * Coordinate Bounds Check: {'PASSED (100% within frame bounds)' if all_within_bounds else 'FAILED'}")
    print(f"  * Inspection Directory:   {output_dir}")
    print("================================================================")


if __name__ == "__main__":
    main()
