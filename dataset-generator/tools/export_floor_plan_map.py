#!/usr/bin/env python3
"""
CrowdSafe - Architectural 2D Floor Plan & CCTV Coverage Map Generator
Reads mall_floor_plan.json and exports a high-resolution CAD-style architectural
drawing with wall layouts, zones, navigation pathways, obstacles, camera mounts,
and semi-transparent Field-of-View (FOV) coverage cones.
"""

import os
import sys
import json
import math
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path


def extract_wall_segments(data: Dict[str, Any]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Extracts all opaque wall line segments from walls and room zone polygons."""
    segments = []
    # 1. Explicit wall segments
    for w in data.get("walls", []):
        s = w.get("start")
        e = w.get("end")
        if s and e:
            segments.append(((float(s[0]), float(s[1])), (float(e[0]), float(e[1]))))
    # 2. Room zone boundary walls
    for zid, zinfo in data.get("zones", {}).items():
        ztype = zinfo.get("type", "")
        if "room" in ztype or ztype in ["retail", "utility", "office"]:
            poly = zinfo.get("polygon", [])
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i + 1) % len(poly)]
                segments.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
    return segments


def compute_raycast_visibility_polygon(
    cam: Dict[str, Any],
    wall_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    n_samples: int = 160
) -> Tuple[np.ndarray, float, float, float]:
    """
    Computes authentic real-life 3D CCTV line-of-sight visibility polygon
    clipped by solid walls (zero wall penetration) with nadir blind spot calculation.
    """
    pos = cam["position"]
    look = cam["look_at"]
    cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
    lx, ly, lz = float(look[0]), float(look[1]), float(look[2])

    f_mm = cam.get("focal_length_mm", 5.0)
    sw_mm = cam.get("sensor_width_mm", 6.17)
    sh_mm = cam.get("sensor_height_mm", 4.55)
    max_range = cam.get("max_range_m", 36.0)

    dx = lx - cx
    dy = ly - cy
    horiz_dist = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)
    heading_deg = math.degrees(heading)

    # 3D downward tilt angle below horizontal
    tilt_rad = math.atan2(max(0.0, cz - lz), max(0.1, horiz_dist))
    tilt_deg = math.degrees(tilt_rad)

    # Sensor field of view
    hfov_rad = 2.0 * math.atan(sw_mm / (2.0 * f_mm))
    vfov_rad = 2.0 * math.atan(sh_mm / (2.0 * f_mm))
    half_hfov = hfov_rad / 2.0

    # Near nadir blind spot distance
    bottom_angle = tilt_rad + vfov_rad / 2.0
    if math.tan(bottom_angle) > 0.05:
        d_min = cz / math.tan(bottom_angle)
    else:
        d_min = 1.0
    d_min = max(0.5, min(d_min, 6.0))

    theta_start = heading - half_hfov
    theta_end = heading + half_hfov

    # Angular ray sampling
    angles = list(np.linspace(theta_start, theta_end, n_samples))
    for (x1, y1), (x2, y2) in wall_segments:
        for vx, vy in [(x1, y1), (x2, y2)]:
            vang = math.atan2(vy - cy, vx - cx)
            diff = (vang - heading + math.pi) % (2.0 * math.pi) - math.pi
            if -half_hfov <= diff <= half_hfov:
                angles.append(heading + diff - 1e-4)
                angles.append(heading + diff)
                angles.append(heading + diff + 1e-4)

    angles = sorted(list(set(angles)))

    far_pts = []
    near_pts = []

    for ang in angles:
        rdx = math.cos(ang)
        rdy = math.sin(ang)

        # Intersect with all wall segments
        closest_t = max_range
        for (wx1, wy1), (wx2, wy2) in wall_segments:
            denom = rdx * (wy2 - wy1) - rdy * (wx2 - wx1)
            if abs(denom) < 1e-8:
                continue
            t = ((wx1 - cx) * (wy2 - wy1) - (wy1 - cy) * (wx2 - wx1)) / denom
            u = ((wx1 - cx) * rdy - (wy1 - cy) * rdx) / denom
            if t > 0.05 and 0.0 <= u <= 1.0:
                if t < closest_t:
                    closest_t = t

        t_far = closest_t
        t_near = min(closest_t, d_min)
        far_pts.append((cx + rdx * t_far, cy + rdy * t_far))
        near_pts.append((cx + rdx * t_near, cy + rdy * t_near))

    poly_pts = far_pts + list(reversed(near_pts))
    return np.array(poly_pts), d_min, heading_deg, math.degrees(hfov_rad)


def calculate_camera_fov(cam: Dict[str, Any], default_range: float = 14.0) -> Dict[str, Any]:
    """Calculate the 2D horizontal FOV angle, direction, and cone coordinates."""
    pos = cam["position"]
    look = cam["look_at"]

    # Sensor & Lens parameters
    f_mm = cam.get("focal_length_mm", 6.0)
    sw_mm = cam.get("sensor_width_mm", 6.17)

    # Horizontal Field of View (degrees)
    fov_h_rad = 2.0 * math.atan(sw_mm / (2.0 * f_mm))
    fov_h_deg = math.degrees(fov_h_rad)

    # 2D orientation angle on XY ground plane
    dx = look[0] - pos[0]
    dy = look[1] - pos[1]
    heading_deg = math.degrees(math.atan2(dy, dx))

    theta1 = heading_deg - (fov_h_deg / 2.0)
    theta2 = heading_deg + (fov_h_deg / 2.0)

    return {
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "heading_deg": heading_deg,
        "fov_deg": fov_h_deg,
        "theta1": theta1,
        "theta2": theta2,
        "range_m": default_range
    }


def draw_cad_floor_plan(
    plan_path: str,
    output_png: str,
    output_svg: str = None,
    dpi: int = 300,
    show_nav: bool = True
) -> str:
    """Renders the venue floor plan with CCTV FOV cones in clean CAD aesthetic."""
    with open(plan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "dimensions" in data:
        W = float(data["dimensions"].get("width_x", 50.0))
        H = float(data["dimensions"].get("depth_y", 30.0))
    elif "global_dimensions" in data:
        W = float(data["global_dimensions"].get("width_m", 36.0))
        H = float(data["global_dimensions"].get("depth_m", 36.0))
    else:
        W, H = 36.0, 36.0

    # Create Figure with 16:10 architectural proportions
    fig_w = 18
    fig_h = (H / W) * fig_w + 3.0  # extra margin for title block and legend
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # Set background to crisp architectural white/light cream
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FCFDFD')

    # Grid lines (1m subtle grid, 5m structural grid)
    ax.set_xlim(-2.0, W + 2.0)
    ax.set_ylim(-3.5, H + 3.5)
    ax.set_aspect('equal')

    # Subtle 5m major grid
    ax.set_xticks(np.arange(0, W + 1, 5.0))
    ax.set_yticks(np.arange(0, H + 1, 5.0))
    ax.grid(True, which='both', color='#E2E8F0', linestyle='-', linewidth=0.6, alpha=0.8)

    # 1. Draw Zones (soft pastel fill)
    zone_palette = [
        ("#EBF5FB", "#2980B9"),  # Soft Blue
        ("#E8F8F5", "#16A085"),  # Soft Mint
        ("#FEF9E7", "#D4AC0D"),  # Soft Amber
        ("#F4ECF7", "#8E44AD"),  # Soft Purple
        ("#FDEDEC", "#C0392B"),  # Soft Coral
        ("#EAECEE", "#7F8C8D"),  # Soft Gray
        ("#EAF2F8", "#3498DB"),  # Sky
        ("#EAFAF1", "#27AE60"),  # Light Green
    ]

    zones = data.get("zones", {})
    for idx, (zid, zone_info) in enumerate(zones.items()):
        poly_pts = np.array(zone_info["polygon"])
        fill_col, edge_col = zone_palette[idx % len(zone_palette)]

        zone_patch = patches.Polygon(
            poly_pts,
            closed=True,
            facecolor=fill_col,
            edgecolor=edge_col,
            linewidth=1.2,
            linestyle='--',
            alpha=0.65,
            zorder=2
        )
        ax.add_patch(zone_patch)

        # Centroid for label
        cx = np.mean(poly_pts[:, 0])
        cy = np.mean(poly_pts[:, 1])
        area = zone_info.get("area_m2", 0)
        zname = zone_info.get("name", zid)

        ax.text(
            cx, cy,
            f"{zname}\n({area:.1f} m²)",
            color="#2C3E50",
            fontsize=8.5,
            fontweight='semibold',
            ha='center',
            va='center',
            zorder=3,
            bbox=dict(boxstyle="round,pad=0.25", facecolor='#FFFFFF', edgecolor='#BDC3C7', alpha=0.85, linewidth=0.7)
        )

    # 2. Draw Navigation Graph (waypoint nodes and connectors)
    if show_nav and "navigation_graph" in data:
        nav = data["navigation_graph"]
        nodes = nav.get("nodes", {})
        edges = nav.get("edges", [])

        # Edges
        for edge in edges:
            u_pos = nodes[edge["from"]]["pos"]
            v_pos = nodes[edge["to"]]["pos"]
            ax.plot(
                [u_pos[0], v_pos[0]], [u_pos[1], v_pos[1]],
                color='#95A5A6',
                linestyle=':',
                linewidth=1.0,
                alpha=0.6,
                zorder=4
            )

        # Nodes
        for nid, node_info in nodes.items():
            pos = node_info["pos"]
            ntype = node_info.get("type", "waypoint")
            if "exit" in ntype:
                node_col = '#E74C3C'
                m_size = 40
            elif "entrance" in ntype:
                node_col = '#27AE60'
                m_size = 40
            else:
                node_col = '#7F8C8D'
                m_size = 20

            ax.scatter(pos[0], pos[1], s=m_size, c=node_col, edgecolors='#2C3E50', linewidths=0.8, zorder=5)

    # 3. Draw Obstacles (Pillars, tables, benches, kiosks)
    obstacles = data.get("obstacles", [])
    for obs in obstacles:
        pos = obs["position"]
        otype = obs.get("type", "box")
        oname = obs.get("name", "")

        if otype == "cylinder":
            radius = obs.get("radius", 0.5)
            # Pillars are solid concrete gray, tables are amber
            if "pillar" in oname:
                fc = '#7F8C8D'
                ec = '#2C3E50'
            else:
                fc = '#FAD7A0'
                ec = '#B9770E'
            circle = patches.Circle(
                (pos[0], pos[1]),
                radius=radius,
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.2,
                zorder=6
            )
            ax.add_patch(circle)
        elif otype == "box":
            size = obs.get("size", [1.0, 1.0, 1.0])
            sx, sy = size[0], size[1]
            bx = pos[0] - sx / 2.0
            by = pos[1] - sy / 2.0
            if "kiosk" in oname:
                fc = '#D5D8DC'
                ec = '#2C3E50'
            else:
                fc = '#D7BDE2'
                ec = '#6C3483'
            rect = patches.Rectangle(
                (bx, by), sx, sy,
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.2,
                zorder=6
            )
            ax.add_patch(rect)

    # 4. Draw Walls (CAD solid dark lines / filled bounds)
    walls = data.get("walls", [])
    for w in walls:
        s = w["start"]
        e = w["end"]
        thick = w.get("thickness", 0.2)
        # Architectural double-line / thick dark gray wall
        ax.plot(
            [s[0], e[0]], [s[1], e[1]],
            color='#1B2631',
            linewidth=thick * 22,  # scale to visual CAD wall weight
            solid_capstyle='projecting',
            zorder=7
        )

    # 5. Draw Cameras and Ray-Cast Wall-Occluded Line-of-Sight Visibility Polygons
    cam_colors = ['#E74C3C', '#2980B9', '#16A085', '#8E44AD', '#D35400', '#2C3E50']
    cameras = data.get("cameras", [])
    wall_segments = extract_wall_segments(data)

    for idx, cam in enumerate(cameras):
        color = cam.get("color") or cam_colors[idx % len(cam_colors)]
        cid = cam["camera_id"]
        cname = cam.get("name", cid)
        cpos = cam["position"]

        # Compute authentic 3D ray-cast visibility polygon with wall occlusion & nadir blind spot
        vis_poly, d_min, heading_deg, hfov_deg = compute_raycast_visibility_polygon(cam, wall_segments)

        # 1. Wall-Occluded Line-of-Sight Polygon (Stops at walls, zero penetration)
        if len(vis_poly) >= 3:
            los_patch = patches.Polygon(
                vis_poly,
                closed=True,
                facecolor=color,
                alpha=0.22,
                edgecolor=color,
                linewidth=1.4,
                linestyle='-',
                zorder=8
            )
            ax.add_patch(los_patch)

        # 2. Near Nadir Blind Spot Arc (physical blind zone immediately under camera)
        half_hfov = hfov_deg / 2.0
        blind_wedge = patches.Wedge(
            center=(cpos[0], cpos[1]),
            r=d_min,
            theta1=heading_deg - half_hfov,
            theta2=heading_deg + half_hfov,
            facecolor='#FF3838',
            alpha=0.18,
            edgecolor='#FF3838',
            linewidth=1.0,
            linestyle='--',
            zorder=8
        )
        ax.add_patch(blind_wedge)

        # Draw camera center axis ray
        rad_h = math.radians(heading_deg)
        look_dist = min(cam.get("max_range_m", 35.0), 16.0)
        rx = cpos[0] + look_dist * math.cos(rad_h)
        ry = cpos[1] + look_dist * math.sin(rad_h)
        ax.plot([cpos[0], rx], [cpos[1], ry], color=color, linestyle='--', linewidth=1.2, alpha=0.7, zorder=9)

        # Camera physical mount marker (CCTV Icon / Diamond & circle)
        ax.scatter(cpos[0], cpos[1], s=160, c='#FFFFFF', edgecolors=color, linewidths=2.5, zorder=10)
        ax.scatter(cpos[0], cpos[1], s=50, c=color, marker='s', zorder=11)

        # Camera directional chevron
        ch_len = 1.2
        ch_x = cpos[0] + ch_len * math.cos(rad_h)
        ch_y = cpos[1] + ch_len * math.sin(rad_h)
        ax.annotate(
            '',
            xy=(ch_x, ch_y),
            xytext=(cpos[0], cpos[1]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=15),
            zorder=12
        )

        # Camera Label Callout
        # Compute offset position for clean text label
        offset_dir = rad_h + math.pi
        label_x = cpos[0] + 2.4 * math.cos(offset_dir)
        label_y = cpos[1] + 1.8 * math.sin(offset_dir)

        # Keep label inside plot limits
        label_x = max(1.5, min(W - 1.5, label_x))
        label_y = max(1.0, min(H - 1.0, label_y))

        ax.annotate(
            f"{cid}\n{cname}\nHFOV: {hfov_deg:.1f}° | Mount: {float(cpos[2]):.1f}m\nLOS Wall-Clipped",
            xy=(cpos[0], cpos[1]),
            xytext=(label_x, label_y),
            fontsize=8.0,
            fontweight='bold',
            color='#17202A',
            ha='center',
            va='center',
            zorder=13,
            bbox=dict(boxstyle="square,pad=0.35", facecolor='#FFFFFF', edgecolor=color, linewidth=1.5, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2, shrinkA=4, shrinkB=6)
        )

    # 6. Architectural Scale Bar (10 meters)
    scale_x0 = 3.0
    scale_y0 = -2.2
    scale_len = 10.0  # 10m
    ax.plot([scale_x0, scale_x0 + scale_len], [scale_y0, scale_y0], color='#1B2631', linewidth=3.0, zorder=15)
    ax.plot([scale_x0, scale_x0], [scale_y0 - 0.3, scale_y0 + 0.3], color='#1B2631', linewidth=2.0, zorder=15)
    ax.plot([scale_x0 + 5.0, scale_x0 + 5.0], [scale_y0 - 0.2, scale_y0 + 0.2], color='#1B2631', linewidth=1.5, zorder=15)
    ax.plot([scale_x0 + scale_len, scale_x0 + scale_len], [scale_y0 - 0.3, scale_y0 + 0.3], color='#1B2631', linewidth=2.0, zorder=15)
    ax.text(scale_x0 + 5.0, scale_y0 - 0.7, "SCALE BAR: 10 METERS (1:100 CAD)", ha='center', va='top', fontsize=8.0, fontweight='bold', color='#1B2631')

    # 7. North Arrow (Top-Right)
    na_x = W - 3.0
    na_y = H + 2.0
    ax.annotate(
        'N',
        xy=(na_x, na_y),
        xytext=(na_x, na_y - 1.2),
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='heavy',
        color='#1B2631',
        arrowprops=dict(facecolor='#1B2631', edgecolor='#1B2631', width=2.5, headwidth=8.0, headlength=10.0),
        zorder=15
    )

    # 8. Architectural Title Block (Bottom-Right)
    tb_w = 16.0
    tb_h = 2.4
    tb_x0 = W - tb_w
    tb_y0 = -3.2
    tb_rect = patches.Rectangle(
        (tb_x0, tb_y0), tb_w, tb_h,
        facecolor='#F8F9FA',
        edgecolor='#1B2631',
        linewidth=1.5,
        zorder=14
    )
    ax.add_patch(tb_rect)

    title_text = (
        "PROJECT: CROWDSAFE SYNTHETIC DATASET ENGINE\n"
        f"DRAWING: LEVEL 01 ARCHITECTURAL FLOOR PLAN & CCTV FOV COVERAGE\n"
        f"VENUE: {data.get('venue_name', 'MALL GROUND FLOOR')} ({W:.0f}m x {H:.0f}m)\n"
        f"SURVEILLANCE: {len(cameras)} CALIBRATED CAMERAS | FOV RANGE: 15.0m"
    )
    ax.text(
        tb_x0 + 0.4, tb_y0 + tb_h / 2.0,
        title_text,
        ha='left',
        va='center',
        fontsize=7.5,
        fontweight='semibold',
        family='monospace',
        color='#17202A',
        zorder=15
    )

    # Axis Labels & Title
    ax.set_xlabel("X Dimension (Meters)", fontsize=10, fontweight='bold', color='#2C3E50')
    ax.set_ylabel("Y Dimension (Meters)", fontsize=10, fontweight='bold', color='#2C3E50')
    ax.set_title(
        "CROWDSAFE ARCHITECTURAL FLOOR PLAN & CALIBRATED CCTV COVERAGE MAP",
        fontsize=14,
        fontweight='heavy',
        pad=18,
        color='#1A252F'
    )

    # Adjust layout & save
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"[✓] Floor plan PNG exported to: {output_png}")

    if output_svg:
        fig.savefig(output_svg, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"[✓] Floor plan SVG exported to: {output_svg}")

    plt.close(fig)
    return output_png


def main():
    parser = argparse.ArgumentParser(description="Export CAD 2D floor plan with CCTV FOV cones.")
    parser.add_argument(
        "--plan",
        type=str,
        default="dataset-generator/venue/mall_floor_plan.json",
        help="Path to mall_floor_plan.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset-generator/output/floor_plan_with_cameras.png",
        help="Path to output PNG"
    )
    parser.add_argument(
        "--svg",
        type=str,
        default="dataset-generator/output/floor_plan_with_cameras.svg",
        help="Path to output SVG (optional)"
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    args = parser.parse_args()

    draw_cad_floor_plan(
        plan_path=args.plan,
        output_png=args.output,
        output_svg=args.svg,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
