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
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path


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

    dims = data.get("dimensions", {"width_x": 50.0, "depth_y": 30.0})
    W = dims["width_x"]
    H = dims["depth_y"]

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

    # 5. Draw Cameras and Semi-Transparent FOV Cones
    cam_colors = ['#E74C3C', '#2980B9', '#16A085', '#8E44AD', '#D35400', '#2C3E50']
    cameras = data.get("cameras", [])

    for idx, cam in enumerate(cameras):
        c_meta = calculate_camera_fov(cam, default_range=15.0)
        color = cam_colors[idx % len(cam_colors)]
        cid = cam["camera_id"]
        cname = cam.get("name", cid)

        # FOV Wedge (Conical coverage)
        wedge = patches.Wedge(
            center=(c_meta["x"], c_meta["y"]),
            r=c_meta["range_m"],
            theta1=c_meta["theta1"],
            theta2=c_meta["theta2"],
            facecolor=color,
            alpha=0.20,
            edgecolor=color,
            linewidth=1.5,
            linestyle='-',
            zorder=8
        )
        ax.add_patch(wedge)

        # Draw camera center axis ray
        rad_h = math.radians(c_meta["heading_deg"])
        rx = c_meta["x"] + c_meta["range_m"] * math.cos(rad_h)
        ry = c_meta["y"] + c_meta["range_m"] * math.sin(rad_h)
        ax.plot([c_meta["x"], rx], [c_meta["y"], ry], color=color, linestyle='--', linewidth=1.2, alpha=0.7, zorder=9)

        # Camera physical mount marker (CCTV Icon / Diamond & circle)
        ax.scatter(c_meta["x"], c_meta["y"], s=160, c='#FFFFFF', edgecolors=color, linewidths=2.5, zorder=10)
        ax.scatter(c_meta["x"], c_meta["y"], s=50, c=color, marker='s', zorder=11)

        # Camera directional chevron
        ch_len = 1.2
        ch_x = c_meta["x"] + ch_len * math.cos(rad_h)
        ch_y = c_meta["y"] + ch_len * math.sin(rad_h)
        ax.annotate(
            '',
            xy=(ch_x, ch_y),
            xytext=(c_meta["x"], c_meta["y"]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=15),
            zorder=12
        )

        # Camera Label Callout
        # Compute offset position for clean text label
        offset_dir = rad_h + math.pi
        label_x = c_meta["x"] + 2.4 * math.cos(offset_dir)
        label_y = c_meta["y"] + 1.8 * math.sin(offset_dir)

        # Keep label inside plot limits
        label_x = max(1.5, min(W - 1.5, label_x))
        label_y = max(1.0, min(H - 1.0, label_y))

        ax.annotate(
            f"{cid}\n{cname}\nHFOV: {c_meta['fov_deg']:.1f}° | Mount: {c_meta['z']:.1f}m",
            xy=(c_meta["x"], c_meta["y"]),
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
