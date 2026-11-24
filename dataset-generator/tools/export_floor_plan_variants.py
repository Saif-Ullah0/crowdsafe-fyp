#!/usr/bin/env python3
"""
CrowdSafe - Floor Plan Variant Exporter (Design 4 Transit Interchange)
Generates 3 specialized 2D floor plans:
  1. Clean Architectural Base (NO camera FOV cones - pure architectural CAD)
  2. Pedestrian / Human Location Dots (Live crowd tracking & density visualization)
  3. Tactical Game Map / Dashboard View (Realistic dark-mode tactical command map)
"""

import os
import sys
import json
import math
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patheffects as pe

OUTPUT_DIR = "dataset-generator/output/floor_plans_2d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# Geometry Definitions for Design 4
# -------------------------------------------------------------
W = 38.0
H = 38.0
rw = 14.0   # corner room dimension
cw = 10.0   # walkway corridor width (14m to 24m)
r_fountain = 2.5
r_rotary = 5.0
center = (19.0, 19.0)

rooms_data = [
    {
        "id": "ROOM_NW",
        "name": "TICKETING & TRANSIT CONCOURSE",
        "sub": "Customer Service & Automation Hub",
        "rect": (0.0, 24.0, 14.0, 14.0),
        "door": (14.0, 31.0, 1.2, "east", "inward_left"),
        "code": "ZONE-NW (196 m²)",
        "color_cad": "#F3F4F6",
        "color_tac": "#0f172a"
    },
    {
        "id": "ROOM_NE",
        "name": "FACILITY INFRASTRUCTURE",
        "sub": "Electrical, HVAC & Restrooms",
        "rect": (24.0, 24.0, 14.0, 14.0),
        "door": (24.0, 31.0, 1.2, "west", "inward_right"),
        "code": "ZONE-NE (196 m²)",
        "color_cad": "#F3F4F6",
        "color_tac": "#0f172a"
    },
    {
        "id": "ROOM_SW",
        "name": "SECURITY OPERATIONS CENTER",
        "sub": "CCTV Dispatch & Emergency Ops",
        "rect": (0.0, 0.0, 14.0, 14.0),
        "door": (14.0, 7.0, 1.2, "east", "inward_left"),
        "code": "ZONE-SW (196 m²)",
        "color_cad": "#F3F4F6",
        "color_tac": "#0f172a"
    },
    {
        "id": "ROOM_SE",
        "name": "PASSENGER LOUNGE & REST",
        "sub": "Seating & Transit Information",
        "rect": (24.0, 0.0, 14.0, 14.0),
        "door": (24.0, 7.0, 1.2, "west", "inward_right"),
        "code": "ZONE-SE (196 m²)",
        "color_cad": "#F3F4F6",
        "color_tac": "#0f172a"
    }
]

gateways_data = [
    {"id": "GW-S", "name": "SOUTH EXIT GATEWAY", "dir": "S", "pos": (19.0, 0.0), "span": (14.0, 0.0, 24.0, 0.0), "cam_pos": (19.0, 0.6), "cam_dir": (0, 1)},
    {"id": "GW-N", "name": "NORTH EXIT GATEWAY", "dir": "N", "pos": (19.0, 38.0), "span": (14.0, 38.0, 24.0, 38.0), "cam_pos": (19.0, 37.4), "cam_dir": (0, -1)},
    {"id": "GW-W", "name": "WEST EXIT GATEWAY", "dir": "W", "pos": (0.0, 19.0), "span": (0.0, 14.0, 0.0, 24.0), "cam_pos": (0.6, 19.0), "cam_dir": (1, 0)},
    {"id": "GW-E", "name": "EAST EXIT GATEWAY", "dir": "E", "pos": (38.0, 19.0), "span": (38.0, 14.0, 38.0, 24.0), "cam_pos": (37.4, 19.0), "cam_dir": (-1, 0)}
]

wall_segments = [
    ((0.0, 0.0), (14.0, 0.0)),
    ((24.0, 0.0), (38.0, 0.0)),
    ((0.0, 38.0), (14.0, 38.0)),
    ((24.0, 38.0), (38.0, 38.0)),
    ((0.0, 0.0), (0.0, 14.0)),
    ((0.0, 24.0), (0.0, 38.0)),
    ((38.0, 0.0), (38.0, 14.0)),
    ((38.0, 24.0), (38.0, 38.0)),
    ((14.0, 0.0), (14.0, 14.0)),
    ((0.0, 14.0), (14.0, 14.0)),
    ((24.0, 0.0), (24.0, 14.0)),
    ((24.0, 14.0), (38.0, 14.0)),
    ((0.0, 24.0), (14.0, 24.0)),
    ((14.0, 24.0), (14.0, 38.0)),
    ((24.0, 24.0), (38.0, 24.0)),
    ((24.0, 24.0), (24.0, 38.0))
]


# =============================================================
# 1. CLEAN ARCHITECTURAL BASE FLOOR PLAN (No Camera Projections)
# =============================================================
def draw_clean_base_plan(output_png: str, output_svg: str, dpi: int = 300):
    fig, ax = plt.subplots(figsize=(14, 14), dpi=dpi)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FAFAFA')

    # Grid
    ax.set_xlim(-3.5, 41.5)
    ax.set_ylim(-4.0, 42.0)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', color='#D5D8DC', alpha=0.7, zorder=0)

    # Concourse Walkway Surface (Plus Shape)
    plus_poly = [
        (14.0, 0.0), (24.0, 0.0),
        (24.0, 14.0), (38.0, 14.0),
        (38.0, 24.0), (24.0, 24.0),
        (24.0, 38.0), (14.0, 38.0),
        (14.0, 24.0), (0.0, 24.0),
        (0.0, 14.0), (14.0, 14.0)
    ]
    concourse_patch = patches.Polygon(plus_poly, closed=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.5, zorder=1)
    ax.add_patch(concourse_patch)

    # Walkway subtle tile lines (every 2m)
    for x in range(16, 24, 2):
        ax.plot([x, x], [0, 38], color='#F1F5F9', linewidth=1.0, zorder=2)
    for y in range(16, 24, 2):
        ax.plot([0, 38], [y, y], color='#F1F5F9', linewidth=1.0, zorder=2)

    # 4 Corner Rooms
    for r in rooms_data:
        rx, ry, rw_dim, rh_dim = r["rect"]
        room_bg = patches.Rectangle((rx, ry), rw_dim, rh_dim, facecolor=r["color_cad"], edgecolor='#94A3B8', linewidth=1.5, zorder=3)
        ax.add_patch(room_bg)

        # Architectural cross-hatch for solid room structure
        hatch_bg = patches.Rectangle((rx, ry), rw_dim, rh_dim, facecolor='none', edgecolor='#CBD5E1', hatch='...', linewidth=0, zorder=3)
        ax.add_patch(hatch_bg)

        # Labels
        ax.text(rx + rw_dim/2.0, ry + rh_dim/2.0 + 1.2, r["name"], ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E293B', zorder=5)
        ax.text(rx + rw_dim/2.0, ry + rh_dim/2.0 - 0.4, r["sub"], ha='center', va='center', fontsize=7.5, color='#64748B', style='italic', zorder=5)
        ax.text(rx + rw_dim/2.0, ry + rh_dim/2.0 - 1.8, r["code"], ha='center', va='center', fontsize=8.0, fontweight='semibold', color='#0284C7', zorder=5)

    # Inward Doors & Swing Arcs
    for r in rooms_data:
        dx, dy, dw, wall, swing = r["door"]
        ax.plot([dx, dx], [dy - dw/2, dy + dw/2] if wall in ["east", "west"] else [dx - dw/2, dx + dw/2], [dy, dy], color='#0284C7', linewidth=3.5, zorder=8)
        
        # Draw 90 deg inward door swing
        if wall == "east" and swing == "inward_left":
            arc = patches.Arc((dx, dy - dw/2), dw*2, dw*2, angle=0, theta1=90, theta2=180, color='#0284C7', linestyle='--', linewidth=1.5, zorder=7)
            ax.add_patch(arc)
            ax.plot([dx, dx - dw], [dy - dw/2, dy - dw/2], color='#0284C7', linewidth=2.0, zorder=8)
            ax.text(dx - 0.4, dy + 1.2, "INWARD DOOR (1.2m)", ha='right', va='center', fontsize=6.5, fontweight='bold', color='#0284C7', zorder=8)
        elif wall == "west" and swing == "inward_right":
            arc = patches.Arc((dx, dy - dw/2), dw*2, dw*2, angle=0, theta1=0, theta2=90, color='#0284C7', linestyle='--', linewidth=1.5, zorder=7)
            ax.add_patch(arc)
            ax.plot([dx, dx + dw], [dy - dw/2, dy - dw/2], color='#0284C7', linewidth=2.0, zorder=8)
            ax.text(dx + 0.4, dy + 1.2, "INWARD DOOR (1.2m)", ha='left', va='center', fontsize=6.5, fontweight='bold', color='#0284C7', zorder=8)

    # Central Roundabout & Clock Tower Fountain (Green rotary path removed)
    # Fountain Base (Outer Basin)
    fountain_basin = patches.Circle(center, r_fountain, facecolor='#E0F2FE', edgecolor='#0284C7', linewidth=2.0, zorder=6)
    ax.add_patch(fountain_basin)

    # Water ripple concentric rings
    for r_rip in [1.8, 1.1]:
        rip = patches.Circle(center, r_rip, facecolor='none', edgecolor='#38BDF8', linestyle=':', linewidth=1.2, zorder=7)
        ax.add_patch(rip)

    # Center Clock Tower Core Column
    clock_core = patches.Circle(center, 0.6, facecolor='#0369A1', edgecolor='#082F49', linewidth=2.0, zorder=8)
    ax.add_patch(clock_core)
    ax.text(center[0], center[1], "TIME", ha='center', va='center', fontsize=8, fontweight='heavy', color='#FFFFFF', zorder=9)
    ax.text(center[0], center[1] - 3.2, "ROTARY CLOCK FOUNTAIN ROUNDABOUT", ha='center', va='top', fontsize=8.0, fontweight='bold', color='#0369A1', zorder=8)
    ax.text(center[0], center[1] - 4.1, "Ø 5.0m Central Fountain Basin", ha='center', va='top', fontsize=7.0, color='#64748B', zorder=8)

    # 4 Exit Gateways & Speed Gates
    for gw in gateways_data:
        gx, gy = gw["pos"]
        gdir = gw["dir"]
        
        # Gantry banner
        if gdir in ["S", "N"]:
            # Horizontal portal
            gate_box = patches.Rectangle((14.0, gy - 0.7 if gdir == "S" else gy - 0.7), 10.0, 1.4, facecolor='#F8FAFC', edgecolor='#0284C7', linewidth=2.0, zorder=8)
            ax.add_patch(gate_box)
            # Threshold matting
            mat = patches.Rectangle((14.0, 0.0 if gdir == "S" else 36.6), 10.0, 1.4, facecolor='#E2E8F0', edgecolor='none', zorder=2)
            ax.add_patch(mat)
            # Turnstile flippers
            for st_x in np.linspace(15.2, 22.8, 5):
                ax.plot([st_x, st_x], [gy - 0.5 if gdir == "S" else gy - 0.9, gy + 0.5 if gdir == "S" else gy + 0.1], color='#334155', linewidth=3.5, zorder=9)
            # Gateway Tag
            ax.text(19.0, -1.6 if gdir == "S" else 39.4, f"[{gw['id']}] {gw['name']} (4-LANE SPEED GATES)", ha='center', va='center', fontsize=8.5, fontweight='heavy', color='#0F172A', zorder=10)
        else:
            # Vertical portal
            gate_box = patches.Rectangle((gx - 0.7 if gdir == "W" else gx - 0.7, 14.0), 1.4, 10.0, facecolor='#F8FAFC', edgecolor='#0284C7', linewidth=2.0, zorder=8)
            ax.add_patch(gate_box)
            # Threshold matting
            mat = patches.Rectangle((0.0 if gdir == "W" else 36.6, 14.0), 1.4, 10.0, facecolor='#E2E8F0', edgecolor='none', zorder=2)
            ax.add_patch(mat)
            # Turnstile flippers
            for st_y in np.linspace(15.2, 22.8, 5):
                ax.plot([gx - 0.5 if gdir == "W" else gx - 0.9, gx + 0.5 if gdir == "W" else gx + 0.1], [st_y, st_y], color='#334155', linewidth=3.5, zorder=9)
            # Gateway Tag
            ax.text(-2.0 if gdir == "W" else 40.0, 19.0, f"[{gw['id']}] " + gw['name'], ha='center', va='center', fontsize=7.5, fontweight='heavy', color='#0F172A', rotation=90 if gdir == "W" else -90, zorder=10)

    # CCTV Mount Icons (Markers ONLY - NO colored FOV cone projection)
    for gw in gateways_data:
        cx, cy = gw["cam_pos"]
        cid = gw["id"].replace("GW-", "CAM-")
        ax.plot(cx, cy, marker='o', markersize=9, color='#DC2626', markeredgecolor='#7F1D1D', markeredgewidth=1.8, zorder=12)
        ax.plot(cx, cy, marker='.', markersize=4, color='#FFFFFF', zorder=13)
        # Small Camera Mount Box Tag
        offset_y = 1.3 if gw["dir"] == "S" else (-1.3 if gw["dir"] == "N" else 0.0)
        offset_x = 1.6 if gw["dir"] == "W" else (-1.6 if gw["dir"] == "E" else 0.0)
        ax.text(cx + offset_x, cy + offset_y, f"{cid}\n(4.6m Mount)", ha='center', va='center', fontsize=7.0, fontweight='bold', color='#B91C1C', bbox=dict(boxstyle='round,pad=0.2', facecolor='#FEF2F2', edgecolor='#DC2626', lw=1), zorder=14)

    # Solid Structural Walls
    for (p1, p2) in wall_segments:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#0F172A', linewidth=4.0, solid_capstyle='projecting', zorder=10)

    # Perimeter Outer Wall Lines
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], color='#475569', linewidth=1.5, linestyle='-', zorder=9)

    # Architectural Scale Bar (10m)
    sb_x, sb_y = 3.0, -2.6
    ax.plot([sb_x, sb_x + 10.0], [sb_y, sb_y], color='#0F172A', linewidth=3.5, zorder=15)
    ax.plot([sb_x, sb_x], [sb_y - 0.3, sb_y + 0.3], color='#0F172A', linewidth=2.0, zorder=15)
    ax.plot([sb_x + 5.0, sb_x + 5.0], [sb_y - 0.2, sb_y + 0.2], color='#0F172A', linewidth=1.5, zorder=15)
    ax.plot([sb_x + 10.0, sb_x + 10.0], [sb_y - 0.3, sb_y + 0.3], color='#0F172A', linewidth=2.0, zorder=15)
    ax.text(sb_x + 5.0, sb_y - 0.7, "SCALE: 10 METERS (1:100 CAD)", ha='center', va='top', fontsize=8.0, fontweight='bold', color='#0F172A')

    # North Arrow
    ax.annotate('N', xy=(W - 2.5, H + 2.4), xytext=(W - 2.5, H + 1.2),
                ha='center', va='bottom', fontsize=12, fontweight='heavy', color='#0F172A',
                arrowprops=dict(facecolor='#0F172A', edgecolor='#0F172A', width=2.5, headwidth=8.0, headlength=10.0), zorder=15)

    # Title Block
    tb_w, tb_h = 17.5, 2.6
    tb_x, tb_y = W - tb_w, -3.7
    tb_rect = patches.Rectangle((tb_x, tb_y), tb_w, tb_h, facecolor='#FFFFFF', edgecolor='#0F172A', linewidth=1.5, zorder=14)
    ax.add_patch(tb_rect)
    ax.text(tb_x + 0.4, tb_y + tb_h/2.0,
            "CROWDSAFE SYNTHETIC DATASET ENGINE — BASE ARCHITECTURAL 2D PLAN\n"
            "DRAWING: LEVEL 01 BASE CONCOURSE (NO CCTV PROJECTIONS)\n"
            f"VENUE: CONTROLLED TRANSIT INTERCHANGE ({W:.0f}m x {H:.0f}m)\n"
            "PURPOSE: CLEAN BASE FOR ARBITRARY CAMERA PROJECTION OVERLAYS",
            ha='left', va='center', fontsize=7.0, fontweight='bold', family='monospace', color='#0F172A', zorder=15)

    ax.set_title("CROWDSAFE — TRANSIT INTERCHANGE BASE ARCHITECTURAL FLOOR PLAN\n[CLEAN BASE: NO CAMERA FOV CONES]",
                 fontsize=13, fontweight='heavy', pad=18, color='#0F172A')
    ax.set_xlabel("X Dimension (Meters)", fontsize=9.5, fontweight='bold', color='#334155')
    ax.set_ylabel("Y Dimension (Meters)", fontsize=9.5, fontweight='bold', color='#334155')

    plt.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    if output_svg:
        fig.savefig(output_svg, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Generated Clean Base Floor Plan: {output_png}")


# =============================================================
# 2. PEDESTRIAN / HUMAN LOCATION DOTS FLOOR PLAN
# =============================================================
def draw_human_locations_plan(output_png: str, output_svg: str, dpi: int = 300):
    fig, ax = plt.subplots(figsize=(14, 14), dpi=dpi)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    ax.set_xlim(-3.5, 41.5)
    ax.set_ylim(-4.0, 42.0)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', color='#E2E8F0', alpha=0.8, zorder=0)

    # Concourse Walkway Surface
    plus_poly = [
        (14.0, 0.0), (24.0, 0.0),
        (24.0, 14.0), (38.0, 14.0),
        (38.0, 24.0), (24.0, 24.0),
        (24.0, 38.0), (14.0, 38.0),
        (14.0, 24.0), (0.0, 24.0),
        (0.0, 14.0), (14.0, 14.0)
    ]
    ax.add_patch(patches.Polygon(plus_poly, closed=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.5, zorder=1))

    # Floor grid tiles
    for x in range(16, 24, 2):
        ax.plot([x, x], [0, 38], color='#F1F5F9', linewidth=1.0, zorder=2)
    for y in range(16, 24, 2):
        ax.plot([0, 38], [y, y], color='#F1F5F9', linewidth=1.0, zorder=2)

    # Rooms
    for r in rooms_data:
        rx, ry, rw_dim, rh_dim = r["rect"]
        ax.add_patch(patches.Rectangle((rx, ry), rw_dim, rh_dim, facecolor='#F1F5F9', edgecolor='#94A3B8', linewidth=1.5, zorder=3))
        ax.text(rx + rw_dim/2.0, ry + rh_dim/2.0 + 0.5, r["name"], ha='center', va='center', fontsize=9.0, fontweight='bold', color='#334155', zorder=5)
        ax.text(rx + rw_dim/2.0, ry + rh_dim/2.0 - 1.2, r["code"], ha='center', va='center', fontsize=7.5, color='#64748B', zorder=5)

    # Central Roundabout (Green rotary path removed)
    ax.add_patch(patches.Circle(center, r_fountain, facecolor='#E0F2FE', edgecolor='#0284C7', linewidth=2.0, zorder=5))
    ax.add_patch(patches.Circle(center, 0.6, facecolor='#0369A1', edgecolor='#082F49', linewidth=2.0, zorder=6))
    ax.text(center[0], center[1], "TIME", ha='center', va='center', fontsize=8, fontweight='heavy', color='#FFFFFF', zorder=7)

    # Exit Gateways
    for gw in gateways_data:
        gx, gy = gw["pos"]
        gdir = gw["dir"]
        if gdir in ["S", "N"]:
            ax.add_patch(patches.Rectangle((14.0, gy - 0.7 if gdir == "S" else gy - 0.7), 10.0, 1.4, facecolor='#F8FAFC', edgecolor='#0284C7', linewidth=1.8, zorder=6))
            for st_x in np.linspace(15.2, 22.8, 5):
                ax.plot([st_x, st_x], [gy - 0.5 if gdir == "S" else gy - 0.9, gy + 0.5 if gdir == "S" else gy + 0.1], color='#334155', linewidth=3.0, zorder=7)
            ax.text(19.0, -1.6 if gdir == "S" else 39.4, f"[{gw['id']}] {gw['name']}", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#0F172A', zorder=8)
        else:
            ax.add_patch(patches.Rectangle((gx - 0.7 if gdir == "W" else gx - 0.7, 14.0), 1.4, 10.0, facecolor='#F8FAFC', edgecolor='#0284C7', linewidth=1.8, zorder=6))
            for st_y in np.linspace(15.2, 22.8, 5):
                ax.plot([gx - 0.5 if gdir == "W" else gx - 0.9, gx + 0.5 if gdir == "W" else gx + 0.1], [st_y, st_y], color='#334155', linewidth=3.0, zorder=7)
            ax.text(-2.0 if gdir == "W" else 40.0, 19.0, f"[{gw['id']}] " + gw['name'], ha='center', va='center', fontsize=7.0, fontweight='bold', color='#0F172A', rotation=90 if gdir == "W" else -90, zorder=8)

    # Structural Walls
    for (p1, p2) in wall_segments:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#0F172A', linewidth=4.0, solid_capstyle='projecting', zorder=10)

    # -------------------------------------------------------------
    # Simulated Real-Life Crowd Pedestrian Dots
    # -------------------------------------------------------------
    random.seed(42)
    np.random.seed(42)
    pedestrians = []

    # Stream 1: South Gateway Ingress/Egress (28 pedestrians)
    for _ in range(28):
        px = np.random.uniform(14.8, 23.2)
        py = np.random.uniform(1.2, 13.5)
        v_dir = random.choice([(0, 1), (0, 1), (0, -1), (0.2, 0.9), (-0.2, 0.9)])
        speed = np.random.uniform(0.8, 1.4)
        pedestrians.append((px, py, v_dir[0] * speed, v_dir[1] * speed, "stream_south"))

    # Stream 2: North Gateway Ingress/Egress (28 pedestrians)
    for _ in range(28):
        px = np.random.uniform(14.8, 23.2)
        py = np.random.uniform(24.5, 36.8)
        v_dir = random.choice([(0, -1), (0, -1), (0, 1), (0.2, -0.9), (-0.2, -0.9)])
        speed = np.random.uniform(0.8, 1.4)
        pedestrians.append((px, py, v_dir[0] * speed, v_dir[1] * speed, "stream_north"))

    # Stream 3: West Gateway Ingress/Egress (25 pedestrians)
    for _ in range(25):
        px = np.random.uniform(1.2, 13.5)
        py = np.random.uniform(14.8, 23.2)
        v_dir = random.choice([(1, 0), (1, 0), (-1, 0), (0.9, 0.2), (0.9, -0.2)])
        speed = np.random.uniform(0.8, 1.4)
        pedestrians.append((px, py, v_dir[0] * speed, v_dir[1] * speed, "stream_west"))

    # Stream 4: East Gateway Ingress/Egress (25 pedestrians)
    for _ in range(25):
        px = np.random.uniform(24.5, 36.8)
        py = np.random.uniform(14.8, 23.2)
        v_dir = random.choice([(-1, 0), (-1, 0), (1, 0), (-0.9, 0.2), (-0.9, -0.2)])
        speed = np.random.uniform(0.8, 1.4)
        pedestrians.append((px, py, v_dir[0] * speed, v_dir[1] * speed, "stream_east"))

    # Stream 5: Rotary Pedestrian Circulation (26 pedestrians in circular motion)
    for i in range(26):
        theta = np.random.uniform(0, 2 * math.pi)
        r = np.random.uniform(2.8, 4.7)  # inside rotary lane
        px = center[0] + r * math.cos(theta)
        py = center[1] + r * math.sin(theta)
        # Tangential counter-clockwise velocity
        vx = -math.sin(theta) * 1.1
        vy = math.cos(theta) * 1.1
        pedestrians.append((px, py, vx, vy, "rotary"))

    # Stream 6: Door transitions (10 pedestrians entering/exiting rooms)
    door_targets = [(14.0, 7.0), (24.0, 7.0), (14.0, 31.0), (24.0, 31.0)]
    for (dx, dy) in door_targets:
        for _ in range(2):
            px = dx + np.random.uniform(-0.8, 0.8)
            py = dy + np.random.uniform(-1.0, 1.0)
            pedestrians.append((px, py, 0.5, 0.2, "door"))

    total_count = len(pedestrians)

    # Density heatmap contours (low-opacity blobs under clusters)
    for (cx_k, cy_k, r_k, density_lvl) in [
        (19.0, 5.0, 3.2, "MODERATE (0.18 p/m²)"),
        (19.0, 33.0, 3.2, "MODERATE (0.19 p/m²)"),
        (19.0, 19.0, 4.8, "FREE FLOW (0.12 p/m²)")
    ]:
        cl_patch = patches.Circle((cx_k, cy_k), r_k, facecolor='#FEF08A', edgecolor='#EAB308', linestyle='--', linewidth=1.2, alpha=0.35, zorder=8)
        ax.add_patch(cl_patch)
        ax.text(cx_k, cy_k + r_k + 0.3, density_lvl, ha='center', va='bottom', fontsize=6.5, fontweight='bold', color='#A16207', zorder=9)

    # Plot Pedestrian Dots & Velocity Direction Tails
    for (px, py, vx, vy, group) in pedestrians:
        # Direction arrow / tail
        tail_len = 0.8
        norm = math.hypot(vx, vy)
        if norm > 0.1:
            dx_t = (vx / norm) * tail_len
            dy_t = (vy / norm) * tail_len
            ax.plot([px, px + dx_t], [py, py + dy_t], color='#0284C7', linewidth=1.2, alpha=0.7, zorder=11)
            ax.plot(px + dx_t, py + dy_t, marker='.', markersize=2.5, color='#0369A1', zorder=11)

        # Dot marker (Outer halo + inner core)
        ax.plot(px, py, marker='o', markersize=5.5, color='#0284C7', alpha=0.3, zorder=12)
        ax.plot(px, py, marker='o', markersize=3.8, color='#0284C7', markeredgecolor='#0369A1', markeredgewidth=0.8, zorder=13)
        ax.plot(px, py, marker='o', markersize=1.6, color='#FFFFFF', zorder=14)

    # CCTV Mount Icons
    for gw in gateways_data:
        cx, cy = gw["cam_pos"]
        cid = gw["id"].replace("GW-", "CAM-")
        ax.plot(cx, cy, marker='s', markersize=7, color='#DC2626', markeredgecolor='#7F1D1D', zorder=15)
        ax.text(cx, cy + (1.0 if gw['dir'] == 'S' else (-1.0 if gw['dir'] == 'N' else 0.0)),
                cid, ha='center', va='center', fontsize=6.5, fontweight='bold', color='#B91C1C', zorder=16)

    # Live Telemetry Header Banner
    ax.text(W/2.0, H + 2.0,
            f"LIVE OCCUPANCY: {total_count} PERSONS  |  AVERAGE SPEED: 1.18 m/s  |  LEVEL OF SERVICE: A (NORMAL FREE FLOW)",
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0F172A',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0FDF4', edgecolor='#22C55E', lw=1.5), zorder=20)

    # Legend Block (Top-Left)
    leg_x, leg_y = 1.0, H - 3.5
    leg_rect = patches.Rectangle((leg_x, leg_y), 12.0, 3.0, facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.2, zorder=18)
    ax.add_patch(leg_rect)
    ax.plot(leg_x + 0.8, leg_y + 2.2, marker='o', markersize=4.5, color='#0284C7', markeredgecolor='#0369A1')
    ax.plot([leg_x + 0.8, leg_x + 1.8], [leg_y + 2.2, leg_y + 2.2], color='#0284C7', lw=1.5)
    ax.text(leg_x + 2.3, leg_y + 2.2, "Pedestrian Location & Flow Vector", va='center', fontsize=7.5, fontweight='semibold', color='#1E293B')
    
    cl_dot = patches.Circle((leg_x + 0.8, leg_y + 1.0), 0.4, facecolor='#FEF08A', edgecolor='#EAB308')
    ax.add_patch(cl_dot)
    ax.text(leg_x + 2.3, leg_y + 1.0, "Congestion Cluster Contour (p/m²)", va='center', fontsize=7.5, color='#64748B')

    # Title Block
    tb_w, tb_h = 17.5, 2.6
    tb_x, tb_y = W - tb_w, -3.7
    ax.add_patch(patches.Rectangle((tb_x, tb_y), tb_w, tb_h, facecolor='#FFFFFF', edgecolor='#0F172A', linewidth=1.5, zorder=14))
    ax.text(tb_x + 0.4, tb_y + tb_h/2.0,
            "CROWDSAFE SYNTHETIC DATASET ENGINE — PEDESTRIAN TRACKING MAP\n"
            f"LIVE TRACKING: {total_count} DETECTED PERSON CENTROIDS (DOTS)\n"
            "ALGORITHM: REAL-TIME CENTROID LOCALIZATION & VELOCITY VECTORS\n"
            "APPLICATION: DASHBOARD SPATIAL DENSITY & BOTTLENECK MONITOR",
            ha='left', va='center', fontsize=7.0, fontweight='bold', family='monospace', color='#0F172A', zorder=15)

    ax.set_title("CROWDSAFE — TRANSIT INTERCHANGE PEDESTRIAN LOCATION MAP\n[REAL-TIME HUMAN CENTROID TRACKING DOTS]",
                 fontsize=13, fontweight='heavy', pad=22, color='#0F172A')
    ax.set_xlabel("X Dimension (Meters)", fontsize=9.5, fontweight='bold', color='#334155')
    ax.set_ylabel("Y Dimension (Meters)", fontsize=9.5, fontweight='bold', color='#334155')

    plt.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    if output_svg:
        fig.savefig(output_svg, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Generated Human Location Dots Floor Plan: {output_png}")


# =============================================================
# 3. TACTICAL GAME MAP / DASHBOARD REALISTIC FLOOR PLAN
# =============================================================
def draw_tactical_gamemap_plan(output_png: str, output_svg: str, dpi: int = 300):
    """
    Tactical Game Map / Dashboard View with Clean Architecture:
    Matches the exact clean layout, room text, inward doors, and fountain of design_4_clean_base,
    rendered with a premium dark-mode / game-map tactical command center aesthetic
    WITHOUT bulky camera FOV projection cones obscuring the walkway.
    """
    fig, ax = plt.subplots(figsize=(14, 14), dpi=dpi)
    bg_color = '#070B14'
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    ax.set_xlim(-3.5, 41.5)
    ax.set_ylim(-4.0, 42.0)
    ax.set_aspect('equal')

    # 1. Tactical Blueprint Coordinate Grid (Subtle dark cyan grid)
    for x in range(0, 40, 2):
        ax.plot([x, x], [-2, 40], color='#111A2E', linewidth=0.6, linestyle='-', zorder=0)
    for y in range(0, 40, 2):
        ax.plot([-2, 40], [y, y], color='#111A2E', linewidth=0.6, linestyle='-', zorder=0)

    # Major 10m grid lines
    for x in [0, 14, 24, 38]:
        ax.plot([x, x], [-2, 40], color='#1E293B', linewidth=1.2, linestyle='--', alpha=0.7, zorder=1)
    for y in [0, 14, 24, 38]:
        ax.plot([-2, 40], [y, y], color='#1E293B', linewidth=1.2, linestyle='--', alpha=0.7, zorder=1)

    # 2. Concourse Walkway Surface (Plus Shape - Dark Polished Slate with Glowing Cyan Edge)
    plus_poly = [
        (14.0, 0.0), (24.0, 0.0),
        (24.0, 14.0), (38.0, 14.0),
        (38.0, 24.0), (24.0, 24.0),
        (24.0, 38.0), (14.0, 38.0),
        (14.0, 24.0), (0.0, 24.0),
        (0.0, 14.0), (14.0, 14.0)
    ]
    concourse_patch = patches.Polygon(
        plus_poly, closed=True,
        facecolor='#0D1527', edgecolor='#00F0FF',
        linewidth=2.0, zorder=2
    )
    ax.add_patch(concourse_patch)

    # Walkway subtle tile lines (every 2m - matching clean base)
    for x in range(16, 24, 2):
        ax.plot([x, x], [0, 38], color='#1E2B45', linewidth=1.0, linestyle='-', zorder=3)
    for y in range(16, 24, 2):
        ax.plot([0, 38], [y, y], color='#1E2B45', linewidth=1.0, linestyle='-', zorder=3)

    # 3. 4 Corner Rooms (Matching exact clean architecture and typography of clean_base)
    for r in rooms_data:
        rx, ry, rw_dim, rh_dim = r["rect"]
        room_bg = patches.Rectangle(
            (rx, ry), rw_dim, rh_dim,
            facecolor='#0A0F1D', edgecolor='#334155',
            linewidth=1.8, zorder=3
        )
        ax.add_patch(room_bg)

        # Subtle dark hatch for room structural depth
        hatch_bg = patches.Rectangle(
            (rx, ry), rw_dim, rh_dim,
            facecolor='none', edgecolor='#1E293B',
            hatch='...', linewidth=0, alpha=0.5, zorder=3
        )
        ax.add_patch(hatch_bg)

        # Tactical corner accent bracket
        accent_color = '#38BDF8' if 'NW' in r['id'] else ('#818CF8' if 'NE' in r['id'] else ('#F59E0B' if 'SW' in r['id'] else '#34D399'))
        ax.plot([rx + 0.6, rx + 2.5], [ry + rh_dim - 0.6, ry + rh_dim - 0.6], color=accent_color, lw=2.2, zorder=5)
        ax.plot([rx + 0.6, rx + 0.6], [ry + rh_dim - 0.6, ry + rh_dim - 2.5], color=accent_color, lw=2.2, zorder=5)

        # Clean Room Labels (Identical to clean base)
        ax.text(rx + rw_dim/2.0, ry + rh_dim/2.0 + 1.2, r["name"], ha='center', va='center', fontsize=9.5, fontweight='bold', color='#F1F5F9', zorder=5)
        ax.text(rx + rw_dim/2.0, ry + rh_dim/2.0 - 0.4, r["sub"], ha='center', va='center', fontsize=7.5, color='#94A3B8', style='italic', zorder=5)
        ax.text(rx + rw_dim/2.0, ry + rh_dim/2.0 - 1.8, r["code"], ha='center', va='center', fontsize=8.0, fontweight='semibold', color=accent_color, zorder=5)

    # 4. Inward Doors & Swing Arcs (Crisp 90° inward swing into room chamber)
    for r in rooms_data:
        dx, dy, dw, wall, swing = r["door"]
        ax.plot([dx, dx], [dy - dw/2, dy + dw/2] if wall in ["east", "west"] else [dx - dw/2, dx + dw/2], [dy, dy], color='#00F0FF', linewidth=3.5, zorder=8)
        
        if wall == "east" and swing == "inward_left":
            arc = patches.Arc((dx, dy - dw/2), dw*2, dw*2, angle=0, theta1=90, theta2=180, color='#00F0FF', linestyle='--', linewidth=1.5, zorder=7)
            ax.add_patch(arc)
            ax.plot([dx, dx - dw], [dy - dw/2, dy - dw/2], color='#00F0FF', linewidth=2.0, zorder=8)
            ax.text(dx - 0.4, dy + 1.2, "INWARD DOOR (1.2m)", ha='right', va='center', fontsize=6.5, fontweight='bold', family='monospace', color='#00F0FF', zorder=8)
        elif wall == "west" and swing == "inward_right":
            arc = patches.Arc((dx, dy - dw/2), dw*2, dw*2, angle=0, theta1=0, theta2=90, color='#00F0FF', linestyle='--', linewidth=1.5, zorder=7)
            ax.add_patch(arc)
            ax.plot([dx, dx + dw], [dy - dw/2, dy - dw/2], color='#00F0FF', linewidth=2.0, zorder=8)
            ax.text(dx + 0.4, dy + 1.2, "INWARD DOOR (1.2m)", ha='left', va='center', fontsize=6.5, fontweight='bold', family='monospace', color='#00F0FF', zorder=8)

    # 5. Central Roundabout & Clock Tower Fountain (Clean Architecture + Tactical Glow - Green rotary path removed)
    # Outer Fountain Basin (Glowing deep blue-cyan)
    fountain_basin = patches.Circle(center, r_fountain, facecolor='#082F49', edgecolor='#00F0FF', linewidth=2.2, zorder=6)
    ax.add_patch(fountain_basin)

    # Water ripple concentric rings
    for r_rip in [1.8, 1.1]:
        rip = patches.Circle(center, r_rip, facecolor='none', edgecolor='#38BDF8', linestyle=':', linewidth=1.3, alpha=0.8, zorder=7)
        ax.add_patch(rip)

    # Center Clock Tower Core Column
    clock_core = patches.Circle(center, 0.6, facecolor='#0284C7', edgecolor='#FFFFFF', linewidth=2.0, zorder=8)
    ax.add_patch(clock_core)
    ax.text(center[0], center[1], "TIME", ha='center', va='center', fontsize=8, fontweight='heavy', family='monospace', color='#FFFFFF', zorder=9)
    ax.text(center[0], center[1] - 3.2, "ROTARY CLOCK FOUNTAIN ROUNDABOUT", ha='center', va='top', fontsize=8.0, fontweight='heavy', family='monospace', color='#00F0FF', zorder=8)
    ax.text(center[0], center[1] - 4.1, "Ø 5.0m Central Fountain Basin", ha='center', va='top', fontsize=7.0, family='monospace', color='#94A3B8', zorder=8)

    # 6. 4 Exit Gateways & Speed Gates (Clean Frames + Green LED Pass Bars)
    for gw in gateways_data:
        gx, gy = gw["pos"]
        gdir = gw["dir"]
        
        if gdir in ["S", "N"]:
            # Portal frame
            gate_box = patches.Rectangle((14.0, gy - 0.7 if gdir == "S" else gy - 0.7), 10.0, 1.4, facecolor='#0F172A', edgecolor='#00F0FF', linewidth=2.0, zorder=8)
            ax.add_patch(gate_box)
            # Threshold matting
            mat = patches.Rectangle((14.0, 0.0 if gdir == "S" else 36.6), 10.0, 1.4, facecolor='#1E293B', edgecolor='none', zorder=2)
            ax.add_patch(mat)
            # Turnstile flippers with LED green indicator bars
            for st_x in np.linspace(15.2, 22.8, 5):
                ax.plot([st_x, st_x], [gy - 0.5 if gdir == "S" else gy - 0.9, gy + 0.5 if gdir == "S" else gy + 0.1], color='#22C55E', linewidth=3.5, zorder=9)
                ax.plot(st_x, gy, marker='o', markersize=3.0, color='#FFFFFF', zorder=10)
            # Clean Gateway Tag
            ax.text(19.0, -1.6 if gdir == "S" else 39.4, f"[{gw['id']}] {gw['name']} (4-LANE SPEED GATES)", ha='center', va='center', fontsize=8.5, fontweight='heavy', family='monospace', color='#22C55E', zorder=10)
        else:
            # Vertical portal
            gate_box = patches.Rectangle((gx - 0.7 if gdir == "W" else gx - 0.7, 14.0), 1.4, 10.0, facecolor='#0F172A', edgecolor='#00F0FF', linewidth=2.0, zorder=8)
            ax.add_patch(gate_box)
            # Threshold matting
            mat = patches.Rectangle((0.0 if gdir == "W" else 36.6, 14.0), 1.4, 10.0, facecolor='#1E293B', edgecolor='none', zorder=2)
            ax.add_patch(mat)
            # Turnstile flippers with LED green indicator bars
            for st_y in np.linspace(15.2, 22.8, 5):
                ax.plot([gx - 0.5 if gdir == "W" else gx - 0.9, gx + 0.5 if gdir == "W" else gx + 0.1], [st_y, st_y], color='#22C55E', linewidth=3.5, zorder=9)
                ax.plot(gx, st_y, marker='o', markersize=3.0, color='#FFFFFF', zorder=10)
            # Clean Gateway Tag
            ax.text(-2.0 if gdir == "W" else 40.0, 19.0, f"[{gw['id']}] " + gw['name'], ha='center', va='center', fontsize=7.5, fontweight='heavy', family='monospace', color='#22C55E', rotation=90 if gdir == "W" else -90, zorder=10)

    # 7. CCTV Mount Icons (CLEAN ARCHITECTURE: Sleek Turret Mount Icons ONLY - NO FOV Projections)
    for gw in gateways_data:
        cx, cy = gw["cam_pos"]
        cid = gw["id"].replace("GW-", "CAM-")
        
        # Sleek tactical camera turret marker
        ax.plot(cx, cy, marker='o', markersize=9, color='#0284C7', markeredgecolor='#00F0FF', markeredgewidth=1.8, zorder=12)
        ax.plot(cx, cy, marker='.', markersize=4, color='#FFFFFF', zorder=13)
        
        # Clean camera mount tag (matching clean base style, styled for dark theme)
        offset_y = 1.3 if gw["dir"] == "S" else (-1.3 if gw["dir"] == "N" else 0.0)
        offset_x = 1.6 if gw["dir"] == "W" else (-1.6 if gw["dir"] == "E" else 0.0)
        ax.text(cx + offset_x, cy + offset_y, f"{cid}\n(4.6m Mount)",
                ha='center', va='center', fontsize=7.0, fontweight='bold', family='monospace', color='#00F0FF',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0B132B', edgecolor='#00F0FF', lw=1.2), zorder=14)

    # 8. Solid Structural Walls (Dark body with cyber cyan illuminated edge)
    for (p1, p2) in wall_segments:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#1E293B', linewidth=5.5, solid_capstyle='projecting', zorder=10)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#38BDF8', linewidth=1.8, solid_capstyle='projecting', zorder=11)

    # Outer Facility Perimeter
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], color='#475569', linewidth=1.5, linestyle='-', zorder=9)

    # Corner Crosshair Brackets (Subtle tactical minimap styling)
    bracket_len = 2.5
    for cx_b, cy_b in [(0, 0), (W, 0), (0, H), (W, H)]:
        bx_dir = 1 if cx_b == 0 else -1
        by_dir = 1 if cy_b == 0 else -1
        ax.plot([cx_b, cx_b + bx_dir * bracket_len], [cy_b, cy_b], color='#00F0FF', lw=2.0, zorder=12)
        ax.plot([cx_b, cx_b], [cy_b, cy_b + by_dir * bracket_len], color='#00F0FF', lw=2.0, zorder=12)

    # 9. Architectural Scale Bar (10m in Cyan)
    sb_x, sb_y = 3.0, -2.6
    ax.plot([sb_x, sb_x + 10.0], [sb_y, sb_y], color='#00F0FF', linewidth=3.0, zorder=15)
    ax.plot([sb_x, sb_x], [sb_y - 0.3, sb_y + 0.3], color='#00F0FF', linewidth=2.0, zorder=15)
    ax.plot([sb_x + 5.0, sb_x + 5.0], [sb_y - 0.2, sb_y + 0.2], color='#00F0FF', linewidth=1.5, zorder=15)
    ax.plot([sb_x + 10.0, sb_x + 10.0], [sb_y - 0.3, sb_y + 0.3], color='#00F0FF', linewidth=2.0, zorder=15)
    ax.text(sb_x + 5.0, sb_y - 0.7, "SCALE: 10 METERS // TACTICAL GRID (1:100 CAD)", ha='center', va='top', fontsize=8.0, family='monospace', fontweight='bold', color='#00F0FF')

    # 10. Tactical North Arrow
    ax.annotate('N', xy=(W - 2.5, H + 2.4), xytext=(W - 2.5, H + 1.2),
                ha='center', va='bottom', fontsize=12, fontweight='heavy', family='monospace', color='#00F0FF',
                arrowprops=dict(facecolor='#00F0FF', edgecolor='#00F0FF', width=2.5, headwidth=8.0, headlength=10.0), zorder=15)

    # 11. Tactical Title Block (Clean Monospace)
    tb_w, tb_h = 17.5, 2.6
    tb_x, tb_y = W - tb_w, -3.7
    tb_rect = patches.Rectangle((tb_x, tb_y), tb_w, tb_h, facecolor='#0F172A', edgecolor='#00F0FF', linewidth=1.5, zorder=14)
    ax.add_patch(tb_rect)
    ax.text(tb_x + 0.4, tb_y + tb_h/2.0,
            "CROWDSAFE SENTINEL — TACTICAL ARCHITECTURAL 2D PLAN\n"
            "DRAWING: LEVEL 01 TACTICAL CONCOURSE (CLEAN ARCHITECTURE)\n"
            f"VENUE: CONTROLLED TRANSIT INTERCHANGE ({W:.0f}m x {H:.0f}m)\n"
            "STYLING: DARK-MODE COMMAND CENTER // NO CAM CONE CLUTTER",
            ha='left', va='center', fontsize=7.0, fontweight='bold', family='monospace', color='#E2E8F0', zorder=15)

    # Title
    ax.set_title("CROWDSAFE — TRANSIT INTERCHANGE TACTICAL GAME MAP\n[CLEAN ARCHITECTURE // DARK COMMAND CENTER BASE]",
                 fontsize=13, fontweight='heavy', family='monospace', pad=18, color='#00F0FF')
    ax.set_xlabel("X Dimension (Meters)", fontsize=9.5, family='monospace', fontweight='bold', color='#64748B')
    ax.set_ylabel("Y Dimension (Meters)", fontsize=9.5, family='monospace', fontweight='bold', color='#64748B')

    plt.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    if output_svg:
        fig.savefig(output_svg, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Generated Clean Tactical Game Map Floor Plan: {output_png}")


def main():
    print("=" * 75)
    print("Generating 3 Specialized Floor Plan Variants (Design 4)")
    print("=" * 75)

    # 1. Clean Base Floor Plan
    p1_png = os.path.join(OUTPUT_DIR, "design_4_clean_base.png")
    p1_svg = os.path.join(OUTPUT_DIR, "design_4_clean_base.svg")
    draw_clean_base_plan(p1_png, p1_svg)

    # 2. Human Location Dots Floor Plan
    p2_png = os.path.join(OUTPUT_DIR, "design_4_human_locations.png")
    p2_svg = os.path.join(OUTPUT_DIR, "design_4_human_locations.svg")
    draw_human_locations_plan(p2_png, p2_svg)

    # 3. Tactical Game Map Floor Plan
    p3_png = os.path.join(OUTPUT_DIR, "design_4_tactical_gamemap.png")
    p3_svg = os.path.join(OUTPUT_DIR, "design_4_tactical_gamemap.svg")
    draw_tactical_gamemap_plan(p3_png, p3_svg)

    print("=" * 75)
    print("[✓] All 3 specialized floor plan variants generated successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
