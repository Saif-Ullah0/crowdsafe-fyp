"""
CrowdSafe Synthetic Simulation — Occlusion Tracker & Line-of-Sight Culler
Performs two-stage visibility filtering:
1. Camera frustum projection (world_to_camera_view)
2. Line-of-sight ray casting (scene.ray_cast) to detect occlusions by walls, pillars, or people
Calculates per-zone crowd density (people/m^2) and Fruin Level of Service risk categories.
"""

from typing import Dict, List, Tuple, Optional, Any
import math
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


def point_in_polygon(x: float, y: float, poly: List[List[float]]) -> bool:
    """Standard ray-casting algorithm to test if (x, y) is inside a 2D polygon."""
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def classify_fruin_level(density_per_m2: float) -> str:
    """Fruin Level of Service international safety classification."""
    if density_per_m2 < 2.0:
        return "SAFE"
    elif density_per_m2 < 3.5:
        return "CAUTION"
    elif density_per_m2 < 5.0:
        return "WARNING"
    else:
        return "CRITICAL_DANGER"


def track_camera_crowd(
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
    agent_models: Dict[int, Any],
    zones_data: Dict[str, Any],
    depsgraph: Optional[bpy.types.Depsgraph] = None
) -> Dict[str, Any]:
    """
    Evaluates visibility, ray-cast occlusions, head pixel coordinates,
    and zone densities for the given camera.
    """
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()

    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.0
    width_px = res_x * scale
    height_px = res_y * scale

    cam_origin = camera_obj.matrix_world.translation
    f_mm = camera_obj.data.lens
    sensor_w = camera_obj.data.sensor_width
    fx_px = (f_mm * width_px) / sensor_w

    visible_heads = []
    occluded_heads = []
    all_in_frustum = []

    # Initialize zone counts
    zone_stats = {}
    for zid, zinfo in zones_data.items():
        zone_stats[zid] = {
            "name": zinfo.get("name", zid),
            "area_m2": zinfo.get("area_m2", 1.0),
            "visible_count": 0,
            "total_count": 0,
            "density_per_m2": 0.0,
            "risk_level": "SAFE"
        }

    for aid, model in agent_models.items():
        # Get exact 3D world location of head sphere
        head_world = model.get_head_world_position()
        hx, hy, hz = head_world.x, head_world.y, head_world.z

        # Determine which zone the agent is standing in
        agent_zone_id = "unassigned"
        for zid, zinfo in zones_data.items():
            if point_in_polygon(hx, hy, zinfo["polygon"]):
                agent_zone_id = zid
                zone_stats[zid]["total_count"] += 1
                break

        # ---------------------------------------------------------------------
        # 1. Frustum Boundary Test
        # ---------------------------------------------------------------------
        co_2d = world_to_camera_view(scene, camera_obj, head_world)

        # In front of camera and within [0, 1] normalized view
        if not (0.0 <= co_2d.x <= 1.0 and 0.0 <= co_2d.y <= 1.0 and co_2d.z > 0.1):
            continue  # Outside camera FOV

        # Convert to pixel coordinates (Blender origin: bottom-left; Image: top-left)
        pixel_x = round(co_2d.x * width_px, 1)
        pixel_y = round((1.0 - co_2d.y) * height_px, 1)
        depth_m = round(co_2d.z, 2)

        # Perspective head bounding box estimation (assuming r=0.12m)
        r_head = 0.12 * model.scale_factor
        r_px = max(2.0, (fx_px * r_head) / max(0.1, depth_m))
        bbox = [
            round(max(0.0, pixel_x - r_px), 1),
            round(max(0.0, pixel_y - r_px), 1),
            round(min(width_px, r_px * 2.0), 1),
            round(min(height_px, r_px * 2.0), 1)
        ]

        # ---------------------------------------------------------------------
        # 2. Line-of-Sight Ray Casting (Occlusion Test)
        # ---------------------------------------------------------------------
        ray_vec = head_world - cam_origin
        target_dist = ray_vec.length

        # Cast ray from camera optical center toward head center
        is_hit, hit_loc, hit_norm, hit_idx, hit_obj, hit_mat = scene.ray_cast(
            depsgraph, cam_origin, ray_vec
        )

        is_occluded = False
        if is_hit:
            hit_dist = (hit_loc - cam_origin).length
            # If hit something before target head (tolerance: 0.22m to account for head sphere radius)
            if hit_dist < (target_dist - 0.22):
                # Verify hit object is not the agent's own head or cap
                if hit_obj != model.head_obj and hit_obj != model.accessory_obj:
                    is_occluded = True

        head_record = {
            "agent_id": aid,
            "x_pixel": pixel_x,
            "y_pixel": pixel_y,
            "x_world": round(hx, 3),
            "y_world": round(hy, 3),
            "z_world": round(hz, 3),
            "depth_m": depth_m,
            "bbox_xywh": bbox,
            "is_occluded": is_occluded,
            "zone_id": agent_zone_id
        }

        all_in_frustum.append(head_record)
        if is_occluded:
            occluded_heads.append(head_record)
        else:
            visible_heads.append(head_record)
            if agent_zone_id in zone_stats:
                zone_stats[agent_zone_id]["visible_count"] += 1

    # Calculate densities and risk levels per zone
    for zid, zdata in zone_stats.items():
        area = zdata["area_m2"]
        count = zdata["visible_count"]
        density = round(count / max(area, 1.0), 3)
        zdata["density_per_m2"] = density
        zdata["risk_level"] = classify_fruin_level(density)

    return {
        "camera_id": camera_obj.name,
        "image_width": int(width_px),
        "image_height": int(height_px),
        "total_visible_count": len(visible_heads),
        "total_occluded_count": len(occluded_heads),
        "frustum_count": len(all_in_frustum),
        "visible_heads": visible_heads,
        "occluded_heads": occluded_heads,
        "zone_densities": zone_stats
    }
