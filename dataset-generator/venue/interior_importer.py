"""
CrowdSafe Synthetic Simulation — External Open-Source Venue Importer for Blender 4.2 LTS
Imports external open-source 3D interior mall venue models (.blend, .gltf, .glb, .obj):
1. Loads external 3D geometry and materials into Blender.
2. Centers and scales bounding dimensions to fit venue scale (e.g. 50m x 30m).
3. Merges and binds the calibrated virtual CCTV camera array from mall_floor_plan.json.
4. Injects overhead area lighting and EEVEE Next rendering configuration.
5. Saves the ready-to-simulate scene to mall_scene.blend.
"""

import sys
import os
import json
import math
import argparse
from typing import Dict, Any, List

import bpy
from mathutils import Vector, Euler, Matrix


def import_external_model(filepath: str) -> List[bpy.types.Object]:
    """Imports 3D model depending on extension and returns imported objects."""
    ext = os.path.splitext(filepath)[1].lower()
    before_objs = set(bpy.data.objects)

    if ext in [".gltf", ".glb"]:
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=filepath)
        else:
            bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == ".blend":
        # Link or append collection/objects from .blend
        with bpy.data.libraries.load(filepath) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            if obj:
                bpy.context.scene.collection.objects.link(obj)
    else:
        raise ValueError(f"Unsupported 3D model format '{ext}'. Supported formats: .gltf, .glb, .obj, .blend")

    after_objs = set(bpy.data.objects)
    imported = list(after_objs - before_objs)
    print(f"[✓] Successfully imported {len(imported)} objects from '{filepath}'")
    return imported


def fit_model_to_dimensions(objects: List[bpy.types.Object], target_w: float = 50.0, target_d: float = 30.0):
    """Calculates aggregate bounding box and centers/scales objects to fit the target footprint."""
    if not objects:
        return

    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in objects:
        if obj.type == 'MESH':
            for corner in [obj.matrix_world @ Vector(c) for c in obj.bound_box]:
                min_corner.x = min(min_corner.x, corner.x)
                min_corner.y = min(min_corner.y, corner.y)
                min_corner.z = min(min_corner.z, corner.z)
                max_corner.x = max(max_corner.x, corner.x)
                max_corner.y = max(max_corner.y, corner.y)
                max_corner.z = max(max_corner.z, corner.z)

    if math.isinf(min_corner.x):
        return

    cur_w = max(0.1, max_corner.x - min_corner.x)
    cur_d = max(0.1, max_corner.y - min_corner.y)

    scale_factor = min(target_w / cur_w, target_d / cur_d)
    center_offset = Vector(((max_corner.x + min_corner.x) / 2.0, (max_corner.y + min_corner.y) / 2.0, min_corner.z))
    target_center = Vector((target_w / 2.0, target_d / 2.0, 0.0))

    # Create root anchor for scaling and translation
    anchor = bpy.data.objects.new("External_Mall_Anchor", None)
    bpy.context.scene.collection.objects.link(anchor)

    for obj in objects:
        if obj.parent is None:
            obj.parent = anchor

    anchor.scale = (scale_factor, scale_factor, scale_factor)
    anchor.location = target_center - (center_offset * scale_factor)
    print(f"[✓] Scaled external model by {scale_factor:.3f} and aligned to venue center.")


def inject_cctv_and_lights(floor_plan: Dict[str, Any]):
    """Instantiates the calibrated CCTV camera array and lighting into the scene."""
    # Cameras
    cam_col = bpy.data.collections.new("Cameras")
    bpy.context.scene.collection.children.link(cam_col)

    first_cam = None
    for c_data in floor_plan.get("cameras", []):
        cid = c_data["camera_id"]
        pos = Vector(c_data["position"])
        look_at = Vector(c_data["look_at"])

        cam_data = bpy.data.cameras.new(name=f"Data_{cid}")
        cam_data.lens = c_data.get("focal_length_mm", 6.0)
        cam_data.sensor_width = c_data.get("sensor_width_mm", 6.17)
        cam_data.sensor_height = c_data.get("sensor_height_mm", 4.55)
        cam_data.sensor_fit = 'HORIZONTAL'

        cam_obj = bpy.data.objects.new(name=cid, object_data=cam_data)
        cam_obj.location = pos
        direction = look_at - pos
        cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        cam_col.objects.link(cam_obj)

        if first_cam is None:
            first_cam = cam_obj

    if first_cam:
        bpy.context.scene.camera = first_cam

    # Lights
    light_col = bpy.data.collections.new("Lights")
    bpy.context.scene.collection.children.link(light_col)
    for l_data in floor_plan.get("lights", []):
        name = l_data["name"]
        pos = l_data["position"]
        power = l_data.get("power", 450.0)
        size = l_data.get("size", [4.0, 2.0])
        color = l_data.get("color", [1.0, 1.0, 1.0])

        light_data = bpy.data.lights.new(name=f"Data_{name}", type='AREA')
        light_data.energy = power
        light_data.size = size[0]
        light_data.size_y = size[1]
        light_data.shape = 'RECTANGLE'
        light_data.color = tuple(color[:3])

        light_obj = bpy.data.objects.new(name=f"Light_{name}", object_data=light_data)
        light_obj.location = tuple(pos)
        light_col.objects.link(light_obj)


def import_venue_and_save(model_path: str, floor_plan_path: str, output_blend: str):
    """Orchestrates importing an external venue model, calibrating cameras, and saving."""
    with open(floor_plan_path, "r") as f:
        floor_plan = json.load(f)

    # Reset scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for col in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(col):
            col.remove(item)

    # 1. Import external geometry
    imported_objs = import_external_model(model_path)

    # 2. Fit to 50m x 30m floor plan dimensions
    dims = floor_plan.get("dimensions", {"width_x": 50.0, "depth_y": 30.0})
    fit_model_to_dimensions(imported_objs, dims["width_x"], dims["depth_y"])

    # 3. Inject CCTV cameras and lighting
    inject_cctv_and_lights(floor_plan)

    # 4. EEVEE configuration
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.eevee.taa_render_samples = 4

    os.makedirs(os.path.dirname(os.path.abspath(output_blend)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_blend)
    print(f"[✓] External 3D Venue integrated and saved to: {output_blend}")


def main():
    argv = sys.argv
    if "--" in argv:
        script_args = argv[argv.index("--") + 1:]
    else:
        script_args = []

    parser = argparse.ArgumentParser(description="Import external open-source 3D mall model into Blender.")
    parser.add_argument("--model", required=True, help="Path to external .blend, .gltf, .glb, or .obj file")
    parser.add_argument("--floor-plan", default="dataset-generator/venue/mall_floor_plan.json")
    parser.add_argument("--output", default="dataset-generator/venue/mall_scene.blend")
    args = parser.parse_args(script_args)

    import_venue_and_save(args.model, args.floor_plan, args.output)


if __name__ == "__main__":
    main()
