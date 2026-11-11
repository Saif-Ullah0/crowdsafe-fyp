"""
CrowdSafe Synthetic Simulation — 3D Venue Builder for Blender 4.2 LTS
Reads mall_floor_plan.json and constructs:
1. Ground floor with procedural tile shader
2. Perimeter and interior partition walls
3. Architectural obstacles (pillars, food court tables, kiosks, benches)
4. Overhead CCTV cameras (CAM-01 to CAM-04) with exact look-at tracking
5. Overhead diffuse area lighting
6. EEVEE render engine settings
Saves output to mall_scene.blend
"""

import sys
import os
import json
import math
import argparse

import bpy
from mathutils import Vector, Quaternion, Matrix


def parse_args():
    # Args after '--' are forwarded to script
    argv = sys.argv
    if "--" in argv:
        script_args = argv[argv.index("--") + 1:]
    else:
        script_args = []

    parser = argparse.ArgumentParser(description="Build 3D venue in Blender from floor plan JSON.")
    parser.add_argument(
        "--floor-plan",
        default="dataset-generator/venue/mall_floor_plan.json",
        help="Path to mall_floor_plan.json"
    )
    parser.add_argument(
        "--output",
        default="dataset-generator/venue/mall_scene.blend",
        help="Output .blend file path"
    )
    return parser.parse_args(script_args)


def clear_scene():
    """Removes all default objects, meshes, materials, lights, and cameras."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Remove any leftover data blocks
    for col in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(col):
            col.remove(item)


def create_material(name, color_rgba, roughness=0.5, metallic=0.0):
    """Creates a Principled BSDF material compatible with Blender 4.2+."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color_rgba
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
    return mat


def create_tile_material():
    """Creates a procedural checkered tile material for the floor."""
    mat = bpy.data.materials.new(name="Floor_Tile_Shader")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Shader Nodes
    node_out = nodes.new("ShaderNodeOutputMaterial")
    node_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    node_checker = nodes.new("ShaderNodeTexChecker")
    node_coord = nodes.new("ShaderNodeTexCoord")
    node_mapping = nodes.new("ShaderNodeMapping")

    # Positions
    node_coord.location = (-800, 0)
    node_mapping.location = (-600, 0)
    node_checker.location = (-400, 0)
    node_bsdf.location = (-150, 0)
    node_out.location = (200, 0)

    # Scale mapping to ~1m tile grid
    node_mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    node_checker.inputs["Color1"].default_value = (0.16, 0.16, 0.18, 1.0)  # Dark slate tile
    node_checker.inputs["Color2"].default_value = (0.24, 0.24, 0.26, 1.0)  # Subtle accent tile
    node_checker.inputs["Scale"].default_value = 1.0

    if "Roughness" in node_bsdf.inputs:
        node_bsdf.inputs["Roughness"].default_value = 0.35  # Polished commercial floor

    # Connect nodes
    links.new(node_coord.outputs["Generated"], node_mapping.inputs["Vector"])
    links.new(node_mapping.outputs["Vector"], node_checker.inputs["Vector"])
    links.new(node_checker.outputs["Color"], node_bsdf.inputs["Base Color"])
    links.new(node_bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    return mat


def build_floor(width, depth, floor_mat):
    """Creates the main floor plane."""
    bpy.ops.mesh.primitive_plane_add(
        size=1.0,
        enter_editmode=False,
        align='WORLD',
        location=(width / 2.0, depth / 2.0, 0.0)
    )
    floor_obj = bpy.context.active_object
    floor_obj.name = "Mall_Floor"
    floor_obj.scale = (width, depth, 1.0)
    floor_obj.data.materials.append(floor_mat)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return floor_obj


def build_walls(walls_data, wall_mat):
    """Constructs extruded 3D walls from line segments."""
    wall_collection = bpy.data.collections.new("Walls")
    bpy.context.scene.collection.children.link(wall_collection)

    for w in walls_data:
        p1 = w["start"]
        p2 = w["end"]
        height = w.get("height", 3.0)
        thickness = w.get("thickness", 0.2)
        name = w.get("name", "wall")

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length < 1e-4:
            continue

        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        mid_z = height / 2.0
        angle = math.atan2(dy, dx)

        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(mid_x, mid_y, mid_z),
            rotation=(0.0, 0.0, angle)
        )
        wall_obj = bpy.context.active_object
        wall_obj.name = f"Wall_{name}"
        wall_obj.scale = (length, thickness, height)
        wall_obj.data.materials.append(wall_mat)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Unlink from default scene collection and link to Walls collection
        for col in list(wall_obj.users_collection):
            col.objects.unlink(wall_obj)
        wall_collection.objects.link(wall_obj)


def build_obstacles(obstacles_data, concrete_mat, table_mat):
    """Constructs structural pillars, kiosks, and food court tables."""
    obs_collection = bpy.data.collections.new("Obstacles")
    bpy.context.scene.collection.children.link(obs_collection)

    for obs in obstacles_data:
        name = obs["name"]
        obs_type = obs.get("type", "box")
        pos = obs["position"]

        if obs_type == "cylinder":
            radius = obs.get("radius", 0.45)
            height = obs.get("height", 3.2)
            bpy.ops.mesh.primitive_cylinder_add(
                radius=radius,
                depth=height,
                location=(pos[0], pos[1], pos[2])
            )
            obj = bpy.context.active_object
            obj.name = f"Obs_{name}"
            # Apply appropriate material
            if "pillar" in name:
                obj.data.materials.append(concrete_mat)
            else:
                obj.data.materials.append(table_mat)

        elif obs_type == "box":
            size = obs.get("size", [1.0, 1.0, 1.0])
            bpy.ops.mesh.primitive_cube_add(
                size=1.0,
                location=(pos[0], pos[1], pos[2])
            )
            obj = bpy.context.active_object
            obj.name = f"Obs_{name}"
            obj.scale = tuple(size)
            obj.data.materials.append(table_mat)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Move to Obstacles collection
        for col in list(obj.users_collection):
            col.objects.unlink(obj)
        obs_collection.objects.link(obj)


def build_lights(lights_data):
    """Places diffuse overhead area lighting simulating mall fixtures."""
    light_collection = bpy.data.collections.new("Lights")
    bpy.context.scene.collection.children.link(light_collection)

    for l_data in lights_data:
        name = l_data["name"]
        pos = l_data["position"]
        power = l_data.get("power", 400.0)
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
        # Area lights shine along -Z by default, pointing directly down
        light_obj.rotation_euler = (0.0, 0.0, 0.0)
        light_collection.objects.link(light_obj)


def build_cameras(cameras_data):
    """Creates virtual CCTV cameras with wide-angle lens and look-at tracking."""
    cam_collection = bpy.data.collections.new("Cameras")
    bpy.context.scene.collection.children.link(cam_collection)

    first_cam = None
    for c_data in cameras_data:
        cam_id = c_data["camera_id"]
        pos = Vector(c_data["position"])
        look_at = Vector(c_data["look_at"])
        focal_length = c_data.get("focal_length_mm", 6.0)
        sensor_w = c_data.get("sensor_width_mm", 6.17)
        sensor_h = c_data.get("sensor_height_mm", 4.55)

        cam_data = bpy.data.cameras.new(name=f"Data_{cam_id}")
        cam_data.lens = focal_length
        cam_data.sensor_width = sensor_w
        cam_data.sensor_height = sensor_h
        cam_data.sensor_fit = 'HORIZONTAL'
        cam_data.clip_start = 0.1
        cam_data.clip_end = 150.0

        cam_obj = bpy.data.objects.new(name=cam_id, object_data=cam_data)
        cam_obj.location = pos

        # Calculate rotation: camera points along -Z, with +Y as up
        direction = look_at - pos
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()

        cam_collection.objects.link(cam_obj)
        if first_cam is None:
            first_cam = cam_obj

    if first_cam:
        bpy.context.scene.camera = first_cam


def configure_scene_render():
    """Configures high-speed EEVEE rendering parameters."""
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.compression = 15

    # 4 samples for rapid headless rendering
    scene.eevee.taa_render_samples = 4
    
    # World background: neutral warm ambient
    world = bpy.data.worlds.new("Mall_World")
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.08, 0.08, 0.10, 1.0)
        bg_node.inputs["Strength"].default_value = 0.4
    scene.world = world


def main():
    args = parse_args()
    print("==================================================")
    print("Building 3D Venue from:", args.floor_plan)
    print("Target .blend output:", args.output)
    print("==================================================")

    if not os.path.exists(args.floor_plan):
        raise FileNotFoundError(f"Floor plan JSON not found: {args.floor_plan}")

    with open(args.floor_plan, "r") as f:
        floor_plan = json.load(f)

    # 1. Reset scene
    clear_scene()

    # 2. Materials
    floor_mat = create_tile_material()
    wall_mat = create_material("Wall_Concrete", (0.78, 0.78, 0.80, 1.0), roughness=0.85)
    pillar_mat = create_material("Pillar_Stone", (0.65, 0.65, 0.68, 1.0), roughness=0.70)
    table_mat = create_material("Furniture_Laminate", (0.35, 0.25, 0.20, 1.0), roughness=0.50)

    # 3. Geometry
    dims = floor_plan.get("dimensions", {"width_x": 50.0, "depth_y": 30.0})
    build_floor(dims["width_x"], dims["depth_y"], floor_mat)
    build_walls(floor_plan.get("walls", []), wall_mat)
    build_obstacles(floor_plan.get("obstacles", []), pillar_mat, table_mat)

    # 4. Lighting & Cameras
    build_lights(floor_plan.get("lights", []))
    build_cameras(floor_plan.get("cameras", []))

    # 5. Render settings
    configure_scene_render()

    # 6. Save .blend file
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print(f">>> Successfully generated 3D venue scene: {args.output}")
    print(f">>> Objects in scene: {len(bpy.data.objects)}")
    print(f">>> Cameras placed: {len(bpy.data.cameras)}")
    print(f">>> Lights placed: {len(bpy.data.lights)}")
    print("==================================================")


if __name__ == "__main__":
    main()
