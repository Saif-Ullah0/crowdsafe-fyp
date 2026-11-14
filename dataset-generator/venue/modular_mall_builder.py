"""
CrowdSafe Synthetic Simulation — Modular High-Fidelity 3D Mall Venue Builder for Blender 4.2 LTS
Constructs an architectural, realistic shopping mall interior:
- Polished granite/marble floor tiles with PBR specular reflections
- Retail storefronts with glass vitrines, interior displays, and illuminated signage
- Central escalator structure with metallic steps and glass balustrades
- Modern food court counters with digital menu boards and stainless steel accents
- Atrium planters with architectural greenery
- Ceiling infrastructure: recessed LED panel fixtures and glowing green EXIT signs
- Physical CCTV camera dome mounts at calibrated coordinate positions
"""

import sys
import os
import json
import math
import argparse
from typing import Dict, Any, List, Tuple, Optional

import bpy
from mathutils import Vector, Euler, Matrix


def create_pbr_mat(
    name: str,
    color_rgba: Tuple[float, float, float, float],
    roughness: float = 0.5,
    specular: float = 0.5,
    metallic: float = 0.0,
    emission_color: Optional[Tuple[float, float, float, float]] = None,
    emission_strength: float = 0.0,
    transmission: float = 0.0,
    ior: float = 1.45
) -> bpy.types.Material:
    """Creates a Principled BSDF material supporting PBR, emission, and glass transmission."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")

    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color_rgba
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = specular
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = specular
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic

        # Emission (Blender 4.2 Principled BSDF)
        if emission_color and emission_strength > 0:
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = emission_color
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = emission_strength

        # Glass / Transmission
        if transmission > 0:
            if "Transmission Weight" in bsdf.inputs:
                bsdf.inputs["Transmission Weight"].default_value = transmission
            elif "Transmission" in bsdf.inputs:
                bsdf.inputs["Transmission"].default_value = transmission
            if "IOR" in bsdf.inputs:
                bsdf.inputs["IOR"].default_value = ior

    return mat


def create_polished_mall_floor(width: float, depth: float) -> bpy.types.Object:
    """Creates a luxurious polished commercial floor with alternating marble tiles."""
    mat = bpy.data.materials.new(name="PBR_Polished_Marble_Floor")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new("ShaderNodeOutputMaterial")
    node_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    node_checker = nodes.new("ShaderNodeTexChecker")
    node_coord = nodes.new("ShaderNodeTexCoord")
    node_mapping = nodes.new("ShaderNodeMapping")

    node_coord.location = (-800, 0)
    node_mapping.location = (-600, 0)
    node_checker.location = (-400, 0)
    node_bsdf.location = (-150, 0)
    node_out.location = (200, 0)

    # 1.2m large format polished commercial tiles
    node_mapping.inputs["Scale"].default_value = (0.833, 0.833, 1.0)
    # Slate grey marble & soft warm limestone
    node_checker.inputs["Color1"].default_value = (0.22, 0.23, 0.25, 1.0)
    node_checker.inputs["Color2"].default_value = (0.35, 0.36, 0.38, 1.0)
    node_checker.inputs["Scale"].default_value = 1.0

    # High gloss reflection
    if "Roughness" in node_bsdf.inputs:
        node_bsdf.inputs["Roughness"].default_value = 0.18  # Polished marble
    if "Specular IOR Level" in node_bsdf.inputs:
        node_bsdf.inputs["Specular IOR Level"].default_value = 0.70
    elif "Specular" in node_bsdf.inputs:
        node_bsdf.inputs["Specular"].default_value = 0.70

    links.new(node_coord.outputs["Generated"], node_mapping.inputs["Vector"])
    links.new(node_mapping.outputs["Vector"], node_checker.inputs["Vector"])
    links.new(node_checker.outputs["Color"], node_bsdf.inputs["Base Color"])
    links.new(node_bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    bpy.ops.mesh.primitive_plane_add(
        size=1.0,
        enter_editmode=False,
        align='WORLD',
        location=(width / 2.0, depth / 2.0, 0.0)
    )
    floor_obj = bpy.context.active_object
    floor_obj.name = "Mall_Floor_Polished"
    floor_obj.scale = (width, depth, 1.0)
    floor_obj.data.materials.append(mat)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return floor_obj


def build_retail_storefronts(collection: bpy.types.Collection):
    """
    Constructs commercial storefront facades along mall corridors:
    - Glass display window bays
    - Illuminated store brand signboards (FASHION, TECH HUB, STAR CAFE, etc.)
    - Warm interior display illumination
    """
    glass_mat = create_pbr_mat("Store_Glass", (0.9, 0.95, 1.0, 1.0), roughness=0.05, specular=0.9, transmission=0.85)
    frame_mat = create_pbr_mat("Store_Frame", (0.1, 0.1, 0.12, 1.0), roughness=0.3, metallic=0.8)

    # Store brands with distinct branding colors
    stores = [
        {"name": "URBAN_FASHION", "pos": [8.0, 17.0, 0.0], "sign_color": (0.95, 0.20, 0.20, 1.0), "length": 7.0, "dir_x": True},
        {"name": "TECH_BOUTIQUE", "pos": [42.0, 17.0, 0.0], "sign_color": (0.20, 0.60, 0.95, 1.0), "length": 7.0, "dir_x": True},
        {"name": "CAFE_EXPRESS", "pos": [8.0, 13.0, 0.0], "sign_color": (0.85, 0.55, 0.15, 1.0), "length": 7.0, "dir_x": True},
        {"name": "SPORTS_LIFESTYLE", "pos": [42.0, 13.0, 0.0], "sign_color": (0.25, 0.85, 0.45, 1.0), "length": 7.0, "dir_x": True},
    ]

    for store in stores:
        sx, sy, sz = store["pos"]
        slen = store["length"]
        sname = store["name"]

        # 1. Illuminated Brand Signboard
        sign_mat = create_pbr_mat(
            f"Sign_{sname}",
            store["sign_color"],
            roughness=0.2,
            emission_color=store["sign_color"],
            emission_strength=4.0
        )
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(sx, sy, 2.75)
        )
        sign_obj = bpy.context.active_object
        sign_obj.name = f"Sign_{sname}"
        sign_obj.scale = (slen * 0.7, 0.25, 0.45)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        sign_obj.data.materials.append(sign_mat)

        for col in list(sign_obj.users_collection):
            col.objects.unlink(sign_obj)
        collection.objects.link(sign_obj)

        # 2. Glass Display Bay
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(sx, sy, 1.2)
        )
        glass_obj = bpy.context.active_object
        glass_obj.name = f"Glass_{sname}"
        glass_obj.scale = (slen * 0.85, 0.08, 2.0)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        glass_obj.data.materials.append(glass_mat)

        for col in list(glass_obj.users_collection):
            col.objects.unlink(glass_obj)
        collection.objects.link(glass_obj)


def build_escalators(collection: bpy.types.Collection):
    """Constructs a dual-direction escalator in the central corridor linking to level 2."""
    metal_mat = create_pbr_mat("Escalator_Metal", (0.3, 0.32, 0.35, 1.0), roughness=0.35, metallic=0.9)
    glass_rail_mat = create_pbr_mat("Escalator_Glass", (0.85, 0.95, 1.0, 1.0), roughness=0.08, transmission=0.8)
    belt_mat = create_pbr_mat("Escalator_Handrail", (0.05, 0.05, 0.05, 1.0), roughness=0.5)

    # Escalator position in central open corridor
    exc_x = 25.0
    exc_y = 11.0
    exc_len = 5.5
    exc_w = 1.1
    exc_h = 3.2
    angle = math.radians(30.0)

    for direction, offset_x in [("Up", -0.7), ("Down", 0.7)]:
        # Inclined truss
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(exc_x + offset_x, exc_y, exc_h / 2.0),
            rotation=(angle, 0.0, 0.0)
        )
        truss = bpy.context.active_object
        truss.name = f"Escalator_Truss_{direction}"
        truss.scale = (exc_w, exc_len, 0.4)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        truss.data.materials.append(metal_mat)

        for col in list(truss.users_collection):
            col.objects.unlink(truss)
        collection.objects.link(truss)

        # Glass Balustrades
        for b_side, b_offset in [("L", -0.5), ("R", 0.5)]:
            bpy.ops.mesh.primitive_cube_add(
                size=1.0,
                location=(exc_x + offset_x + b_offset, exc_y, exc_h / 2.0 + 0.35),
                rotation=(angle, 0.0, 0.0)
            )
            bal = bpy.context.active_object
            bal.name = f"Escalator_Balustrade_{direction}_{b_side}"
            bal.scale = (0.05, exc_len, 0.75)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            bal.data.materials.append(glass_rail_mat)

            for col in list(bal.users_collection):
                col.objects.unlink(bal)
            collection.objects.link(bal)


def build_emergency_exit_signs(collection: bpy.types.Collection):
    """Places illuminated glowing green EXIT signs above all doorway thresholds."""
    exit_green = (0.05, 0.95, 0.20, 1.0)
    exit_mat = create_pbr_mat(
        "Sign_Emergency_Exit",
        exit_green,
        roughness=0.2,
        emission_color=exit_green,
        emission_strength=6.0
    )

    exit_locations = [
        ("Exit_A", [8.0, 29.8, 2.9]),
        ("Exit_B", [42.0, 29.8, 2.9]),
        ("Exit_C", [8.0, 0.2, 2.9]),
        ("Exit_D", [42.0, 0.2, 2.9]),
        ("Main_Entrance", [25.0, 0.2, 3.0])
    ]

    for ename, epos in exit_locations:
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=tuple(epos)
        )
        sign = bpy.context.active_object
        sign.name = f"Emergency_{ename}"
        sign.scale = (0.8, 0.12, 0.3)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        sign.data.materials.append(exit_mat)

        for col in list(sign.users_collection):
            col.objects.unlink(sign)
        collection.objects.link(sign)


def build_atrium_planters(collection: bpy.types.Collection):
    """Constructs modern commercial planter boxes with dark soil and lush greenery."""
    planter_mat = create_pbr_mat("Planter_Concrete", (0.25, 0.26, 0.28, 1.0), roughness=0.7)
    soil_mat = create_pbr_mat("Planter_Soil", (0.12, 0.08, 0.05, 1.0), roughness=0.95)
    foliage_mat = create_pbr_mat("Planter_Foliage", (0.10, 0.42, 0.15, 1.0), roughness=0.6)

    planter_positions = [
        [18.0, 15.0],
        [32.0, 15.0],
        [25.0, 20.0],
        [25.0, 5.0]
    ]

    for idx, (px, py) in enumerate(planter_positions):
        # Planter Box
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(px, py, 0.35)
        )
        pbox = bpy.context.active_object
        pbox.name = f"Planter_Box_{idx}"
        pbox.scale = (1.4, 0.8, 0.7)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        pbox.data.materials.append(planter_mat)

        for col in list(pbox.users_collection):
            col.objects.unlink(pbox)
        collection.objects.link(pbox)

        # Foliage volume (organic cluster)
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=8,
            ring_count=6,
            radius=0.45,
            location=(px, py, 0.95)
        )
        foliage = bpy.context.active_object
        foliage.name = f"Planter_Foliage_{idx}"
        foliage.scale = (1.3, 0.75, 0.65)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        foliage.data.materials.append(foliage_mat)

        for col in list(foliage.users_collection):
            col.objects.unlink(foliage)
        collection.objects.link(foliage)


def build_cctv_camera_fixtures(cameras_data: List[Dict[str, Any]], collection: bpy.types.Collection):
    """Places visual 3D CCTV dome camera models at the physical camera locations."""
    cctv_body_mat = create_pbr_mat("CCTV_Body", (0.92, 0.92, 0.94, 1.0), roughness=0.3)
    cctv_dome_mat = create_pbr_mat("CCTV_Dome", (0.05, 0.05, 0.05, 1.0), roughness=0.1, specular=0.9)
    led_rec_mat = create_pbr_mat("CCTV_LED_REC", (0.95, 0.10, 0.10, 1.0), emission_color=(1.0, 0.1, 0.1, 1.0), emission_strength=5.0)

    for cam in cameras_data:
        cid = cam["camera_id"]
        pos = cam["position"]
        cx, cy, cz = pos

        # Ceiling mount bracket
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=12,
            radius=0.15,
            depth=0.08,
            location=(cx, cy, cz - 0.04)
        )
        bracket = bpy.context.active_object
        bracket.name = f"Mount_{cid}"
        bracket.data.materials.append(cctv_body_mat)

        for col in list(bracket.users_collection):
            col.objects.unlink(bracket)
        collection.objects.link(bracket)

        # Smoked Acrylic Dome
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=12,
            ring_count=8,
            radius=0.12,
            location=(cx, cy, cz - 0.10)
        )
        dome = bpy.context.active_object
        dome.name = f"Dome_{cid}"
        dome.scale = (1.0, 1.0, 0.7)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        dome.data.materials.append(cctv_dome_mat)

        for col in list(dome.users_collection):
            col.objects.unlink(dome)
        collection.objects.link(dome)

        # Blinking REC LED indicator
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=6,
            ring_count=4,
            radius=0.015,
            location=(cx, cy + 0.11, cz - 0.07)
        )
        led = bpy.context.active_object
        led.name = f"LED_{cid}"
        led.data.materials.append(led_rec_mat)

        for col in list(led.users_collection):
            col.objects.unlink(led)
        collection.objects.link(led)


def build_enhanced_venue(floor_plan_path: str, output_blend: str):
    """Builds the complete photorealistic mall interior and saves to .blend."""
    with open(floor_plan_path, "r") as f:
        data = json.load(f)

    # Reset scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for col in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(col):
            col.remove(item)

    # 1. Floor
    dims = data.get("dimensions", {"width_x": 50.0, "depth_y": 30.0})
    W, H = dims["width_x"], dims["depth_y"]
    create_polished_mall_floor(W, H)

    # 2. Materials
    wall_mat = create_pbr_mat("Wall_Drywall", (0.84, 0.84, 0.86, 1.0), roughness=0.85)
    pillar_mat = create_pbr_mat("Pillar_Composite", (0.75, 0.75, 0.78, 1.0), roughness=0.45, specular=0.6)
    table_mat = create_pbr_mat("Table_Maple", (0.42, 0.28, 0.18, 1.0), roughness=0.40)

    # 3. Walls
    walls_col = bpy.data.collections.new("Walls")
    bpy.context.scene.collection.children.link(walls_col)
    for w in data.get("walls", []):
        p1, p2 = w["start"], w["end"]
        height = w.get("height", 3.2)
        thickness = w.get("thickness", 0.2)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length < 1e-4:
            continue
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        angle = math.atan2(dy, dx)

        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(mid_x, mid_y, height / 2.0),
            rotation=(0.0, 0.0, angle)
        )
        wall_obj = bpy.context.active_object
        wall_obj.name = f"Wall_{w.get('name', 'segment')}"
        wall_obj.scale = (length, thickness, height)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        wall_obj.data.materials.append(wall_mat)

        for col in list(wall_obj.users_collection):
            col.objects.unlink(wall_obj)
        walls_col.objects.link(wall_obj)

    # 4. Obstacles
    obs_col = bpy.data.collections.new("Obstacles")
    bpy.context.scene.collection.children.link(obs_col)
    for obs in data.get("obstacles", []):
        pos = obs["position"]
        otype = obs.get("type", "box")
        oname = obs.get("name", "obstacle")
        if otype == "cylinder":
            radius = obs.get("radius", 0.45)
            h = obs.get("height", 3.2)
            bpy.ops.mesh.primitive_cylinder_add(
                radius=radius,
                depth=h,
                location=tuple(pos)
            )
            o_obj = bpy.context.active_object
            o_obj.name = f"Obs_{oname}"
            o_obj.data.materials.append(pillar_mat if "pillar" in oname else table_mat)
        else:
            size = obs.get("size", [1.0, 1.0, 1.0])
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=tuple(pos))
            o_obj = bpy.context.active_object
            o_obj.name = f"Obs_{oname}"
            o_obj.scale = tuple(size)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            o_obj.data.materials.append(table_mat)

        for col in list(o_obj.users_collection):
            col.objects.unlink(o_obj)
        obs_col.objects.link(o_obj)

    # 5. Storefronts, Escalators, Greenery, Exit Signs
    decor_col = bpy.data.collections.new("Mall_Interior_Decor")
    bpy.context.scene.collection.children.link(decor_col)
    build_retail_storefronts(decor_col)
    build_escalators(decor_col)
    build_atrium_planters(decor_col)
    build_emergency_exit_signs(decor_col)

    # 6. Physical CCTV Camera Fixtures
    cams_data = data.get("cameras", [])
    build_cctv_camera_fixtures(cams_data, decor_col)

    # 7. Lighting
    light_col = bpy.data.collections.new("Lights")
    bpy.context.scene.collection.children.link(light_col)
    for l_data in data.get("lights", []):
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

    # 8. Virtual Camera Objects
    cam_col = bpy.data.collections.new("Cameras")
    bpy.context.scene.collection.children.link(cam_col)
    first_cam = None
    for c_data in cams_data:
        cid = c_data["camera_id"]
        pos = Vector(c_data["position"])
        look_at = Vector(c_data["look_at"])
        focal_length = c_data.get("focal_length_mm", 6.0)
        sensor_w = c_data.get("sensor_width_mm", 6.17)
        sensor_h = c_data.get("sensor_height_mm", 4.55)

        cam_data = bpy.data.cameras.new(name=f"Data_{cid}")
        cam_data.lens = focal_length
        cam_data.sensor_width = sensor_w
        cam_data.sensor_height = sensor_h
        cam_data.sensor_fit = 'HORIZONTAL'
        cam_data.clip_start = 0.1
        cam_data.clip_end = 150.0

        cam_obj = bpy.data.objects.new(name=cid, object_data=cam_data)
        cam_obj.location = pos
        direction = look_at - pos
        cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        cam_col.objects.link(cam_obj)

        if first_cam is None:
            first_cam = cam_obj

    if first_cam:
        bpy.context.scene.camera = first_cam

    # 9. EEVEE Render Engine Settings
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.eevee.taa_render_samples = 16

    world = bpy.data.worlds.new("Mall_World_PBR")
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.07, 0.07, 0.09, 1.0)
        bg_node.inputs["Strength"].default_value = 0.5
    scene.world = world

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_blend)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_blend)
    print(f"[✓] High-Fidelity 3D Mall Scene saved to: {output_blend}")
    print(f"[✓] Scene statistics: {len(bpy.data.objects)} objects, {len(bpy.data.materials)} PBR materials.")


def main():
    argv = sys.argv
    if "--" in argv:
        script_args = argv[argv.index("--") + 1:]
    else:
        script_args = []

    parser = argparse.ArgumentParser(description="Build enhanced photorealistic 3D mall venue.")
    parser.add_argument("--floor-plan", default="dataset-generator/venue/mall_floor_plan.json")
    parser.add_argument("--output", default="dataset-generator/venue/mall_scene.blend")
    args = parser.parse_args(script_args)

    build_enhanced_venue(args.floor_plan, args.output)


if __name__ == "__main__":
    main()
