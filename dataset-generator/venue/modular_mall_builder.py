"""
CrowdSafe Synthetic Simulation — Master High-Fidelity 3D Mall Venue Builder for Blender 4.2 LTS
Builds a realistic, modern architectural shopping mall venue:
- Polished Carrara marble & dark Nero Marquina floor tiles with PBR specular reflections
- 8 Retail storefronts with floor-to-ceiling glass vitrines, interior store lighting, and illuminated brand signboards (Apple, Zara, Nike, Starbucks, Sephora, etc.)
- Central atrium escalator with glowing under-step lighting and glass balustrades
- Modern food court counters and concourse seating benches
- Architectural structural columns with brushed stainless steel kick-plates
- Ceiling grid with recessed linear LED lighting panels casting realistic downward light
- Indoor atrium planters with lush greenery
- Physical dome CCTV camera assemblies with authentic mounting brackets and smoked acrylic domes
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


def create_luxury_marble_floor(width: float, depth: float) -> bpy.types.Object:
    """Creates a polished Italian Crema Marfil marble floor with subtle stone veining and satin reflections."""
    mat = bpy.data.materials.new(name="PBR_Luxury_Marble_Floor")
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

    # 1.5m large format commercial marble tiles
    node_mapping.inputs["Scale"].default_value = (0.667, 0.667, 1.0)
    # Refined Italian Crema Marfil (warm light ivory marble) & Statuario Venato (subtle pearl stone)
    node_checker.inputs["Color1"].default_value = (0.88, 0.86, 0.82, 1.0)
    node_checker.inputs["Color2"].default_value = (0.78, 0.76, 0.72, 1.0)
    node_checker.inputs["Scale"].default_value = 1.0

    # High-end commercial satin polish with soft specular highlights (no harsh dark reflections)
    if "Roughness" in node_bsdf.inputs:
        node_bsdf.inputs["Roughness"].default_value = 0.22
    if "Specular IOR Level" in node_bsdf.inputs:
        node_bsdf.inputs["Specular IOR Level"].default_value = 0.65
    elif "Specular" in node_bsdf.inputs:
        node_bsdf.inputs["Specular"].default_value = 0.65

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
    Constructs 8 realistic retail storefronts along concourse corridors:
    - Floor-to-ceiling tempered glass vitrines
    - High-contrast glowing brand signboards (Apple, Zara, Nike, Starbucks, Sephora, etc.)
    - Warm interior showroom lighting behind the glass
    - Display pedestals and merchandise silhouettes inside
    """
    glass_mat = create_pbr_mat("Store_Glass", (0.92, 0.96, 1.0, 1.0), roughness=0.03, specular=0.95, transmission=0.90)
    frame_mat = create_pbr_mat("Store_Frame", (0.08, 0.08, 0.10, 1.0), roughness=0.25, metallic=0.85)
    interior_wall_mat = create_pbr_mat("Store_Interior_Wall", (0.85, 0.83, 0.80, 1.0), roughness=0.80)
    pedestal_mat = create_pbr_mat("Display_Pedestal", (0.15, 0.15, 0.15, 1.0), roughness=0.30)
    mannequin_mat = create_pbr_mat("Display_Mannequin", (0.95, 0.95, 0.95, 1.0), roughness=0.40)

    stores = [
        # North concourse stores (along North corridor wall y=17.0, flanking West & East corridor openings)
        {"name": "APPLE_STORE", "pos": [3.5, 17.0, 0.0], "sign_color": (0.98, 0.98, 1.0, 1.0), "length": 6.4, "interior_y": 2.2, "light_color": (1.0, 0.95, 0.90)},
        {"name": "ZARA_FASHION", "pos": [12.5, 17.0, 0.0], "sign_color": (0.95, 0.82, 0.50, 1.0), "length": 6.4, "interior_y": 2.2, "light_color": (1.0, 0.88, 0.70)},
        {"name": "NIKE_SPORTS", "pos": [37.5, 17.0, 0.0], "sign_color": (1.0, 0.25, 0.05, 1.0), "length": 6.4, "interior_y": 2.2, "light_color": (1.0, 0.92, 0.85)},
        {"name": "SEPHORA", "pos": [46.5, 17.0, 0.0], "sign_color": (0.95, 0.10, 0.55, 1.0), "length": 6.4, "interior_y": 2.2, "light_color": (1.0, 0.85, 0.95)},
        # South concourse stores (along South corridor wall y=13.0, flanking West & East corridor openings)
        {"name": "STARBUCKS", "pos": [3.5, 13.0, 0.0], "sign_color": (0.05, 0.65, 0.25, 1.0), "length": 6.4, "interior_y": -2.2, "light_color": (1.0, 0.80, 0.55)},
        {"name": "SONY_TECH", "pos": [12.5, 13.0, 0.0], "sign_color": (0.10, 0.60, 1.0, 1.0), "length": 6.4, "interior_y": -2.2, "light_color": (0.85, 0.95, 1.0)},
        {"name": "RAY_BAN", "pos": [37.5, 13.0, 0.0], "sign_color": (0.95, 0.15, 0.15, 1.0), "length": 6.4, "interior_y": -2.2, "light_color": (1.0, 0.92, 0.80)},
        {"name": "HM_APPAREL", "pos": [46.5, 13.0, 0.0], "sign_color": (0.88, 0.10, 0.15, 1.0), "length": 6.4, "interior_y": -2.2, "light_color": (1.0, 0.90, 0.80)},
    ]

    for store in stores:
        sx, sy, sz = store["pos"]
        slen = store["length"]
        sname = store["name"]
        int_y = store["interior_y"]

        # 1. Illuminated Brand Signboard
        sign_mat = create_pbr_mat(
            f"Sign_{sname}",
            store["sign_color"],
            roughness=0.15,
            emission_color=store["sign_color"],
            emission_strength=5.5
        )
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(sx, sy, 2.85)
        )
        sign_obj = bpy.context.active_object
        sign_obj.name = f"Sign_{sname}"
        sign_obj.scale = (slen * 0.75, 0.22, 0.42)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        sign_obj.data.materials.append(sign_mat)
        for col in list(sign_obj.users_collection):
            col.objects.unlink(sign_obj)
        collection.objects.link(sign_obj)

        # 2. Storefront Architectural Outer Frame (Transom header + base kick-rail + mullions)
        # Base kick-rail
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, sy, 0.075))
        base_obj = bpy.context.active_object
        base_obj.name = f"Frame_Base_{sname}"
        base_obj.scale = (slen, 0.12, 0.15)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        base_obj.data.materials.append(frame_mat)
        for col in list(base_obj.users_collection): col.objects.unlink(base_obj)
        collection.objects.link(base_obj)

        # Top transom header
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, sy, 2.70))
        top_obj = bpy.context.active_object
        top_obj.name = f"Frame_Top_{sname}"
        top_obj.scale = (slen, 0.12, 0.20)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        top_obj.data.materials.append(frame_mat)
        for col in list(top_obj.users_collection): col.objects.unlink(top_obj)
        collection.objects.link(top_obj)

        # Vertical side mullions
        for m_sign, m_label in [(-1, "L"), (1, "R")]:
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx + m_sign * (slen / 2.0 - 0.05), sy, 1.40))
            mul_obj = bpy.context.active_object
            mul_obj.name = f"Frame_Mullion_{sname}_{m_label}"
            mul_obj.scale = (0.10, 0.12, 2.80)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            mul_obj.data.materials.append(frame_mat)
            for col in list(mul_obj.users_collection): col.objects.unlink(mul_obj)
            collection.objects.link(mul_obj)

        # 3. Tempered Glass Vitrine (Crystal clear showroom window)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, sy, 1.40))
        glass_obj = bpy.context.active_object
        glass_obj.name = f"Glass_{sname}"
        glass_obj.scale = (slen - 0.20, 0.04, 2.45)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        glass_obj.data.materials.append(glass_mat)
        for col in list(glass_obj.users_collection):
            col.objects.unlink(glass_obj)
        collection.objects.link(glass_obj)

        # 4. Interior Showroom Back Wall & Side Dividing Walls
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, sy + int_y, 1.5))
        back_wall = bpy.context.active_object
        back_wall.name = f"BackWall_{sname}"
        back_wall.scale = (slen, 0.10, 3.0)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        back_wall.data.materials.append(interior_wall_mat)
        for col in list(back_wall.users_collection):
            col.objects.unlink(back_wall)
        collection.objects.link(back_wall)

        # Side dividing partition walls
        for side_sign, s_name in [(-1, "Left"), (1, "Right")]:
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx + side_sign * (slen / 2.0), sy + (int_y / 2.0), 1.5))
            sw = bpy.context.active_object
            sw.name = f"SideWall_{sname}_{s_name}"
            sw.scale = (0.10, abs(int_y), 3.0)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            sw.data.materials.append(interior_wall_mat)
            for col in list(sw.users_collection):
                col.objects.unlink(sw)
            collection.objects.link(sw)

        # Illuminated Showroom Display Shelving
        shelf_mat = create_pbr_mat(
            f"Shelf_{sname}",
            store["sign_color"],
            roughness=0.25,
            emission_color=store["sign_color"],
            emission_strength=2.2
        )
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, sy + int_y - (0.05 if int_y > 0 else -0.05), 1.5))
        shelf = bpy.context.active_object
        shelf.name = f"DisplayShelf_{sname}"
        shelf.scale = (slen * 0.70, 0.08, 1.8)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        shelf.data.materials.append(shelf_mat)
        for col in list(shelf.users_collection):
            col.objects.unlink(shelf)
        collection.objects.link(shelf)

        # 5. Display Pedestal & Mannequin inside the Vitrine
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, sy + (int_y * 0.4), 0.35))
        ped = bpy.context.active_object
        ped.name = f"Pedestal_{sname}"
        ped.scale = (1.2, 0.6, 0.7)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        ped.data.materials.append(pedestal_mat)
        for col in list(ped.users_collection):
            col.objects.unlink(ped)
        collection.objects.link(ped)

        # Mannequin silhouette on pedestal
        bpy.ops.mesh.primitive_cylinder_add(radius=0.18, depth=1.3, location=(sx, sy + (int_y * 0.4), 1.35))
        man = bpy.context.active_object
        man.name = f"Mannequin_{sname}"
        man.data.materials.append(mannequin_mat)
        for col in list(man.users_collection):
            col.objects.unlink(man)
        collection.objects.link(man)

        # 6. Warm Interior Display Spot/Area Light
        light_data = bpy.data.lights.new(name=f"Light_Store_{sname}", type='SPOT')
        light_data.energy = 650.0
        light_data.spot_size = math.radians(80)
        light_data.spot_blend = 0.5
        light_data.color = tuple(store["light_color"])
        light_obj = bpy.data.objects.new(f"Light_Store_{sname}", light_data)
        light_obj.location = (sx, sy + (int_y * 0.5), 2.9)
        light_obj.rotation_euler = (0.0, 0.0, 0.0)
        collection.objects.link(light_obj)


def build_escalators(collection: bpy.types.Collection):
    """Constructs dual escalators in central atrium with glowing under-step lighting and glass balustrades."""
    metal_mat = create_pbr_mat("Escalator_Metal", (0.32, 0.34, 0.38, 1.0), roughness=0.25, metallic=0.92)
    glass_rail_mat = create_pbr_mat("Escalator_Glass", (0.88, 0.96, 1.0, 1.0), roughness=0.04, transmission=0.88)
    handrail_mat = create_pbr_mat("Escalator_Handrail", (0.04, 0.04, 0.04, 1.0), roughness=0.45)
    step_glow_mat = create_pbr_mat("Escalator_UnderStep_Glow", (0.2, 0.6, 1.0, 1.0), emission_color=(0.2, 0.6, 1.0, 1.0), emission_strength=4.0)

    exc_x = 25.0
    exc_y = 11.0
    exc_len = 5.5
    exc_w = 1.1
    exc_h = 3.2
    angle = math.radians(30.0)

    for direction, offset_x in [("Up", -0.75), ("Down", 0.75)]:
        # Inclined metal truss
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

        # Under-step LED neon strip
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(exc_x + offset_x, exc_y, exc_h / 2.0 + 0.18),
            rotation=(angle, 0.0, 0.0)
        )
        glow = bpy.context.active_object
        glow.name = f"Escalator_Glow_{direction}"
        glow.scale = (exc_w * 0.85, exc_len * 0.95, 0.05)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        glow.data.materials.append(step_glow_mat)
        for col in list(glow.users_collection):
            col.objects.unlink(glow)
        collection.objects.link(glow)

        # Glass Balustrades
        for b_side, b_offset in [("L", -0.52), ("R", 0.52)]:
            bpy.ops.mesh.primitive_cube_add(
                size=1.0,
                location=(exc_x + offset_x + b_offset, exc_y, exc_h / 2.0 + 0.38),
                rotation=(angle, 0.0, 0.0)
            )
            bal = bpy.context.active_object
            bal.name = f"Escalator_Balustrade_{direction}_{b_side}"
            bal.scale = (0.04, exc_len, 0.78)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            bal.data.materials.append(glass_rail_mat)
            for col in list(bal.users_collection):
                col.objects.unlink(bal)
            collection.objects.link(bal)

            # Black rubber handrail on top of glass
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.035,
                depth=exc_len,
                location=(exc_x + offset_x + b_offset, exc_y, exc_h / 2.0 + 0.78),
                rotation=(angle + math.radians(90.0), 0.0, 0.0)
            )
            rail = bpy.context.active_object
            rail.name = f"Escalator_Handrail_{direction}_{b_side}"
            rail.data.materials.append(handrail_mat)
            for col in list(rail.users_collection):
                col.objects.unlink(rail)
            collection.objects.link(rail)


def build_concourse_benches_and_planters(collection: bpy.types.Collection):
    """Adds modern architectural seating benches and circular planters with greenery."""
    bench_wood = create_pbr_mat("Bench_Teak", (0.45, 0.28, 0.16, 1.0), roughness=0.35)
    bench_steel = create_pbr_mat("Bench_Steel", (0.85, 0.85, 0.88, 1.0), roughness=0.20, metallic=0.95)
    planter_brass = create_pbr_mat("Planter_Brass", (0.75, 0.60, 0.25, 1.0), roughness=0.30, metallic=0.85)
    foliage_mat = create_pbr_mat("Planter_Ficus_Foliage", (0.08, 0.40, 0.12, 1.0), roughness=0.55)

    # 1. Concourse wooden benches
    bench_positions = [
        (13.0, 15.0),
        (37.0, 15.0),
        (25.0, 24.5),
        (25.0, 3.5),
    ]
    for idx, (bx, by) in enumerate(bench_positions):
        # Bench seat
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(bx, by, 0.45))
        seat = bpy.context.active_object
        seat.name = f"Mall_Bench_{idx}"
        seat.scale = (2.2, 0.65, 0.10)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        seat.data.materials.append(bench_wood)
        for col in list(seat.users_collection):
            col.objects.unlink(seat)
        collection.objects.link(seat)

        # Chrome legs
        for leg_ox in [-0.9, 0.9]:
            bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.40, location=(bx + leg_ox, by, 0.20))
            leg = bpy.context.active_object
            leg.name = f"Mall_Bench_Leg_{idx}_{leg_ox}"
            leg.data.materials.append(bench_steel)
            for col in list(leg.users_collection):
                col.objects.unlink(leg)
            collection.objects.link(leg)

    # 2. Modern circular brass planters with lush ficus trees
    planter_positions = [
        (20.0, 15.0),
        (30.0, 15.0),
        (25.0, 19.5),
        (25.0, 6.5),
    ]
    for idx, (px, py) in enumerate(planter_positions):
        # Brass planter cylinder
        bpy.ops.mesh.primitive_cylinder_add(radius=0.75, depth=0.75, location=(px, py, 0.375))
        pot = bpy.context.active_object
        pot.name = f"Planter_Pot_{idx}"
        pot.data.materials.append(planter_brass)
        for col in list(pot.users_collection):
            col.objects.unlink(pot)
        collection.objects.link(pot)

        # Ficus greenery canopy
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=0.65, location=(px, py, 1.25))
        foliage = bpy.context.active_object
        foliage.name = f"Planter_Greenery_{idx}"
        foliage.scale = (1.2, 1.2, 1.1)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        foliage.data.materials.append(foliage_mat)
        for col in list(foliage.users_collection):
            col.objects.unlink(foliage)
        collection.objects.link(foliage)

    # 3. Modern Architectural Center Kiosk (Coffee & Specialty Display)
    kiosk_wood = create_pbr_mat("Kiosk_Dark_Oak", (0.22, 0.16, 0.11, 1.0), roughness=0.35)
    kiosk_quartz = create_pbr_mat("Kiosk_Quartz_Top", (0.92, 0.92, 0.90, 1.0), roughness=0.15, specular=0.85)
    kiosk_glow = create_pbr_mat("Kiosk_UnderGlow", (1.0, 0.88, 0.60, 1.0), emission_color=(1.0, 0.88, 0.60, 1.0), emission_strength=3.5)
    kiosk_menu = create_pbr_mat("Kiosk_Menu_Screen", (0.95, 0.95, 0.98, 1.0), emission_color=(0.95, 0.95, 0.98, 1.0), emission_strength=4.5)

    kx, ky = 25.0, 15.0
    # Base body
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(kx, ky, 0.45))
    kbody = bpy.context.active_object
    kbody.name = "Mall_Kiosk_Base"
    kbody.scale = (2.4, 1.2, 0.90)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    kbody.data.materials.append(kiosk_wood)
    for col in list(kbody.users_collection): col.objects.unlink(kbody)
    collection.objects.link(kbody)

    # Quartz Countertop
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(kx, ky, 0.93))
    ktop = bpy.context.active_object
    ktop.name = "Mall_Kiosk_Top"
    ktop.scale = (2.55, 1.35, 0.08)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    ktop.data.materials.append(kiosk_quartz)
    for col in list(ktop.users_collection): col.objects.unlink(ktop)
    collection.objects.link(ktop)

    # Warm under-counter LED strip
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(kx, ky, 0.88))
    kled = bpy.context.active_object
    kled.name = "Mall_Kiosk_LED"
    kled.scale = (2.42, 1.22, 0.03)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    kled.data.materials.append(kiosk_glow)
    for col in list(kled.users_collection): col.objects.unlink(kled)
    collection.objects.link(kled)

    # 4. Dual Freestanding Illuminated Directory / Advertising Totems
    totem_body_mat = create_pbr_mat("Totem_Body", (0.08, 0.08, 0.10, 1.0), roughness=0.20, metallic=0.90)
    totem_screen_mat = create_pbr_mat("Totem_Screen", (0.25, 0.65, 0.95, 1.0), emission_color=(0.25, 0.65, 0.95, 1.0), emission_strength=4.0)

    for tidx, tx in enumerate([16.0, 34.0]):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(tx, 15.0, 1.10))
        totem = bpy.context.active_object
        totem.name = f"Mall_Directory_Totem_{tidx}"
        totem.scale = (0.75, 0.22, 2.20)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        totem.data.materials.append(totem_body_mat)
        for col in list(totem.users_collection): col.objects.unlink(totem)
        collection.objects.link(totem)

        # Illuminated directory face
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(tx, 15.0, 1.25))
        screen = bpy.context.active_object
        screen.name = f"Mall_Directory_Screen_{tidx}"
        screen.scale = (0.65, 0.24, 1.40)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        screen.data.materials.append(totem_screen_mat)
        for col in list(screen.users_collection): col.objects.unlink(screen)
        collection.objects.link(screen)


def build_ceiling_grid_and_lighting(collection: bpy.types.Collection, width: float, depth: float):
    """Constructs commercial suspended ceiling grid with recessed linear LED lighting panels."""
    # Ceiling tile with soft indirect ambient bounce glow to eliminate dark void shadows
    ceiling_mat = create_pbr_mat(
        "Ceiling_Acoustic_Tile",
        (0.92, 0.92, 0.94, 1.0),
        roughness=0.85,
        emission_color=(0.95, 0.95, 0.98, 1.0),
        emission_strength=0.35
    )
    led_diffuser_mat = create_pbr_mat(
        "Ceiling_LED_Diffuser",
        (1.0, 0.98, 0.95, 1.0),
        roughness=0.1,
        emission_color=(1.0, 0.98, 0.94, 1.0),
        emission_strength=8.0
    )

    ceiling_z = 5.25

    # Main ceiling plane
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(width / 2.0, depth / 2.0, ceiling_z))
    ceil = bpy.context.active_object
    ceil.name = "Ceiling_Plane"
    ceil.scale = (width, depth, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    ceil.data.materials.append(ceiling_mat)
    for col in list(ceil.users_collection):
        col.objects.unlink(ceil)
    collection.objects.link(ceil)

    # 29 Recessed Linear LED Panel Arrays uniformly illuminating all concourses, corridors, exits, and rooms
    led_positions = [
        # West corridor
        (8.0, 4.0), (8.0, 9.5), (8.0, 15.0), (8.0, 20.5), (8.0, 26.0),
        # Central concourse West lane
        (18.0, 5.0), (18.0, 10.0), (18.0, 15.0), (18.0, 20.0), (18.0, 25.0),
        # Central concourse East lane
        (32.0, 5.0), (32.0, 10.0), (32.0, 15.0), (32.0, 20.0), (32.0, 25.0),
        # Central spine (Main Entrance, Atrium, Escalator, Kiosk, Food Court)
        (25.0, 4.0), (25.0, 8.5), (25.0, 15.0), (25.0, 21.0), (25.0, 26.5),
        # East corridor
        (42.0, 4.0), (42.0, 9.5), (42.0, 15.0), (42.0, 20.5), (42.0, 26.0),
        # Corner zones (Zone A, B, C, D exits & halls)
        (4.0, 5.0), (4.0, 25.0),
        (46.0, 5.0), (46.0, 25.0)
    ]

    for idx, (lx, ly) in enumerate(led_positions):
        # Recessed LED housing
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(lx, ly, ceiling_z - 0.03))
        panel = bpy.context.active_object
        panel.name = f"Ceiling_LED_Panel_{idx}"
        panel.scale = (2.4, 0.70, 0.05)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        panel.data.materials.append(led_diffuser_mat)
        for col in list(panel.users_collection):
            col.objects.unlink(panel)
        collection.objects.link(panel)

        # Actual downward area light (balanced commercial illumination)
        light_data = bpy.data.lights.new(name=f"Light_Ceiling_{idx}", type='AREA')
        light_data.energy = 340.0
        light_data.size = 2.4
        light_data.size_y = 0.80
        light_data.shape = 'RECTANGLE'
        light_data.color = (1.0, 0.98, 0.95)  # 4200K neutral commercial white
        light_obj = bpy.data.objects.new(f"Light_Ceiling_{idx}", light_data)
        light_obj.location = (lx, ly, ceiling_z - 0.06)
        collection.objects.link(light_obj)


def build_cctv_camera_fixtures(cameras_data: List[Dict[str, Any]], collection: bpy.types.Collection):
    """Places visual 3D CCTV dome camera models at the physical camera locations."""
    cctv_body_mat = create_pbr_mat("CCTV_Body", (0.92, 0.92, 0.94, 1.0), roughness=0.3)
    cctv_dome_mat = create_pbr_mat("CCTV_Dome", (0.05, 0.05, 0.05, 1.0), roughness=0.08, specular=0.95)
    led_rec_mat = create_pbr_mat("CCTV_LED_REC", (1.0, 0.05, 0.05, 1.0), emission_color=(1.0, 0.05, 0.05, 1.0), emission_strength=6.0)

    for cam in cameras_data:
        cid = cam["camera_id"]
        pos = cam["position"]
        cx, cy, cz = pos

        # Ceiling mount bracket
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=14,
            radius=0.15,
            depth=0.08,
            location=(cx, cy, cz - 0.04)
        )
        bracket = bpy.context.active_object
        bracket.name = f"Mount_{cid}"
        bracket.data.materials.append(cctv_body_mat)
        bracket.hide_render = True  # Never occlude the camera lens inside it
        for col in list(bracket.users_collection):
            col.objects.unlink(bracket)
        collection.objects.link(bracket)

        # Smoked Acrylic Dome
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=14,
            ring_count=8,
            radius=0.12,
            location=(cx, cy, cz - 0.10)
        )
        dome = bpy.context.active_object
        dome.name = f"Dome_{cid}"
        dome.scale = (1.0, 1.0, 0.70)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        dome.data.materials.append(cctv_dome_mat)
        dome.hide_render = True  # Never occlude the camera lens inside it
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
        led.hide_render = True
        for col in list(led.users_collection):
            col.objects.unlink(led)
        collection.objects.link(led)


def build_exit_and_entrance_doors(collection: bpy.types.Collection):
    """
    Constructs realistic glass commercial entrance & exit double-doors with:
    - Anodized architectural aluminum frames
    - Dual tempered glass door panels
    - Glowing emergency [EXIT] green signs
    - Exterior daytime backdrop preventing black voids through doorways
    """
    frame_mat = create_pbr_mat("Door_Aluminum", (0.35, 0.35, 0.38, 1.0), roughness=0.30, metallic=0.85)
    glass_mat = create_pbr_mat("Door_Glass", (0.94, 0.96, 1.0, 1.0), roughness=0.04, specular=0.95, transmission=0.88)
    exit_sign_mat = create_pbr_mat(
        "Exit_Sign_Green",
        (0.10, 0.90, 0.25, 1.0),
        roughness=0.20,
        emission_color=(0.10, 1.0, 0.30, 1.0),
        emission_strength=4.5
    )
    exterior_wall_mat = create_pbr_mat(
        "Exterior_Daylight_Backdrop",
        (0.88, 0.90, 0.94, 1.0),
        roughness=0.80,
        emission_color=(0.85, 0.90, 0.95, 1.0),
        emission_strength=0.80
    )

    portals = [
        # Exit A (North-West corridor end)
        {"name": "Exit_A", "pos": [8.0, 30.0, 0.0], "width": 2.0, "is_exit": True, "ext_offset_y": 1.2},
        # Exit B (North-East corridor end)
        {"name": "Exit_B", "pos": [42.0, 30.0, 0.0], "width": 2.0, "is_exit": True, "ext_offset_y": 1.2},
        # Exit C (South-West corridor end)
        {"name": "Exit_C", "pos": [8.0, 0.0, 0.0], "width": 2.0, "is_exit": True, "ext_offset_y": -1.2},
        # Exit D (South-East corridor end)
        {"name": "Exit_D", "pos": [42.0, 0.0, 0.0], "width": 2.0, "is_exit": True, "ext_offset_y": -1.2},
        # Main Entrance (Grand concourse entrance)
        {"name": "Main_Entrance", "pos": [25.0, 0.0, 0.0], "width": 3.0, "is_exit": False, "ext_offset_y": -1.5},
    ]

    for p in portals:
        px, py, pz = p["pos"]
        pw = p["width"]
        pname = p["name"]
        ey = p["ext_offset_y"]

        # 1. Door Frame Header & Jambs
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(px, py, 2.70))
        hdr = bpy.context.active_object
        hdr.name = f"DoorHeader_{pname}"
        hdr.scale = (pw, 0.15, 0.20)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        hdr.data.materials.append(frame_mat)
        for col in list(hdr.users_collection): col.objects.unlink(hdr)
        collection.objects.link(hdr)

        for j_sign, j_name in [(-1, "L"), (1, "R")]:
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(px + j_sign * (pw / 2.0), py, 1.35))
            jamb = bpy.context.active_object
            jamb.name = f"DoorJamb_{pname}_{j_name}"
            jamb.scale = (0.10, 0.15, 2.70)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            jamb.data.materials.append(frame_mat)
            for col in list(jamb.users_collection): col.objects.unlink(jamb)
            collection.objects.link(jamb)

        # 2. Tempered Glass Double Doors
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(px, py, 1.35))
        dglass = bpy.context.active_object
        dglass.name = f"DoorGlass_{pname}"
        dglass.scale = (pw - 0.12, 0.05, 2.50)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        dglass.data.materials.append(glass_mat)
        for col in list(dglass.users_collection): col.objects.unlink(dglass)
        collection.objects.link(dglass)

        # 3. Emergency Green Illuminated Exit Sign
        if p["is_exit"]:
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(px, py, 2.95))
            sign = bpy.context.active_object
            sign.name = f"ExitSign_{pname}"
            sign.scale = (0.70, 0.12, 0.25)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            sign.data.materials.append(exit_sign_mat)
            for col in list(sign.users_collection): col.objects.unlink(sign)
            collection.objects.link(sign)

        # 4. Exterior Daylight Sidewalk & Vestibule Backdrop
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(px, py + ey, 1.80))
        ext_wall = bpy.context.active_object
        ext_wall.name = f"ExtBackdrop_{pname}"
        ext_wall.scale = (pw + 2.0, 0.15, 3.80)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        ext_wall.data.materials.append(exterior_wall_mat)
        for col in list(ext_wall.users_collection): col.objects.unlink(ext_wall)
        collection.objects.link(ext_wall)


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
    create_luxury_marble_floor(W, H)

    # 2. Materials
    wall_mat = create_pbr_mat("Wall_Drywall", (0.65, 0.65, 0.68, 1.0), roughness=0.60)
    pillar_mat = create_pbr_mat("Pillar_Composite", (0.80, 0.80, 0.82, 1.0), roughness=0.40, specular=0.6)
    chrome_kick_mat = create_pbr_mat("Pillar_Chrome_Kick", (0.90, 0.90, 0.92, 1.0), roughness=0.18, metallic=0.95)
    table_mat = create_pbr_mat("Table_Maple", (0.42, 0.28, 0.18, 1.0), roughness=0.40)

    # 3. Walls
    walls_col = bpy.data.collections.new("Walls")
    bpy.context.scene.collection.children.link(walls_col)
    for w in data.get("walls", []):
        p1, p2 = w["start"], w["end"]
        height = 5.25
        thickness = w.get("thickness", 0.22)
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

    # 4. Obstacles (Pillars & Tables)
    obs_col = bpy.data.collections.new("Obstacles")
    bpy.context.scene.collection.children.link(obs_col)
    for obs in data.get("obstacles", []):
        pos = obs["position"]
        otype = obs.get("type", "box")
        oname = obs.get("name", "obstacle")
        if otype == "cylinder":
            radius = obs.get("radius", 0.48)
            h = 5.25

            # Main column shaft
            bpy.ops.mesh.primitive_cylinder_add(
                radius=radius,
                depth=h - 0.6,
                location=(pos[0], pos[1], (h - 0.6) / 2.0 + 0.6)
            )
            o_obj = bpy.context.active_object
            o_obj.name = f"Obs_{oname}"
            o_obj.data.materials.append(pillar_mat)

            for col in list(o_obj.users_collection):
                col.objects.unlink(o_obj)
            obs_col.objects.link(o_obj)

            # Polished Stainless Steel Kick-Plate Base Ring
            bpy.ops.mesh.primitive_cylinder_add(
                radius=radius + 0.03,
                depth=0.6,
                location=(pos[0], pos[1], 0.30)
            )
            kick_obj = bpy.context.active_object
            kick_obj.name = f"Kick_{oname}"
            kick_obj.data.materials.append(chrome_kick_mat)

            for col in list(kick_obj.users_collection):
                col.objects.unlink(kick_obj)
            obs_col.objects.link(kick_obj)

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

    # 5. Storefronts, Escalators, Benches, Greenery, Ceiling
    decor_col = bpy.data.collections.new("Mall_Interior_Decor")
    bpy.context.scene.collection.children.link(decor_col)
    build_retail_storefronts(decor_col)
    build_escalators(decor_col)
    build_concourse_benches_and_planters(decor_col)
    build_exit_and_entrance_doors(decor_col)
    build_ceiling_grid_and_lighting(decor_col, W, H)

    # 6. Physical CCTV Camera Mount Fixtures
    cams_data = data.get("cameras", [])
    build_cctv_camera_fixtures(cams_data, decor_col)

    # 7. Virtual CCTV Cameras
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

        # Realistic CCTV depth of field focused on floor/pedestrian level
        cam_data.dof.use_dof = False

        cam_obj = bpy.data.objects.new(name=cid, object_data=cam_data)
        cam_obj.location = pos
        direction = look_at - pos
        cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        cam_col.objects.link(cam_obj)

        if first_cam is None:
            first_cam = cam_obj

    if first_cam:
        bpy.context.scene.camera = first_cam

    # 8. EEVEE Render Settings (High Fidelity, Clean Digital IP Camera Optics)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.eevee.taa_render_samples = 24  # High quality rasterization

    # Color Management: AgX Neutral Realistic Contrast
    if hasattr(scene, "view_settings"):
        try:
            scene.view_settings.view_transform = 'AgX'
            scene.view_settings.look = 'AgX - Medium High Contrast'
            scene.view_settings.exposure = 0.0
        except Exception:
            scene.view_settings.view_transform = 'Filmic'
            scene.view_settings.look = 'Medium High Contrast'
            scene.view_settings.exposure = 0.0

    world = bpy.data.worlds.new("Mall_World_PBR")
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.82, 0.86, 0.92, 1.0)
        bg_node.inputs["Strength"].default_value = 0.65
    scene.world = world

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_blend)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_blend)
    print(f"[✓] Best-Version 3D Mall Scene saved to: {output_blend}")
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
