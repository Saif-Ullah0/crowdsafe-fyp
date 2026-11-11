"""
CrowdSafe Synthetic Simulation — 3D Humanoid Agent Generator for Blender 4.2 LTS
Constructs lightweight "Capsule-Plus" humanoids (~70-100 polygons per agent):
- UV sphere head (r ≈ 0.12m)
- Cylinder torso
- Leg columns
- Optional accessories (hats, backpacks)
- Anthropometric height randomization (Adults 1.55-1.90m, Children 1.20-1.40m)
- 20+ color palette for clothing and materials
- Synchronized walking gait bobbing and dynamic yaw alignment
"""

import math
import random
from typing import Dict, Tuple, Optional, List

import bpy
from mathutils import Vector, Euler


# ==============================================================================
# Comprehensive Color Palettes (Clothing, Skin, Accessories)
# ==============================================================================
CLOTHING_PALETTE = [
    # Dark tones
    (0.10, 0.10, 0.12, 1.0),  # Charcoal black
    (0.08, 0.12, 0.22, 1.0),  # Deep navy
    (0.18, 0.14, 0.10, 1.0),  # Espresso brown
    (0.16, 0.18, 0.20, 1.0),  # Dark slate
    (0.05, 0.05, 0.05, 1.0),  # Jet black
    # Neutral tones
    (0.85, 0.85, 0.85, 1.0),  # Clean white
    (0.55, 0.55, 0.55, 1.0),  # Heather gray
    (0.70, 0.62, 0.48, 1.0),  # Khaki / beige
    (0.35, 0.38, 0.30, 1.0),  # Olive green
    (0.60, 0.48, 0.35, 1.0),  # Camel tan
    # Vibrant tones
    (0.75, 0.12, 0.12, 1.0),  # Crimson red
    (0.12, 0.35, 0.75, 1.0),  # Royal blue
    (0.15, 0.55, 0.25, 1.0),  # Emerald green
    (0.90, 0.72, 0.10, 1.0),  # Saffron yellow
    (0.85, 0.40, 0.10, 1.0),  # Burnt orange
    (0.50, 0.15, 0.55, 1.0),  # Plum purple
    (0.10, 0.55, 0.60, 1.0),  # Teal cyan
    (0.85, 0.28, 0.45, 1.0),  # Coral magenta
    (0.40, 0.55, 0.75, 1.0),  # Sky denim
    (0.72, 0.30, 0.20, 1.0),  # Terracotta
    (0.25, 0.45, 0.40, 1.0),  # Sage pine
    (0.92, 0.88, 0.75, 1.0)   # Cream linen
]

SKIN_PALETTE = [
    (0.95, 0.80, 0.68, 1.0),  # Light porcelain
    (0.88, 0.70, 0.55, 1.0),  # Fair tan
    (0.75, 0.55, 0.40, 1.0),  # Warm medium
    (0.55, 0.38, 0.26, 1.0),  # Rich olive/brown
    (0.35, 0.22, 0.14, 1.0),  # Deep brown
    (0.22, 0.14, 0.10, 1.0)   # Dark espresso
]

HAIR_PALETTE = [
    (0.08, 0.08, 0.08, 1.0),  # Black
    (0.25, 0.16, 0.10, 1.0),  # Dark brown
    (0.45, 0.30, 0.18, 1.0),  # Medium brown
    (0.70, 0.55, 0.30, 1.0),  # Blonde
    (0.55, 0.25, 0.12, 1.0),  # Auburn red
    (0.65, 0.65, 0.65, 1.0)   # Silver gray
]


class MaterialCache:
    """Manages and reuses Blender materials to prevent memory bloating."""
    _cache = {}

    @classmethod
    def get_or_create(cls, prefix: str, color_rgba: Tuple[float, float, float, float], roughness: float = 0.7):
        key = (prefix, tuple(round(c, 3) for c in color_rgba), round(roughness, 2))
        if key in cls._cache and cls._cache[key].name in bpy.data.materials:
            return cls._cache[key]

        mat_name = f"Mat_{prefix}_{len(cls._cache)}"
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            if "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = color_rgba
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = roughness
        
        cls._cache[key] = mat
        return mat


class HumanoidModel:
    """
    Spawns and manages a low-poly capsule-plus humanoid avatar in Blender.
    """

    def __init__(
        self,
        agent_id: int,
        collection: bpy.types.Collection,
        height: Optional[float] = None,
        is_child: Optional[bool] = None,
        color_idx: Optional[int] = None,
    ):
        self.agent_id = agent_id
        self.collection = collection

        # Anthropometrics determination
        if is_child is None:
            # 10% chance of child/youth in public mall
            self.is_child = (random.random() < 0.10)
        else:
            self.is_child = is_child

        if height is None:
            if self.is_child:
                self.height = round(random.uniform(1.20, 1.40), 2)
            else:
                self.height = round(random.uniform(1.55, 1.90), 2)
        else:
            self.height = height

        self.scale_factor = self.height / 1.72  # Normalized against 1.72m reference adult

        # Color selections
        if color_idx is not None and 0 <= color_idx < len(CLOTHING_PALETTE):
            self.top_color = CLOTHING_PALETTE[color_idx]
        else:
            self.top_color = random.choice(CLOTHING_PALETTE)
        
        self.bottom_color = random.choice(CLOTHING_PALETTE)
        self.skin_color = random.choice(SKIN_PALETTE)
        self.hair_color = random.choice(HAIR_PALETTE)

        # Materials
        self.mat_top = MaterialCache.get_or_create("Top", self.top_color, roughness=0.75)
        self.mat_bottom = MaterialCache.get_or_create("Bottom", self.bottom_color, roughness=0.80)
        self.mat_skin = MaterialCache.get_or_create("Skin", self.skin_color, roughness=0.60)
        self.mat_hair = MaterialCache.get_or_create("Hair", self.hair_color, roughness=0.90)

        # Build avatar mesh hierarchy
        self._build_mesh()

    def _build_mesh(self):
        """Constructs the root empty and child body primitives."""
        # Root object
        self.root = bpy.data.objects.new(f"Agent_{self.agent_id:04d}", None)
        self.root.empty_display_size = 0.2
        self.collection.objects.link(self.root)

        s = self.scale_factor
        head_radius = 0.115 * s
        torso_radius = 0.17 * s
        torso_height = 0.52 * s
        leg_radius = 0.12 * s
        leg_height = 0.75 * s

        # ---------------------------------------------------------------------
        # 1. Head (UV Sphere, ~48 faces)
        # ---------------------------------------------------------------------
        head_z = self.height - head_radius
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=8,
            ring_count=6,
            radius=head_radius,
            location=(0.0, 0.0, head_z)
        )
        self.head_obj = bpy.context.active_object
        self.head_obj.name = f"Agent_{self.agent_id:04d}_Head"
        self.head_obj.data.materials.append(self.mat_skin)
        self.head_obj.parent = self.root

        # Move to collection
        for col in list(self.head_obj.users_collection):
            col.objects.unlink(self.head_obj)
        self.collection.objects.link(self.head_obj)

        # ---------------------------------------------------------------------
        # 2. Torso (Cylinder, 8 sides = ~16 faces)
        # ---------------------------------------------------------------------
        torso_z = leg_height + (torso_height / 2.0)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=8,
            radius=torso_radius,
            depth=torso_height,
            location=(0.0, 0.0, torso_z)
        )
        self.torso_obj = bpy.context.active_object
        self.torso_obj.name = f"Agent_{self.agent_id:04d}_Torso"
        self.torso_obj.data.materials.append(self.mat_top)
        self.torso_obj.parent = self.root

        for col in list(self.torso_obj.users_collection):
            col.objects.unlink(self.torso_obj)
        self.collection.objects.link(self.torso_obj)

        # ---------------------------------------------------------------------
        # 3. Legs (Cylinder, 8 sides = ~16 faces)
        # ---------------------------------------------------------------------
        legs_z = leg_height / 2.0
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=8,
            radius=leg_radius,
            depth=leg_height,
            location=(0.0, 0.0, legs_z)
        )
        self.legs_obj = bpy.context.active_object
        self.legs_obj.name = f"Agent_{self.agent_id:04d}_Legs"
        self.legs_obj.data.materials.append(self.mat_bottom)
        self.legs_obj.parent = self.root

        for col in list(self.legs_obj.users_collection):
            col.objects.unlink(self.legs_obj)
        self.collection.objects.link(self.legs_obj)

        # ---------------------------------------------------------------------
        # 4. Optional Accessories (Backpack ~25%, Cap ~20%)
        # ---------------------------------------------------------------------
        self.accessory_obj = None
        has_backpack = (random.random() < 0.25)
        has_cap = (random.random() < 0.20)

        if has_backpack:
            bp_mat = MaterialCache.get_or_create("Accessory", (0.12, 0.12, 0.14, 1.0), roughness=0.85)
            bp_y = -(torso_radius + 0.08 * s)
            bpy.ops.mesh.primitive_cube_add(
                size=1.0,
                location=(0.0, bp_y, torso_z + 0.05 * s)
            )
            self.accessory_obj = bpy.context.active_object
            self.accessory_obj.name = f"Agent_{self.agent_id:04d}_Backpack"
            self.accessory_obj.scale = (0.24 * s, 0.14 * s, 0.32 * s)
            self.accessory_obj.data.materials.append(bp_mat)
            self.accessory_obj.parent = self.root
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

            for col in list(self.accessory_obj.users_collection):
                col.objects.unlink(self.accessory_obj)
            self.collection.objects.link(self.accessory_obj)

        elif has_cap:
            cap_mat = MaterialCache.get_or_create("Accessory", self.hair_color, roughness=0.7)
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=8,
                radius=head_radius * 1.08,
                depth=0.06 * s,
                location=(0.0, 0.0, head_z + head_radius * 0.7)
            )
            self.accessory_obj = bpy.context.active_object
            self.accessory_obj.name = f"Agent_{self.agent_id:04d}_Cap"
            self.accessory_obj.data.materials.append(cap_mat)
            self.accessory_obj.parent = self.root

            for col in list(self.accessory_obj.users_collection):
                col.objects.unlink(self.accessory_obj)
            self.collection.objects.link(self.accessory_obj)

    def set_pose(self, x: float, y: float, z_offset: float, yaw_rad: float):
        """Updates world position, vertical gait bobbing, and yaw rotation."""
        self.root.location = (x, y, z_offset)
        self.root.rotation_euler = Euler((0.0, 0.0, yaw_rad), 'XYZ')

    def get_head_world_position(self) -> Vector:
        """Returns the exact 3D world center coordinate of the head sphere."""
        # Ensure world matrix is evaluated
        return self.head_obj.matrix_world.translation

    def set_visibility(self, visible: bool):
        """Hides or shows the agent (e.g. before spawn or after exit)."""
        self.root.hide_render = not visible
        self.head_obj.hide_render = not visible
        self.torso_obj.hide_render = not visible
        self.legs_obj.hide_render = not visible
        if self.accessory_obj:
            self.accessory_obj.hide_render = not visible


def spawn_crowd_in_scene(
    scene: bpy.types.Scene,
    agents_data: List[Dict[str, any]],
    collection_name: str = "Crowd_Agents"
) -> Dict[int, HumanoidModel]:
    """
    Instantiates all humanoid agents into the Blender scene under a dedicated collection.
    """
    # Check or create collection
    if collection_name in bpy.data.collections:
        crowd_col = bpy.data.collections[collection_name]
    else:
        crowd_col = bpy.data.collections.new(collection_name)
        scene.collection.children.link(crowd_col)

    agent_models = {}
    for a in agents_data:
        aid = a["id"]
        height = a.get("height", None)
        color_idx = a.get("color_idx", None)
        
        model = HumanoidModel(
            agent_id=aid,
            collection=crowd_col,
            height=height,
            color_idx=color_idx
        )
        
        # Initial placement
        x = a.get("x", 0.0)
        y = a.get("y", 0.0)
        yaw = a.get("yaw", 0.0)
        model.set_pose(x, y, 0.0, yaw)
        
        agent_models[aid] = model

    print(f">>> Successfully spawned {len(agent_models)} capsule-plus humanoids into collection '{collection_name}'")
    return agent_models
