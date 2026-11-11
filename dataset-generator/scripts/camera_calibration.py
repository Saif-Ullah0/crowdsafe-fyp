"""
CrowdSafe Synthetic Simulation — Camera Calibration & Matrix Mathematics
Computes mathematical camera parameters for virtual CCTV cameras in Blender:
- 3x3 Intrinsic Matrix (K) matching 6mm wide-angle lens and 640x480 resolution
- 4x4 Extrinsic Matrix ([R|t]) adhering to standard OpenCV coordinate frame
- 3D-to-2D pinhole projection and frustum verification
"""

from typing import Dict, Tuple, List, Optional
import bpy
from mathutils import Matrix, Vector


def get_camera_intrinsics(camera_obj: bpy.types.Object, scene: Optional[bpy.types.Scene] = None) -> Dict[str, any]:
    """
    Computes the 3x3 Pinhole Intrinsic Matrix K and camera optical specs.
    
    K = [
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ]
    """
    if scene is None:
        scene = bpy.context.scene

    camd = camera_obj.data
    f_mm = camd.lens
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.0

    width_px = res_x * scale
    height_px = res_y * scale

    sensor_w = camd.sensor_width
    sensor_h = camd.sensor_height
    pixel_aspect = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if camd.sensor_fit == 'VERTICAL':
        s_u = width_px / sensor_w / pixel_aspect
        s_v = height_px / sensor_h
    else:  # 'HORIZONTAL' or 'AUTO'
        s_u = width_px / sensor_w
        s_v = height_px * pixel_aspect / sensor_h

    fx = f_mm * s_u
    fy = f_mm * s_v
    cx = width_px / 2.0
    cy = height_px / 2.0

    K = [
        [round(fx, 4), 0.0, round(cx, 4)],
        [0.0, round(fy, 4), round(cy, 4)],
        [0.0, 0.0, 1.0]
    ]

    # Field of View calculation
    fov_x_rad = 2.0 * math.atan(sensor_w / (2.0 * f_mm)) if 'math' in globals() else 0.0
    fov_y_rad = 2.0 * math.atan(sensor_h / (2.0 * f_mm)) if 'math' in globals() else 0.0

    return {
        "K_matrix": K,
        "fx": round(fx, 4),
        "fy": round(fy, 4),
        "cx": round(cx, 4),
        "cy": round(cy, 4),
        "focal_length_mm": f_mm,
        "sensor_width_mm": sensor_w,
        "sensor_height_mm": sensor_h,
        "resolution": [int(width_px), int(height_px)]
    }


import math  # Ensure math is available for FOV


def get_camera_extrinsics(camera_obj: bpy.types.Object) -> Dict[str, any]:
    """
    Computes the 4x4 World-to-Camera Extrinsic Matrix [R | t] in OpenCV convention.
    
    OpenCV Frame: +X Right, +Y Down, +Z Forward.
    Blender Frame: +X Right, +Y Up,   +Z Backward.
    """
    # Blender Camera-to-World (cam pose in world)
    cam_to_world = camera_obj.matrix_world

    # Invert to get World-to-Camera
    world_to_cam_blender = cam_to_world.inverted()

    # Transformation to OpenCV coordinate convention
    to_opencv = Matrix((
        (1.0,  0.0,  0.0,  0.0),
        (0.0, -1.0,  0.0,  0.0),
        (0.0,  0.0, -1.0,  0.0),
        (0.0,  0.0,  0.0,  1.0)
    ))

    # Combined OpenCV World-to-Camera matrix
    RT = to_opencv @ world_to_cam_blender

    # Extract 3x3 Rotation R and 3x1 Translation t
    R = [[round(RT[r][c], 6) for c in range(3)] for r in range(3)]
    t = [round(RT[r][3], 6) for r in range(3)]

    pos = camera_obj.matrix_world.translation
    rot_euler_deg = [round(math.degrees(a), 2) for a in camera_obj.rotation_euler]

    return {
        "RT_matrix_4x4": [[round(RT[r][c], 6) for c in range(4)] for r in range(4)],
        "R_3x3": R,
        "t_3x1": t,
        "world_position": [round(pos.x, 3), round(pos.y, 3), round(pos.z, 3)],
        "euler_rotation_deg": rot_euler_deg
    }


def get_full_camera_calibration(camera_obj: bpy.types.Object, scene: Optional[bpy.types.Scene] = None) -> Dict[str, any]:
    """Returns combined intrinsic and extrinsic calibration payload."""
    intrinsics = get_camera_intrinsics(camera_obj, scene)
    extrinsics = get_camera_extrinsics(camera_obj)
    return {
        "camera_id": camera_obj.name,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics
    }
