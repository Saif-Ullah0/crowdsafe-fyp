"""
CrowdSafe Synthetic Simulation — Master Scenario Orchestrator for Blender 4.2 LTS
Executes complete simulation runs:
1. Loads mall_scene.blend with venue geometry, lighting, and camera array.
2. Spawns crowd agents with randomized anthropometrics and routes.
3. Steps Helbing Social Force physics frame-by-frame.
4. Renders multi-camera frames (EEVEE engine).
5. Applies dual-mode post-processing (CCTV sensor degradation vs Clean).
6. Projects head coordinates, performs ray-cast occlusion culling, and computes Fruin zone densities.
7. Exports per-frame ground truth JSON and scenario summary metadata.
"""

import os
import sys
import json
import time
import math
import random
import argparse
from typing import Dict, List, Any

import bpy
from mathutils import Vector

# Ensure dataset-generator directory is in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from agents.crowd_physics import SocialForceEngine, PedestrianAgent
from agents.humanoid_generator import spawn_crowd_in_scene
from scripts.camera_calibration import get_full_camera_calibration
from scripts.occlusion_tracker import track_camera_crowd
from scripts.render_pipeline import process_rendered_frame


def parse_args():
    argv = sys.argv
    if "--" in argv:
        script_args = argv[argv.index("--") + 1:]
    else:
        script_args = argv[1:]

    parser = argparse.ArgumentParser(description="Run complete crowd simulation scenario in Blender.")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON file")
    parser.add_argument("--floor-plan", default=os.path.join(parent_dir, "venue", "mall_floor_plan.json"), help="Path to floor plan JSON")
    parser.add_argument("--output", default=os.path.join(parent_dir, "output", "scenario_run"), help="Output directory")
    parser.add_argument("--frames", type=int, default=None, help="Override frame count")
    parser.add_argument("--cameras", default="all", help="Cameras to render: 'all' or comma-separated e.g. 'CAM-01,CAM-02'")
    parser.add_argument("--mode", default="cctv", choices=["cctv", "clean"], help="Render mode: cctv (default) or clean")
    return parser.parse_args(script_args)


def setup_agents(scenario: Dict[str, Any], floor_plan: Dict[str, Any], engine: SocialForceEngine) -> List[Dict[str, Any]]:
    """Initializes agents with randomized starting positions, routes, and speeds."""
    nav_nodes = floor_plan["navigation_graph"]["nodes"]
    num_agents = scenario.get("num_agents", 40)
    spawn_dist = scenario.get("spawn_distribution", {"N0": 1.0})
    route_patterns = scenario.get("route_patterns", [])
    speed_range = scenario.get("speed_range", [1.1, 1.4])

    spawn_keys = list(spawn_dist.keys())
    spawn_weights = [spawn_dist[k] for k in spawn_keys]

    agents_init_data = []

    for i in range(num_agents):
        # Pick spawn node based on weights
        spawn_node_id = random.choices(spawn_keys, weights=spawn_weights, k=1)[0]
        base_pos = nav_nodes[spawn_node_id]["pos"]
        # Add jitter around spawn node
        start_x = base_pos[0] + random.uniform(-1.0, 1.0)
        start_y = base_pos[1] + random.uniform(-1.0, 1.0)
        start_x = max(1.0, min(49.0, start_x))
        start_y = max(1.0, min(29.0, start_y))

        # Select matching route
        matching_routes = [r for r in route_patterns if r["from"] == spawn_node_id]
        if matching_routes:
            weights = [r.get("weight", 1.0) for r in matching_routes]
            selected_route_def = random.choices(matching_routes, weights=weights, k=1)[0]
            route = selected_route_def["route"]
        else:
            # Fallback route: wander to central corridor and food court
            route = ["N2", "N4"]

        desired_speed = random.uniform(speed_range[0], speed_range[1])
        lateral_offset = random.uniform(-0.4, 0.4)

        # 10% child, 90% adult
        is_child = (random.random() < 0.10)
        height = round(random.uniform(1.20, 1.40), 2) if is_child else round(random.uniform(1.55, 1.90), 2)
        color_idx = random.randint(0, 21)

        ped = PedestrianAgent(
            agent_id=i,
            start_pos=(start_x, start_y),
            route=route,
            desired_speed=desired_speed,
            radius=0.25,
            mass=70.0 if not is_child else 40.0,
            lateral_offset=lateral_offset,
            height=height,
            color_idx=color_idx
        )
        engine.add_agent(ped)

        agents_init_data.append({
            "id": i,
            "x": start_x,
            "y": start_y,
            "height": height,
            "color_idx": color_idx,
            "yaw": 0.0
        })

    return agents_init_data


def main():
    args = parse_args()
    start_time = time.time()

    print("==================================================")
    print("CrowdSafe Synthetic Simulation Orchestrator")
    print(f"Scenario File: {args.scenario}")
    print(f"Render Mode:   {args.mode.upper()}")
    print(f"Output Path:   {args.output}")
    print("==================================================")

    # 1. Load configurations
    with open(args.scenario, "r") as f:
        scenario = json.load(f)
    with open(args.floor_plan, "r") as f:
        floor_plan = json.load(f)

    num_frames = args.frames if args.frames is not None else scenario.get("duration_frames", 100)
    fps = scenario.get("fps", 1.0)  # Default 1.0 FPS for surveillance CCTV
    frame_interval_sec = 1.0 / float(fps)
    # Perform 10 sub-steps per frame for smooth Social Force numerical stability
    substeps = 10
    sub_dt = frame_interval_sec / float(substeps)
    zones = floor_plan.get("zones", {})

    # 2. Setup Social Force Physics Engine
    nav_nodes = {k: v["pos"] for k, v in floor_plan["navigation_graph"]["nodes"].items()}
    walls = floor_plan.get("walls", [])
    obstacles = floor_plan.get("obstacles", [])

    engine = SocialForceEngine(nav_nodes, walls, obstacles)
    agents_init_data = setup_agents(scenario, floor_plan, engine)

    # 3. Spawn 3D Humanoid Models in Blender Scene
    scene = bpy.context.scene
    agent_models = spawn_crowd_in_scene(scene, agents_init_data)
    bpy.context.view_layer.update()

    # 4. Filter and calibrate cameras
    all_cam_objs = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if args.cameras.lower() == "all":
        selected_cams = all_cam_objs
    else:
        req_cam_names = [c.strip() for c in args.cameras.split(",")]
        selected_cams = [obj for obj in all_cam_objs if obj.name in req_cam_names]

    if not selected_cams:
        raise RuntimeError(f"No cameras found matching request '{args.cameras}'!")

    print(f"Active CCTV Cameras ({len(selected_cams)}): {[c.name for c in selected_cams]}")

    # Prepare output directories
    os.makedirs(args.output, exist_ok=True)
    anno_dir = os.path.join(args.output, "annotations")
    os.makedirs(anno_dir, exist_ok=True)

    cam_output_dirs = {}
    for c in selected_cams:
        cdir = os.path.join(args.output, c.name.lower().replace("-", "_"))
        os.makedirs(cdir, exist_ok=True)
        cam_output_dirs[c.name] = cdir

    # Precompute Camera Calibrations
    camera_calibrations = {}
    for c in selected_cams:
        camera_calibrations[c.name] = get_full_camera_calibration(c, scene)

    # 5. Master Simulation & Render Loop
    print(f"\nStarting Simulation Loop across {num_frames} frames...")
    total_renders = 0

    for frame_idx in range(num_frames):
        # Step physics forward with smooth substeps
        for _ in range(substeps):
            engine.step(dt=sub_dt)
        agent_states = engine.get_agent_states()

        # Update 3D agent positions, gait bobbing, and heading yaw in Blender
        for state in agent_states:
            aid = state["id"]
            if aid in agent_models:
                model = agent_models[aid]
                if state["is_active"]:
                    model.set_visibility(True)
                    bob_offset = state["z"] - state["height"]
                    model.set_pose(state["x"], state["y"], bob_offset, state["yaw"])
                else:
                    model.set_visibility(False)

        # Synchronize dependency graph
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()

        # Render and annotate each camera
        frame_vis_counts = []
        for cam_obj in selected_cams:
            cam_name = cam_obj.name
            scene.camera = cam_obj

            # Raw render destination
            raw_filepath = os.path.join(cam_output_dirs[cam_name], f"raw_{frame_idx:04d}.png")
            final_filepath = os.path.join(cam_output_dirs[cam_name], f"frame_{frame_idx:04d}.png")
            scene.render.filepath = raw_filepath

            # EEVEE rasterizer render
            bpy.ops.render.render(write_still=True)
            total_renders += 1

            # Apply dual-mode post-processing (CCTV vs Clean)
            process_rendered_frame(raw_filepath, final_filepath, mode=args.mode, frame_idx=frame_idx)
            if os.path.exists(raw_filepath) and raw_filepath != final_filepath:
                os.remove(raw_filepath)

            # Compute ground truth annotations & line-of-sight ray-cast occlusion culling
            tracking = track_camera_crowd(scene, cam_obj, agent_models, zones, depsgraph=depsgraph)
            frame_vis_counts.append(tracking["total_visible_count"])

            # Structure Annotation JSON Payload
            annotation_record = {
                "frame_id": frame_idx,
                "timestamp_sec": round(frame_idx * frame_interval_sec, 3),
                "camera_id": cam_name,
                "image_path": os.path.relpath(final_filepath, args.output),
                "image_width": tracking["image_width"],
                "image_height": tracking["image_height"],
                "render_mode": args.mode,
                "total_visible_count": tracking["total_visible_count"],
                "total_occluded_count": tracking["total_occluded_count"],
                "frustum_count": tracking["frustum_count"],
                "head_coordinates": tracking["visible_heads"],
                "occluded_heads": tracking["occluded_heads"],
                "zone_densities": tracking["zone_densities"],
                "calibration": camera_calibrations[cam_name]
            }

            anno_file = os.path.join(anno_dir, f"{cam_name.lower().replace('-', '_')}_frame_{frame_idx:04d}.json")
            with open(anno_file, "w") as af:
                json.dump(annotation_record, af, indent=2)

        # Progress log
        if (frame_idx + 1) % 5 == 0 or frame_idx == 0 or (frame_idx + 1) == num_frames:
            elapsed = time.time() - start_time
            fps_speed = (frame_idx + 1) / max(elapsed, 0.001)
            vis_str = ", ".join(f"{c.name}: {cnt}" for c, cnt in zip(selected_cams, frame_vis_counts))
            print(f"[{frame_idx+1:03d}/{num_frames}] Progress | {fps_speed:.1f} sim-fps | Visible Heads ({vis_str})")

    # 6. Export Scenario Run Metadata
    elapsed_total = round(time.time() - start_time, 2)
    summary_data = {
        "scenario_name": scenario.get("name", "custom"),
        "description": scenario.get("description", ""),
        "total_frames": num_frames,
        "total_renders": total_renders,
        "cameras": [c.name for c in selected_cams],
        "render_mode": args.mode,
        "elapsed_time_sec": elapsed_total,
        "average_fps": round(total_renders / max(elapsed_total, 0.001), 2),
        "calibrations": camera_calibrations
    }

    summary_file = os.path.join(args.output, "scenario_summary.json")
    with open(summary_file, "w") as sf:
        json.dump(summary_data, sf, indent=2)

    print("\n==================================================")
    print(f">>> Scenario run COMPLETED in {elapsed_total} seconds!")
    print(f">>> Total frames rendered: {total_renders}")
    print(f">>> Dataset saved at: {args.output}")
    print("==================================================")


if __name__ == "__main__":
    main()
