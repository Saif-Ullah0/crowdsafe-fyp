#!/usr/bin/env python3
"""
CrowdSafe Synthetic Dataset & Surveillance Video Generator — Master Host CLI Runner
Usage:
  python dataset-generator/generate.py --scenario rush_hour --frames 50 --mode cctv --video
  python dataset-generator/generate.py --scenario bottleneck --character-style realistic --video --tracking-video
  python dataset-generator/generate.py --scenario normal_flow --cameras CAM-02 --mode clean
"""

import os
import sys
import shutil
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def find_blender_binary() -> str:
    """Locates the Blender 4.2 LTS binary in tools/ or system PATH."""
    local_blender = os.path.join(SCRIPT_DIR, "tools", "blender", "blender")
    if os.path.isfile(local_blender) and os.access(local_blender, os.X_OK):
        return local_blender

    system_blender = shutil.which("blender")
    if system_blender:
        return system_blender

    raise FileNotFoundError(
        "Blender executable not found!\n"
        f"Expected at '{local_blender}' or on system PATH.\n"
        "Run Step 1 to download Blender 4.2 LTS into dataset-generator/tools/."
    )


def find_host_python() -> str:
    """Finds active python binary with cv2/matplotlib installed."""
    venv_py = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    if os.path.isfile(venv_py) and os.access(venv_py, os.X_OK):
        return venv_py
    return sys.executable


def ensure_venue_scene(blender_bin: str, rebuild: bool = False) -> str:
    """Ensures mall_scene.blend exists; builds it via modular_mall_builder if missing or requested."""
    blend_path = os.path.join(SCRIPT_DIR, "venue", "mall_scene.blend")
    floor_plan_path = os.path.join(SCRIPT_DIR, "venue", "mall_floor_plan.json")
    build_script = os.path.join(SCRIPT_DIR, "venue", "modular_mall_builder.py")

    if not os.path.exists(blend_path) or rebuild:
        print("[Generator] Building High-Fidelity 3D Mall Venue in Blender...")
        cmd = [
            blender_bin, "-b", "-P", build_script, "--",
            "--floor-plan", floor_plan_path,
            "--output", blend_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            print(result.stdout)
            raise RuntimeError(f"Failed to build 3D venue scene! (Code {result.returncode})")
        print("[Generator] High-Fidelity 3D Mall Scene built successfully.")

    return blend_path


def ensure_floor_plan_map(host_python: str, force: bool = False):
    """Ensures 2D CAD floor plan map with camera FOVs is generated."""
    map_png = os.path.join(SCRIPT_DIR, "output", "floor_plan_with_cameras.png")
    map_script = os.path.join(SCRIPT_DIR, "tools", "export_floor_plan_map.py")
    plan_json = os.path.join(SCRIPT_DIR, "venue", "mall_floor_plan.json")

    if not os.path.exists(map_png) or force:
        print("[Generator] Exporting Architectural 2D Floor Plan with CCTV FOVs...")
        cmd = [
            host_python, map_script,
            "--plan", plan_json,
            "--output", map_png
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode != 0:
            print(f"[!] Warning: Could not generate floor plan map: {res.stdout}")
        else:
            print(f"[Generator] Floor plan map ready at: {map_png}")


def resolve_scenario_path(scenario_arg: str) -> str:
    """Resolves scenario name or path."""
    if os.path.isfile(scenario_arg):
        return os.path.abspath(scenario_arg)

    candidate = os.path.join(SCRIPT_DIR, "scenarios", f"{scenario_arg}.json")
    if os.path.isfile(candidate):
        return candidate

    candidate_direct = os.path.join(SCRIPT_DIR, "scenarios", scenario_arg)
    if os.path.isfile(candidate_direct):
        return candidate_direct

    available = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(SCRIPT_DIR, "scenarios")) if f.endswith(".json")]
    raise FileNotFoundError(
        f"Scenario '{scenario_arg}' not found!\n"
        f"Available scenarios: {', '.join(available)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="CrowdSafe Synthetic Dataset & Surveillance Video Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        default="rush_hour",
        help="Scenario name (normal_flow, rush_hour, bottleneck, stampede, evacuation) or JSON path"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames to generate (defaults to duration in scenario JSON)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Simulation & video playback framerate"
    )
    parser.add_argument(
        "--cameras",
        default="all",
        help="Cameras to render: 'all' or comma-separated list (e.g., 'CAM-01,CAM-02')"
    )
    parser.add_argument(
        "--mode",
        default="clean",
        choices=["clean", "cctv"],
        help="Visual render mode: 'clean' (sharp, crystal-clear 3D) or 'cctv' (subtle digital IP camera optics)"
    )
    parser.add_argument(
        "--character-style",
        default="realistic",
        choices=["realistic", "capsule"],
        help="Humanoid avatar model style: 'realistic' (anatomical, textured, walking kinematics) or 'capsule'"
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=True,
        help="Automatically encode MP4 surveillance video(s) with authentic CCTV OSD"
    )
    parser.add_argument(
        "--no-video",
        dest="video",
        action="store_false",
        help="Skip automatic video encoding"
    )
    parser.add_argument(
        "--tracking-video",
        action="store_true",
        help="Also export ground-truth debug tracking video overlaying bounding boxes and head dots"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Custom output directory (default: dataset-generator/output/<scenario>_<mode>)"
    )
    parser.add_argument(
        "--rebuild-venue",
        action="store_true",
        help="Force rebuild of mall_scene.blend before generating dataset"
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    blender_bin = find_blender_binary()
    host_python = find_host_python()
    scenario_path = resolve_scenario_path(args.scenario)
    scenario_name = os.path.splitext(os.path.basename(scenario_path))[0]
    venue_blend = ensure_venue_scene(blender_bin, rebuild=args.rebuild_venue)
    ensure_floor_plan_map(host_python)

    if args.output:
        output_dir = os.path.abspath(args.output)
    else:
        output_dir = os.path.join(SCRIPT_DIR, "output", f"{scenario_name}_{args.mode}")

    run_script = os.path.join(SCRIPT_DIR, "scripts", "run_scenario.py")
    floor_plan = os.path.join(SCRIPT_DIR, "venue", "mall_floor_plan.json")

    cmd = [
        blender_bin, "-b", venue_blend,
        "-P", run_script, "--",
        "--scenario", scenario_path,
        "--floor-plan", floor_plan,
        "--output", output_dir,
        "--cameras", args.cameras,
        "--mode", args.mode,
        "--character-style", args.character_style,
        "--fps", str(args.fps)
    ]
    if args.frames is not None:
        cmd.extend(["--frames", str(args.frames)])

    print("================================================================")
    print("     CROWDSAFE SYNTHETIC DATASET & SURVEILLANCE VIDEO ENGINE    ")
    print("================================================================")
    print(f"Blender Binary:   {blender_bin}")
    print(f"Python Host:      {host_python}")
    print(f"Scenario:         {scenario_name} ({scenario_path})")
    print(f"Character Style:  {args.character_style.upper()}")
    print(f"Render Mode:      {args.mode.upper()}")
    print(f"Cameras:          {args.cameras}")
    print(f"Target FPS:       {args.fps} FPS")
    print(f"Output Dir:       {output_dir}")
    print("----------------------------------------------------------------")

    # Run Blender headlessly and stream output live
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in iter(proc.stdout.readline, ""):
        # Filter noisy Blender EEVEE frame render messages
        if "Rendering" in line and "samples" in line:
            continue
        if "Fra:" in line and "Mem:" in line:
            continue
        print(line, end="", flush=True)

    proc.stdout.close()
    return_code = proc.wait()

    if return_code != 0:
        print(f"\n[Error] Generation failed with exit code {return_code}")
        sys.exit(return_code)

    print("\n================================================================")
    print(f"[Success] Synthetic frames generated at: {output_dir}")

    # Encode MP4 surveillance videos if requested
    if args.video:
        print("\n[Video Factory] Encoding MP4 surveillance video streams with CCTV OSD...")
        make_video_script = os.path.join(SCRIPT_DIR, "tools", "make_video.py")
        video_cmd = [
            host_python, make_video_script,
            "--scenario-dir", output_dir,
            "--camera", args.cameras,
            "--fps", str(args.fps),
            "--output-dir", output_dir
        ]
        if args.tracking_video:
            video_cmd.append("--tracking")

        vid_res = subprocess.run(video_cmd, text=True)
        if vid_res.returncode == 0:
            print("[✓] Video encoding complete!")
        else:
            print(f"[!] Video encoding encountered an issue (Code {vid_res.returncode})")

    print("================================================================")
    print(f"[Done] Complete dataset & video package ready in:\n{output_dir}")
    print("================================================================")


if __name__ == "__main__":
    main()
