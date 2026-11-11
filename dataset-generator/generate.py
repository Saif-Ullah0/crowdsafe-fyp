#!/usr/bin/env python3
"""
CrowdSafe Synthetic Dataset Generator — Master Host CLI Runner
Usage:
  python dataset-generator/generate.py --scenario rush_hour --frames 50 --mode cctv
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


def ensure_venue_scene(blender_bin: str, rebuild: bool = False) -> str:
    """Ensures mall_scene.blend exists; builds it if missing or requested."""
    blend_path = os.path.join(SCRIPT_DIR, "venue", "mall_scene.blend")
    floor_plan_path = os.path.join(SCRIPT_DIR, "venue", "mall_floor_plan.json")
    build_script = os.path.join(SCRIPT_DIR, "scripts", "build_venue.py")

    if not os.path.exists(blend_path) or rebuild:
        print("[Generator] Building 3D venue scene from floor plan...")
        cmd = [
            blender_bin, "-b", "-P", build_script, "--",
            "--floor-plan", floor_plan_path,
            "--output", blend_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            print(result.stdout)
            raise RuntimeError(f"Failed to build 3D venue scene! (Code {result.returncode})")
        print("[Generator] 3D venue scene built successfully.")

    return blend_path


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
        description="CrowdSafe Synthetic Dataset Generator",
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
        "--cameras",
        default="all",
        help="Cameras to render: 'all' or comma-separated list (e.g., 'CAM-01,CAM-02')"
    )
    parser.add_argument(
        "--mode",
        default="cctv",
        choices=["cctv", "clean"],
        help="Visual render mode: 'cctv' (sensor noise, lens distortion, compression) or 'clean' (raw 3D)"
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

    blender_bin = find_blender_binary()
    scenario_path = resolve_scenario_path(args.scenario)
    scenario_name = os.path.splitext(os.path.basename(scenario_path))[0]
    venue_blend = ensure_venue_scene(blender_bin, rebuild=args.rebuild_venue)

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
        "--mode", args.mode
    ]
    if args.frames is not None:
        cmd.extend(["--frames", str(args.frames)])

    print("================================================================")
    print("           CROWDSAFE SYNTHETIC DATASET GENERATOR                ")
    print("================================================================")
    print(f"Blender Binary: {blender_bin}")
    print(f"Scenario:       {scenario_name} ({scenario_path})")
    print(f"Render Mode:    {args.mode.upper()}")
    print(f"Cameras:        {args.cameras}")
    print(f"Output Dir:     {output_dir}")
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
        print(line, end="")

    proc.stdout.close()
    return_code = proc.wait()

    if return_code != 0:
        print(f"\n[Error] Generation failed with exit code {return_code}")
        sys.exit(return_code)

    print("\n================================================================")
    print(f"[Success] Dataset successfully generated at: {output_dir}")
    print("================================================================")


if __name__ == "__main__":
    main()
