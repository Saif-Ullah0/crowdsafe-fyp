Here are the exact step-by-step phases we will execute to build and verify the dataset generator:

---

### Step 1: Install & Verify Blender 4.2 LTS on the 199 GB Volume

1. Create `dataset-generator/tools/` directory on your 199 GB NVMe drive.
2. Download the official, stable **Blender 4.2 LTS** Linux x64 release tarball.
3. Extract it into `dataset-generator/tools/blender/` (no `sudo` required).
4. Run headless smoke tests:
   - Check version: `blender --version`
   - Check bundled Python & NumPy: `import bpy, numpy; print(bpy.app.version_string)`
   - Test headless EEVEE rasterizer render to `/tmp/test_render.png`.

---

### Step 2: Define Floor Plan & Build the 3D Venue

1. Create `dataset-generator/venue/mall_floor_plan.json`:
   - 2D layout coordinates ($50\text{m} \times 30\text{m}$ mall floor).
   - Interior partitions, perimeter walls, and doorway cutouts.
   - Obstacles (food court tables, structural pillars) that agents steer around and that occlude CCTV line-of-sight.
   - Navigation graph (waypoints, corridors, exits, entrances) with pedestrian capacities.
   - Zone polygons for Fruin Level of Service tracking.
2. Build the venue with `dataset-generator/venue/modular_mall_builder.py`:
   - Reads `mall_floor_plan.json` headlessly inside Blender.
   - Builds the mall floor, walls, obstacles, storefronts, lighting, and CCTV fixtures.
   - Saves the scene to `dataset-generator/venue/mall_scene.blend`.
   - The host CLI builds the scene automatically if it is missing; use `--rebuild-venue` to regenerate it.

---

### Step 3: Implement the 3D Character Model & Domain Randomization

1. Use `dataset-generator/agents/realistic_human_library.py` for articulated, randomized humanoids, or select `--character-style capsule` to use the lightweight capsule models.
   - Height and clothing color are randomized across agents.
   - The realistic models use articulated limbs with walking gait synchronized to agent movement.

---

### Step 4: Implement the Helbing Social Force Model (SFM)

1. Create `dataset-generator/agents/crowd_physics.py`:
   - Pure NumPy implementation of Helbing's pedestrian dynamics:
     - **Goal Attraction Force**: Steers agents toward intermediate navigation graph waypoints.
     - **Pedestrian Repulsion Force**: Prevents agents from walking into each other (exponential personal space buffer).
     - **Wall & Obstacle Repulsion**: Deflects agents away from walls, door frames, and structural columns.
     - **Panic/Compression Physics**: Models dense crowd arching and slowing down at bottleneck exits.
     - **Orientation Alignment**: Automatically rotates the humanoid model to face its instantaneous direction of travel.

---

### Step 5: Virtual CCTV Array, Calibration & Ray-Cast Occlusion Culling

1. Create `dataset-generator/scripts/camera_calibration.py`:
   - Sets up 4 virtual CCTV cameras mounted at $4.0 - 5.0\text{m}$ height, tilted downward at $45^\circ$.
   - Computes mathematical camera parameters:
     - $3 \times 3$ Intrinsic Matrix ($K$) based on 6mm wide-angle lens, sensor dimensions, and $640 \times 480$ resolution.
     - $4 \times 4$ Extrinsic Matrix ($[R|t]$) aligned with OpenCV coordinate convention.
2. Create `dataset-generator/scripts/occlusion_tracker.py`:
   - Projects 3D head coordinates to 2D image pixels via `bpy_extras.object_utils.world_to_camera_view`.
   - Filters out any coordinates outside the $[0, W] \times [0, H]$ frame.
   - **Line-of-Sight Ray Casting**: Shoots a ray from camera optical center to agent head sphere using `scene.ray_cast`.
   - Marks agent as occluded if the ray hits a wall, pillar, or another person first.

---

### Step 6: Dual-Mode Rendering Pipeline (CCTV vs. Clean)

1. `dataset-generator/scripts/render_pipeline.py` supports two modes:
   - **Clean Mode (Default)** exports frames without optical degradation.
   - **CCTV Mode** adds subtle sensor noise, compression, wide-angle lens distortion, and exposure jitter.
   - Controlled via `--mode clean` or `--mode cctv`.

---

### Step 7: Scenario Library & Master Orchestrator CLI

1. Create `dataset-generator/scenarios/` with pre-configured templates:
   - `normal_flow.json`: 40 agents, calm wandering, baseline low density.
   - `rush_hour.json`: 120 agents, corridor directional traffic, moderate density.
   - `bottleneck.json`: 80 agents, single exit congestion, Fruin WARNING/DANGER.
   - `stampede.json`: 200 agents, panic speed ($3.5\text{ m/s}$), critical crush risk.
   - `evacuation.json`: 150 agents, orderly routing to nearest emergency exits.
2. Create `dataset-generator/scripts/run_scenario.py` (Blender runner).
3. Use `dataset-generator/generate.py` (Host CLI wrapper):
   - Example command:
     ```bash
     python3 dataset-generator/generate.py --scenario rush_hour --frames 100 --fps 10 --mode cctv --output output/rush_hour/
     ```
   - Select cameras with `--cameras all` or a comma-separated list such as `--cameras CAM-01,CAM-02`.
   - Select `--character-style realistic` or `--character-style capsule`.
   - MP4 surveillance videos are encoded by default; use `--no-video` to skip them. Add `--tracking-video` to export tracking-overlay videos.
   - The CLI handles Blender headless invocation, progress logging, and JSON annotations, PNG frames, and MP4 video output.

---

### Step 8: Verification & CrowdSafe Integration

1. Create `dataset-generator/tools/verify_dataset.py`:
   - Overlays green dots on visible heads and flags occluded ones.
   - Visual inspection check to guarantee 100% ground-truth accuracy.
2. Ingest generated synthetic CCTV frames directly into CrowdSafe MVP:
   ```bash
   python mvp/app.py --source dataset-generator/output/rush_hour/cam_02/ --model fast_sim --fps 1.0
   ```
   Verify that the MVP streams the synthetic camera footage, estimates density, and triggers risk alerts seamlessly.

---

When you are ready to begin, let me know and we will execute Step 1!
