# CrowdSafe Synthetic Simulation Pipeline — Complete Engineering Blueprint

---

## 1. What We Are Building and Why

### The Goal

Build a **fully scriptable 3D simulation environment** in Blender that generates synthetic CCTV surveillance footage of crowds inside a building, with **perfect ground truth annotations** — so we never need to manually label a single frame.

Every rendered frame comes with:
- The exact head coordinates of every person visible in the camera
- The exact person count per zone
- The exact density ($people/m^2$) per corridor
- The camera's known position on the building floor plan

This data feeds directly into CrowdSafe's density estimation models for training, benchmarking, and the evacuation routing graph.

### The Design Philosophy: Ugly is Beautiful

> [!IMPORTANT]
> **We are NOT making a cinematic Blender scene.**
>
> We are building a **data factory.** The density estimation model (MCNN / CSRNet) sees the world through a $640 \times 480$ CCTV camera mounted 4 meters high, looking down at a 45° angle. At that resolution and distance, a real human being is a **20×40 pixel blob**. The model does not care about facial features, finger joints, shoe laces, or realistic hair physics.
>
> What the model DOES care about:
> - Human-shaped silhouettes at the correct physical scale
> - Correct perspective distortion from overhead cameras
> - Varying crowd densities and spatial distributions
> - Enough visual diversity that it doesn't overfit to one "look"

**Our characters will be capsule-bodied, sphere-headed, flat-shaded humanoids with ~100 polygons each.** They will look like mannequins. That is exactly right.

---

## 2. Your Hardware Assessment

We checked your system. Here is what we're working with:

| Resource | Value | Impact |
| :--- | :--- | :--- |
| **CPU** | Intel i7-8650U (8 threads @ 1.90 GHz) | Adequate for Blender Eevee rendering. Not fast for Cycles. |
| **RAM** | 32 GB (14 GB available) | Excellent. Can handle 200+ low-poly agents in a scene. |
| **GPU** | Intel UHD 620 (Integrated) | No dedicated GPU. We **must** use Eevee (rasterizer) or Workbench, not Cycles (path tracer). This is fine — Eevee renders simple scenes in <1 second per frame. |
| **Disk** | 171 GB free on project drive | More than enough for thousands of rendered frames. |
| **Blender** | **Not installed** | Must install. ~250 MB download. |

> [!NOTE]
> **Render Engine Decision: Eevee (Not Cycles)**
>
> Your machine has no dedicated GPU. Blender's **Eevee** engine is a real-time rasterizer (like a game engine) — it renders simple low-poly scenes in under 1 second per frame on CPU/integrated GPU. **Cycles** is a physically accurate path tracer that would take 30-60 seconds per frame on your hardware. Since we intentionally want low-fidelity, fast output, Eevee is the correct and optimal choice.

---

## 3. Prerequisites & Installation

### Step 1: Install Blender

```bash
# Option A: APT (Ubuntu/Debian) — may be slightly older version
sudo apt install blender

# Option B: Snap (Latest stable, recommended)
sudo snap install blender --classic

# Option C: Direct download from blender.org (Latest 4.x)
wget https://mirror.clarkson.edu/blender/release/Blender4.4/blender-4.4.0-linux-x64.tar.xz
tar -xf blender-4.4.0-linux-x64.tar.xz
# Then run: ./blender-4.4.0-linux-x64/blender
```

### Step 2: Verify Installation

```bash
# Check version (should be 3.6+ or 4.x)
blender --version

# Verify headless mode works
blender -b --python-expr "import bpy; print('Blender headless OK:', bpy.app.version_string)"

# Verify Python inside Blender
blender -b --python-expr "import sys; print('Python:', sys.version)"
```

**Expected output:**
```
Blender 4.x.x
Blender headless OK: 4.x.x
Python: 3.11.x (or 3.12.x)
```

### Step 3: Install Python Dependencies Inside Blender

Blender ships its own embedded Python interpreter. We need `numpy` (usually pre-installed) and optionally `mathutils` extensions:

```bash
# Blender's bundled Python already includes numpy and mathutils
# Verify:
blender -b --python-expr "import numpy; print('numpy:', numpy.__version__)"
```

### Step 4: Verify Eevee Rendering Works Headlessly

```bash
blender -b --python-expr "
import bpy
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
bpy.context.scene.render.resolution_x = 640
bpy.context.scene.render.resolution_y = 480
bpy.context.scene.render.filepath = '/tmp/test_render.png'
bpy.ops.render.render(write_still=True)
print('Render test passed. Output: /tmp/test_render.png')
"
```

If this produces `/tmp/test_render.png` without errors, the pipeline is ready.

---

## 4. Project Directory Structure

```text
crowdsafe-fyp/
├── mvp/                          # Existing CrowdSafe MVP application
│   ├── app.py
│   ├── core/
│   └── ...
│
└── simulation/                   # NEW: Synthetic Simulation Pipeline
    ├── README.md                 # This document (condensed)
    ├── venue/
    │   ├── mall_floor_plan.json  # 2D floor plan definition (nodes, edges, walls)
    │   └── mall_scene.blend      # Blender scene file (3D venue + cameras)
    │
    ├── agents/
    │   └── humanoid_template.py  # Script to generate low-poly capsule humanoids
    │
    ├── scripts/
    │   ├── build_venue.py        # Blender script: creates 3D mall from JSON floor plan
    │   ├── place_cameras.py      # Blender script: positions virtual CCTV cameras
    │   ├── spawn_crowd.py        # Blender script: spawns N agents with behaviors
    │   ├── render_dataset.py     # Blender script: headless batch render + annotation export
    │   └── run_scenario.py       # Master orchestrator: runs a complete scenario end-to-end
    │
    ├── scenarios/
    │   ├── normal_flow.json      # 40 people, calm walking
    │   ├── rush_hour.json        # 120 people, fast directional flow
    │   ├── bottleneck.json       # 80 people, exit corridor jammed
    │   └── stampede.json         # 200 people, panic rush toward one exit
    │
    ├── output/                   # Generated dataset (gitignored)
    │   ├── cam_01/
    │   │   ├── frame_0001.png
    │   │   ├── frame_0002.png
    │   │   └── ...
    │   ├── cam_02/
    │   ├── cam_03/
    │   └── annotations/
    │       ├── frame_0001.json
    │       └── ...
    │
    └── tools/
        └── verify_dataset.py     # Standalone script to visually verify annotations
```

---

## 5. The Venue: A Low-Poly Mall Floor

### Floor Plan Definition (`mall_floor_plan.json`)

We define the building as a simple 2D data structure. No CAD software needed — just coordinates in meters.

```
Mall Floor Plan (Top-Down, 50m × 30m)
┌──────────────────────────────────────────────────────┐
│                                                      │
│   [EXIT-A]                              [EXIT-B]     │
│      ║                                     ║         │
│   ┌──╨──┐    ┌───────────────────┐    ┌────╨───┐     │
│   │     │    │                   │    │        │     │
│   │  A  │    │    FOOD COURT     │    │   B    │     │
│   │     ├────┤     (Open)        ├────┤        │     │
│   │     │    │                   │    │        │     │
│   └──┬──┘    └────────┬──────────┘    └────┬───┘     │
│      │                │                    │         │
│   ═══╪════════════════╪════════════════════╪═══      │
│      │   MAIN CORRIDOR (4m wide)           │         │
│   ═══╪════════════════╪════════════════════╪═══      │
│      │                │                    │         │
│   ┌──┴──┐    ┌────────┴──────────┐    ┌────┴───┐     │
│   │     │    │                   │    │        │     │
│   │  C  │    │    ATRIUM /       │    │   D    │     │
│   │     ├────┤    ESCALATOR      ├────┤        │     │
│   │     │    │     AREA          │    │        │     │
│   └──┬──┘    └────────┬──────────┘    └────┬───┘     │
│      ║                │                    ║         │
│   [EXIT-C]        [ENTRANCE]           [EXIT-D]      │
│                      ║                               │
└──────────────────────╨───────────────────────────────┘
```

**Navigation Graph Nodes:**

| Node ID | Name | Position (x, y) meters | Type |
| :--- | :--- | :--- | :--- |
| N0 | Main Entrance | (25, 0) | entrance |
| N1 | Corridor West | (8, 15) | junction |
| N2 | Corridor Center | (25, 15) | junction |
| N3 | Corridor East | (42, 15) | junction |
| N4 | Food Court | (25, 22) | open_area |
| N5 | Atrium | (25, 8) | open_area |
| N6 | Exit A | (8, 30) | exit |
| N7 | Exit B | (42, 30) | exit |
| N8 | Exit C | (8, 0) | exit |
| N9 | Exit D | (42, 0) | exit |

**Navigation Graph Edges:**

| Edge | Distance (m) | Width (m) | Max Capacity |
| :--- | :--- | :--- | :--- |
| N0 ↔ N2 | 15 | 5.0 | 150 |
| N1 ↔ N2 | 17 | 4.0 | 100 |
| N2 ↔ N3 | 17 | 4.0 | 100 |
| N1 ↔ N4 | 10 | 3.0 | 60 |
| N2 ↔ N4 | 7 | 6.0 | 120 |
| N3 ↔ N4 | 10 | 3.0 | 60 |
| N1 ↔ N5 | 10 | 3.0 | 60 |
| N2 ↔ N5 | 7 | 6.0 | 120 |
| N3 ↔ N5 | 10 | 3.0 | 60 |
| N1 ↔ N6 | 15 | 3.0 | 80 |
| N3 ↔ N7 | 15 | 3.0 | 80 |
| N1 ↔ N8 | 15 | 3.0 | 80 |
| N3 ↔ N9 | 15 | 3.0 | 80 |

### 3D Venue Construction

The `build_venue.py` Blender script reads `mall_floor_plan.json` and:

1. **Creates the floor:** A flat 50m × 30m plane with tile material (dark gray checkerboard).
2. **Extrudes walls:** 3-meter-tall flat-shaded walls along the boundary and interior partitions. Material: concrete gray.
3. **Cuts doorways:** Boolean operations to carve 2m-wide door openings at connection points.
4. **Marks exits:** Green-colored exit signs (simple text planes) above exit doorways.
5. **Adds overhead lighting:** 4-6 area lights simulating mall fluorescent ceiling lights. Flat, even illumination.

**Total polygon count for the entire venue: ~2,000-5,000 faces.** Renders in <0.5 seconds.

---

## 6. The Characters: Minimal Capsule Humanoids

### Design Specification

```
    ●          ← Sphere Head (radius: 0.12m, ~32 faces)
   ╱│╲
  ╱ │ ╲
 ╱  │  ╲       ← Cylinder/Capsule Torso (height: 0.6m, radius: 0.18m, ~24 faces)
 ╲  │  ╱
  ╲ │ ╱
   ╲│╱
    │          ← Cylinder Legs (single merged column, height: 0.8m, radius: 0.12m, ~16 faces)
   ─┴─
```

**Character Specifications:**

| Property | Value | Why |
| :--- | :--- | :--- |
| Total height | 1.7m (± 0.15m randomized) | Average adult human |
| Head | UV Sphere, 0.12m radius, 3 subdivisions (~32 faces) | Visible as a dot from 4m CCTV height |
| Torso | Cylinder, 0.18m radius, 0.6m height (~24 faces) | Body mass silhouette |
| Legs | Cylinder, 0.12m radius, 0.8m height (~16 faces) | Standing height filler |
| Total polygons per agent | **~72 faces** | Allows 300+ agents at <1s render time |
| Shading | **Flat shaded** (no smooth normals) | Speed; CCTV doesn't capture smooth skin |
| Material | Random solid color from a palette of 15 colors | Visual diversity; prevents model overfitting to one color |
| Animation | Simple Z-axis bobbing (±0.03m sine wave) | Simulates walking gait at minimal compute cost |
| Face / Hair / Hands | **None** | Not visible at CCTV resolution |

### Color Palette (15 Solid Materials)

```python
HUMAN_COLORS = [
    (0.15, 0.15, 0.15),  # Black jacket
    (0.85, 0.85, 0.85),  # White shirt
    (0.7, 0.1, 0.1),     # Red sweater
    (0.1, 0.2, 0.6),     # Blue jeans
    (0.4, 0.3, 0.2),     # Brown coat
    (0.2, 0.5, 0.2),     # Green hoodie
    (0.9, 0.75, 0.1),    # Yellow safety vest
    (0.5, 0.5, 0.5),     # Gray uniform
    (0.1, 0.1, 0.3),     # Dark navy
    (0.6, 0.3, 0.1),     # Orange
    (0.4, 0.1, 0.4),     # Purple
    (0.8, 0.6, 0.4),     # Beige / khaki
    (0.05, 0.05, 0.05),  # Near black
    (0.3, 0.3, 0.6),     # Slate blue
    (0.7, 0.4, 0.3),     # Terracotta
]
```

### Why This Level of Detail is Sufficient

At CCTV resolution ($640 \times 480$) from 4 meters height at 45° angle:
- A full human body occupies roughly **15-25 pixels wide × 30-50 pixels tall**.
- The head alone is **5-8 pixels in diameter**.
- At this pixel budget, a photorealistic MetaHuman and our 72-polygon capsule are **visually indistinguishable** to the density estimation model.
- The model learns spatial density patterns (how blobs cluster), not individual anatomy.

---

## 7. The Camera System: Virtual CCTV Array

### Camera Placement

We place 4 virtual cameras simulating real mall CCTV installation:

| Camera ID | Position (x, y, z) meters | Look-At Point | Role |
| :--- | :--- | :--- | :--- |
| CAM-01 | (8, 15, 4.5) | (8, 15, 0) | Watches west corridor + Zone A/C doorways |
| CAM-02 | (25, 22, 5.0) | (25, 15, 0) | Watches food court + main corridor center |
| CAM-03 | (42, 15, 4.5) | (42, 15, 0) | Watches east corridor + Zone B/D doorways |
| CAM-04 | (25, 3, 4.0) | (25, 10, 0) | Watches main entrance + atrium |

### Camera Properties

| Property | Value | Real-World Equivalent |
| :--- | :--- | :--- |
| Lens focal length | 6mm | Wide-angle CCTV lens |
| Sensor size | 6.17mm × 4.55mm (1/2.7") | Standard CCTV sensor |
| Resolution | 640 × 480 | Standard DVR output |
| Clipping near/far | 0.1m / 100m | — |
| Mount height | 4.0 - 5.0m | Typical mall ceiling mount |
| Tilt angle | ~45° downward | Standard surveillance angle |

### What We Record Per Camera

For each camera, per frame, we automatically export:
1. **RGB Image:** The rendered frame as seen by that camera (PNG, $640 \times 480$).
2. **Head Coordinates:** A list of $(x_{pixel}, y_{pixel})$ for every person whose head is visible in that camera's frame.
3. **Person Count:** Total visible persons in this camera's view.
4. **Zone Density:** Density ($people/m^2$) for each floor plan zone visible in this camera.

---

## 8. Crowd Behavior System

### Movement Model

Agents navigate using a simple **waypoint system** (not full NavMesh — keeping it lightweight):

1. Each agent is assigned a **route**: an ordered list of 2-5 waypoints (nodes from the floor plan graph).
2. On each frame, the agent moves toward its next waypoint at a speed of 1.0 - 1.5 m/s (normal walking) or 2.5 - 4.0 m/s (panic/rush).
3. When reaching a waypoint, the agent targets the next one. At the final waypoint, it either loops back or despawns.
4. Agents have a small random lateral offset (±0.5m) from the centerline to avoid unnaturally perfect single-file lines.
5. Simple collision avoidance: if two agents are within 0.4m, they deflect slightly.

### Scenario Configurations

Each scenario is a JSON file defining:

```json
{
  "name": "rush_hour",
  "description": "Heavy foot traffic through main corridor during peak hours",
  "num_agents": 120,
  "spawn_zones": ["N0", "N6", "N7", "N8", "N9"],
  "route_patterns": [
    {"from": "N0", "to": "N6", "weight": 0.3},
    {"from": "N0", "to": "N7", "weight": 0.3},
    {"from": "N8", "to": "N4", "weight": 0.2},
    {"from": "N9", "to": "N4", "weight": 0.2}
  ],
  "speed_range": [1.0, 1.6],
  "duration_frames": 300,
  "density_target": "high_corridor"
}
```

### Pre-Built Scenario Library

| Scenario | Agents | Speed | Behavior | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `normal_flow` | 40 | 1.0-1.3 m/s | Random wandering between zones | Baseline LOW density testing |
| `rush_hour` | 120 | 1.2-1.6 m/s | Directional flow entrance→exits | MODERATE density corridors |
| `bottleneck` | 80 | 0.8-1.2 m/s | All agents converge on Exit A | HIGH density at single chokepoint |
| `stampede` | 200 | 2.5-4.0 m/s | Panic rush, agents cluster near exit | CRITICAL density, crush risk |
| `evacuation` | 150 | 1.5-2.5 m/s | All agents route to nearest exit | Tests evacuation graph routing |
| `event_end` | 180 | 1.0-1.8 m/s | Gradual exit from food court | Natural density wave propagation |

---

## 9. The Rendering & Annotation Pipeline

### Execution Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  $ blender -b venue/mall_scene.blend -P scripts/run_scenario.py │
│    -- --scenario scenarios/rush_hour.json --frames 300           │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │ 1. Load Venue Scene │  (mall_scene.blend: walls, floors, lights)
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ 2. Read Scenario    │  (rush_hour.json: 120 agents, routes, speeds)
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ 3. Spawn Agents     │  (Create 120 capsule humanoids, assign colors/routes)
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 4. Frame Loop (300 iterations):                             │
    │    a) Move all agents one step along their routes            │
    │    b) Apply walking bobble animation                         │
    │    c) FOR each camera (CAM-01..CAM-04):                     │
    │       i)   Set active camera                                │
    │       ii)  Render frame → output/cam_XX/frame_NNNN.png      │
    │       iii) Project all agent head positions to camera pixels │
    │       iv)  Filter: keep only heads visible in camera frustum│
    │       v)   Write annotation JSON                            │
    │    d) Print progress: "Frame 42/300 rendered (4 cameras)"   │
    └─────────────────────────────────────────────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │ 5. Export Summary    │  (total frames, total annotations, render time)
    └─────────────────────┘
```

### Annotation Format (Per Frame)

```json
{
  "frame_id": 42,
  "camera_id": "CAM-02",
  "image_path": "output/cam_02/frame_0042.png",
  "image_width": 640,
  "image_height": 480,
  "total_visible_count": 37,
  "head_coordinates": [
    {"id": 5, "x_pixel": 312, "y_pixel": 198, "x_world": 22.4, "y_world": 16.1},
    {"id": 12, "x_pixel": 445, "y_pixel": 267, "x_world": 28.7, "y_world": 14.3},
    ...
  ],
  "zone_densities": {
    "zone_corridor_west": {"count": 12, "area_m2": 40.0, "density_per_m2": 0.30},
    "zone_food_court": {"count": 18, "area_m2": 80.0, "density_per_m2": 0.23},
    "zone_corridor_east": {"count": 7, "area_m2": 40.0, "density_per_m2": 0.18}
  },
  "camera_position": {"x": 25.0, "y": 22.0, "z": 5.0},
  "camera_rotation": {"pitch": -45.0, "yaw": 0.0, "roll": 0.0}
}
```

### Render Performance Estimate

| Component | Time Per Frame | Basis |
| :--- | :--- | :--- |
| Eevee render (640×480, low-poly scene, 100 agents) | ~0.3 - 0.8 seconds | Intel UHD 620, Eevee rasterizer |
| Agent movement computation | ~0.01 seconds | Pure Python math |
| Head projection + annotation export | ~0.02 seconds | Matrix multiplication |
| **Total per frame (4 cameras)** | **~1.5 - 3.5 seconds** | |
| **300 frames × 4 cameras = 1,200 renders** | **~8 - 18 minutes** | |

For a single 5-minute scenario, you generate **1,200 annotated frames in under 20 minutes**. Run 6 scenarios overnight → **7,200 perfectly labeled training frames**.

---

## 10. Integration with CrowdSafe MVP

### How Synthetic Data Flows Into the Pipeline

```
simulation/output/cam_01/frame_0001.png  ──►  mvp/app.py --source simulation/output/cam_01/
                                                   │
                                                   ▼
                                           DensityEngine.predict(frame)
                                                   │
                                                   ▼
                                           SpatialAnalyzer.analyze(density_map)
                                                   │
                                                   ▼
                                    Compare predicted count vs. annotation ground truth
                                                   │
                                                   ▼
                                        MAE = |predicted - actual| per frame
```

The `simulation/output/cam_XX/` directories are standard image directories. CrowdSafe's `DVRStreamer` already supports image directories as input — so **zero additional code** is needed to ingest synthetic data.

For model evaluation, a simple script compares predicted counts against the annotation JSON:

```python
# Pseudocode
for frame in annotations:
    predicted = density_engine.predict(load_image(frame.image_path))
    actual = frame.total_visible_count
    error = abs(predicted.total_count - actual)
    mae_sum += error
```

---

## 11. Verification Checklist

### Stage 1: Environment Setup
- [ ] Blender installed and `blender --version` returns 3.6+
- [ ] Headless render test produces `/tmp/test_render.png`
- [ ] `blender -b --python-expr "import numpy"` succeeds

### Stage 2: Venue Construction
- [ ] `build_venue.py` produces a `.blend` file with visible floor, walls, and doorways
- [ ] Opening the `.blend` in Blender GUI shows a recognizable mall floor plan from top-down view
- [ ] All 4 cameras are placed and each shows a different corridor/zone when selected

### Stage 3: Character Generation
- [ ] A single capsule humanoid spawns at (0, 0, 0) and is visually a head-on-body shape
- [ ] 100 agents spawn without Blender crashing or exceeding 2 GB RAM
- [ ] Each agent has a different randomly assigned solid color

### Stage 4: Crowd Behavior
- [ ] Agents move along waypoints across 10 frames without teleporting or clipping through walls
- [ ] Speed is visually consistent with walking (1 frame = 1 second → agent moves ~1.2m per frame)
- [ ] Agents do not stack perfectly on top of each other (lateral offset working)

### Stage 5: Rendering Pipeline
- [ ] `run_scenario.py` executes headlessly and produces PNG files in `output/cam_XX/`
- [ ] Rendered frames show a CCTV-like overhead perspective of capsule humans on a mall floor
- [ ] Annotation JSON files contain head coordinates

### Stage 6: Annotation Accuracy
- [ ] Run `verify_dataset.py` which overlays red dots on head coordinates over the rendered frame
- [ ] Visual inspection confirms: every dot lands on a head, no heads are missed, no phantom dots exist
- [ ] Count in annotation JSON matches the number of visible agents in the frame (manual spot-check on 5 frames)

### Stage 7: CrowdSafe Integration
- [ ] `mvp/app.py --source simulation/output/cam_02/ --fps 1.0` ingests and processes synthetic frames
- [ ] Density maps and zone risk assessments are generated from the synthetic input
- [ ] MAE between predicted and ground truth count is calculated and logged

---

## 12. How to Modify the System

### Change the Venue Layout
1. Edit `simulation/venue/mall_floor_plan.json` — change node coordinates, add/remove corridors, resize zones.
2. Re-run `blender -b -P scripts/build_venue.py` to regenerate the 3D venue.
3. All cameras, agents, and scenarios automatically adapt to the new geometry.

### Add More Cameras
1. Add a new entry to the `cameras` array in `mall_floor_plan.json` with position, rotation, and zone coverage.
2. Re-run `place_cameras.py`. The render loop automatically iterates over all cameras.

### Change Crowd Size or Behavior
1. Edit any scenario JSON in `simulation/scenarios/` — change `num_agents`, `speed_range`, or `route_patterns`.
2. Re-run the render pipeline. No code changes needed.

### Generate More Visual Diversity (Domain Randomization)
1. Add more colors to `HUMAN_COLORS` palette.
2. Randomize agent height (already built in: ±0.15m).
3. Randomize lighting intensity and color temperature per scenario.
4. Add environmental variations: wet floor reflections (Eevee supports this cheaply), different ceiling light layouts.

### Switch to a Different Building Type
1. Create a new `transit_station_floor_plan.json` or `stadium_floor_plan.json` with appropriate geometry.
2. The entire pipeline is **venue-agnostic** — it reads geometry from JSON, not hardcoded coordinates.

---

## 13. Complete Command Reference

```bash
# ──────────────────────────────────────────────────────
# ONE-TIME SETUP
# ──────────────────────────────────────────────────────

# Install Blender
sudo snap install blender --classic

# Verify installation
blender --version

# ──────────────────────────────────────────────────────
# BUILD THE VENUE (once, or after floor plan changes)
# ──────────────────────────────────────────────────────

cd /run/media/mobeen/977e7ff6-efb4-4c0c-80f3-6ebc7ecfbbfb/FYP/crowdsafe-fyp/

# Generate 3D venue from floor plan JSON
blender -b -P simulation/scripts/build_venue.py -- \
  --floor-plan simulation/venue/mall_floor_plan.json \
  --output simulation/venue/mall_scene.blend

# ──────────────────────────────────────────────────────
# GENERATE A DATASET (run per scenario)
# ──────────────────────────────────────────────────────

# Normal flow scenario (40 people, 300 frames, 4 cameras)
blender -b simulation/venue/mall_scene.blend \
  -P simulation/scripts/run_scenario.py -- \
  --scenario simulation/scenarios/normal_flow.json \
  --frames 300 \
  --output simulation/output/normal_flow/

# Rush hour scenario
blender -b simulation/venue/mall_scene.blend \
  -P simulation/scripts/run_scenario.py -- \
  --scenario simulation/scenarios/rush_hour.json \
  --frames 300 \
  --output simulation/output/rush_hour/

# ──────────────────────────────────────────────────────
# VERIFY ANNOTATIONS
# ──────────────────────────────────────────────────────

# Overlay ground truth dots on rendered frames for visual check
.venv/bin/python simulation/tools/verify_dataset.py \
  --images simulation/output/rush_hour/cam_02/ \
  --annotations simulation/output/rush_hour/annotations/ \
  --output simulation/output/rush_hour/verified/

# ──────────────────────────────────────────────────────
# FEED INTO CROWDSAFE MVP
# ──────────────────────────────────────────────────────

cd mvp/
.venv/bin/python app.py \
  --source ../simulation/output/rush_hour/cam_02/ \
  --model fast_sim \
  --fps 1.0
```

---

## 14. Expected Output: What It Looks Like

### From the Render (CAM-02, Rush Hour, Frame 150)

```
┌─────────────────────────────────────────────────────────────┐
│  CAM-02 [FOOD COURT OVERVIEW]  |  2026-11-07 14:30:00      │
│                                                              │
│          ┌───────────────────────────────────┐               │
│          │          FOOD COURT               │               │
│          │    ●  ●      ●                    │               │
│          │  ●     ●  ●    ●  ●               │               │
│          │    ●       ● ●      ●             │               │
│          └──────────────┬────────────────────┘               │
│                         │                                    │
│    ● ● ●  ●  ●  ●  ●  ●│●  ●  ●  ●  ●  ● ●                │
│    ●  ●   ●  ●   ● ● ● │ ●  ●  ●  ●   ●                    │
│       MAIN CORRIDOR     │    (Crowded)                       │
│                         │                                    │
│          ┌──────────────┴────────────────────┐               │
│          │          ATRIUM                   │               │
│          │       ●     ●    ●                │               │
│          │                                   │               │
│          └───────────────────────────────────┘               │
│                                                              │
│  (Each ● = one capsule humanoid seen from above)             │
└─────────────────────────────────────────────────────────────┘
```

The actual rendered frame will show flat-shaded colored capsule bodies on a dark tiled floor viewed from a high-angle CCTV perspective. Simple, clean, and exactly what a density model needs to train on.
