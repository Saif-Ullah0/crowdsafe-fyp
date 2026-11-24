# Design 4: Controlled Transit Interchange
## Plus Walkway Floor Plan Catalog & Specialized Visual Variants

---

## 1. Executive Summary & Floor Plan Variants

This catalog provides the complete architectural, tracking, and dashboard visual assets for **Design 4: Controlled Transit & Security Interchange** ($38.0\text{m} \times 38.0\text{m}$).

Four specialized 2D floor plan renderings are now available for direct use in web dashboards, tracking pipelines, and architectural presentations:

| Variant | Filename | Description / Application Role |
| :--- | :--- | :--- |
| **1. Clean Architectural Base** | `design_4_clean_base.png` / `.svg` | **Pure architectural CAD floor plan without camera FOV projections.** Used as a clean background to project custom camera covered regions or zone analytics. |
| **2. Human Location Dots** | `design_4_human_locations.png` / `.svg` | **Real-time pedestrian tracking map.** Displays simulated commuter centroid dots (142 persons), velocity vectors, roundabout circulation, and bottleneck density clusters. |
| **3. Tactical Game Map** | `design_4_tactical_gamemap.png` / `.svg` | **Realistic video-game minimap & SOC dashboard style.** High-contrast dark mode (`#070B14`) with cyber-illuminated walls, holographic CCTV radar sweep cones, glowing fountain, and HUD telemetry. |
| **4. Ray-Cast Coverage Map** | `design_4_transit_interchange.png` / `.svg` | **Physical CCTV optical calibration.** Proves 83.8% net walkable concourse coverage with wall occlusion and nadir blind spots. |

---

## 2. Visual Gallery of Floor Plan Variants

### Variant 1: Clean Architectural Base (No Camera Projections)
*Pure CAD floor plan showing $10\text{m}$ plus walkway, 4 corner rooms with inward doors, speed gates, and central fountain without camera FOV cones.*
![Clean Base Floor Plan](./design_4_clean_base.png)

---

### Variant 2: Pedestrian / Human Location Dots (Live Crowd Tracking)
*Simulated live commuter locations with directional velocity vectors, rotary flow, and cluster density tags.*
![Human Location Dots Floor Plan](./design_4_human_locations.png)

---

### Variant 3: Tactical Game Map (Realistic Security Dashboard View)
*Command center tactical minimap with holographic radar cones, cyber-illuminated walls, glowing roundabout, and live telemetry.*
![Tactical Game Map](./design_4_tactical_gamemap.png)

---

### Variant 4: Calibrated CCTV Ray-Cast Surveillance Coverage
*Ray-cast line-of-sight analysis proving 83.8% optical concourse coverage with zero wall penetration.*
![Ray-Cast Coverage Map](./design_4_transit_interchange.png)

---

## 3. Direct File Locations on Disk

### Project Root Directory (Instant Access):
- Clean Base: [design_4_clean_base.png](file:///run/media/mobeen/Projects/crowdsafe-fyp/design_4_clean_base.png) | [design_4_clean_base.svg](file:///run/media/mobeen/Projects/crowdsafe-fyp/design_4_clean_base.svg)
- Human Dots: [design_4_human_locations.png](file:///run/media/mobeen/Projects/crowdsafe-fyp/design_4_human_locations.png) | [design_4_human_locations.svg](file:///run/media/mobeen/Projects/crowdsafe-fyp/design_4_human_locations.svg)
- Tactical Game Map: [design_4_tactical_gamemap.png](file:///run/media/mobeen/Projects/crowdsafe-fyp/design_4_tactical_gamemap.png) | [design_4_tactical_gamemap.svg](file:///run/media/mobeen/Projects/crowdsafe-fyp/design_4_tactical_gamemap.svg)
- Interactive HTML Catalog: [plus_walkway_design_catalog.html](file:///run/media/mobeen/Projects/crowdsafe-fyp/plus_walkway_design_catalog.html)

### Web Dashboard Assets (`crowdsafe-dashboard/public/`):
- Served at `/design_4_clean_base.png`
- Served at `/design_4_human_locations.png`
- Served at `/design_4_tactical_gamemap.png`
- Venue JSON: `/floor_plans/design_4_transit_interchange.json`

### Python Analytics Pipeline (`mvp/assets/`):
- `mvp/assets/design_4_clean_base.png`
- `mvp/assets/design_4_human_locations.png`
- `mvp/assets/design_4_tactical_gamemap.png`

---

## 4. Reusing in Application Code

### In Next.js Web Dashboard (`crowdsafe-dashboard`):
```tsx
import Image from "next/image";

export function TacticalMinimap() {
  return (
    <div className="relative rounded-xl overflow-hidden border border-cyan-500/30 bg-slate-950 shadow-2xl">
      <Image
        src="/design_4_tactical_gamemap.png"
        alt="Tactical Surveillance Map"
        width={1024}
        height={1024}
        className="w-full h-auto object-contain"
        priority
      />
    </div>
  );
}
```

### In Python OpenCV Pipeline (`mvp`):
```python
# To use clean base as canvas for dynamic camera cone projection:
base_map = cv2.imread("mvp/assets/design_4_clean_base.png")

# Or use tactical game map for high-tech HUD:
tactical_map = cv2.imread("mvp/assets/design_4_tactical_gamemap.png")
```
