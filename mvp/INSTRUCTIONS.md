# CrowdSafe MVP — Implementation Instructions & Development Roadmap

This document outlines the detailed technical specifications, operational rules, and step-by-step tasks required for Antigravity (and developers) to build, test, and verify the CrowdSafe MVP.

---

## 1. System Architecture & Objectives

The goal of the CrowdSafe MVP is to ingest CCTV video frames or image sequences, compute high-accuracy spatial crowd density maps, analyze danger zones/bottlenecks, and issue actionable safety alerts with sub-second latency.

```
+------------------+      +-------------------------+      +---------------------------+
| Video Input      | ---> | Density Engine          | ---> | Spatial Analyzer          |
| (Camera/File)    |      | (MCNN / CSRNet backend) |      | (Grid ROI & Heatmaps)     |
+------------------+      +-------------------------+      +-------------+-------------+
                                                                         |
                                                                         v
+------------------+      +-------------------------+      +---------------------------+
| Output Dash/Logs | <--- | Visualizer / Exporter   | <--- | Safety & Risk Engine      |
| (JSON / Alerts)  |      | (Overlay & Heatmap)     |      | (Thresholds: Safe/Danger) |
+------------------+      +-------------------------+      +---------------------------+
```

---

## 2. Core Functional Requirements

### 2.1 Density Estimation Engine (`core/density_engine.py`)
- Support multi-model backend abstraction with auto-fallback:
  1. **Primary Model:** MCNN (Multi-Column Convolutional Neural Network - chosen for lowest MAE: 6.12 and lightweight footprint).
  2. **Secondary Model:** CSRNet / DM-Count.
  3. **Fallback/Simulation Mode:** Synthetic density map generator if pre-trained weights are not yet loaded, allowing end-to-end integration testing immediately.
- Accept raw RGB frames (NumPy arrays or PIL Images).
- Return:
  - `total_count` (float rounded to nearest integer)
  - `density_map` (2D float array representing head density distribution)
  - `inference_time_ms` (latency metric)

### 2.2 Spatial Risk & Zone Analyzer (`core/spatial_analyzer.py`)
- Divide surveillance frames into configurable Regions of Interest (ROIs / Zones) or regular grid cells.
- Compute metric equivalents:
  - Estimated physical area per zone ($m^2$).
  - Density metric: $\text{Persons per } m^2 = \frac{\text{Zone Count}}{\text{Zone Area}}$.
- Classify risk levels based on international crowd safety standards (Fruin's Level of Service):
  - **LOW / SAFE:** $< 2.0 \text{ people}/m^2$ (Free movement)
  - **MODERATE / CAUTION:** $2.0 - 3.5 \text{ people}/m^2$ (Constrained movement)
  - **HIGH / WARNING:** $3.5 - 5.0 \text{ people}/m^2$ (Crowd compression begins)
  - **CRITICAL / DANGER:** $> 5.0 \text{ people}/m^2$ (High crush / stampede risk)
- Congestion Blob Detection:
  - Threshold the continuous density map to isolate high-density clusters using morphological operations or contour detection.

### 2.3 Alerting & Event Dispatcher (`core/alert_system.py`)
- Trigger alerts when a zone exceeds warning or critical thresholds for more than $K$ consecutive frames (hysteresis to prevent flickering alerts).
- Structure alert payload:
  ```json
  {
    "timestamp": "2026-11-07T13:10:00Z",
    "camera_id": "cam_north_gate",
    "alert_level": "CRITICAL",
    "zone_id": "zone_exit_corridor",
    "density_per_m2": 5.4,
    "head_count": 82,
    "recommended_action": "Divert incoming crowd flow to Exit Gate B"
  }
  ```
- Persist triggered alerts to an append-only JSONL log file.

### 2.4 Visualization & Video Pipeline (`app.py`)
- Provide a unified runner supporting:
  - `--source`: Video file path, directory of frame images, webcam index, or synthetic stream (`mock`).
  - `--visualize`: Render side-by-side display of original frame with bounding/zone overlays and Jet colormap density heatmap.
  - `--save-output`: Export annotated video or frame snapshots.

---

## 3. Development Phases & Tasks for Antigravity

### Phase 1: Foundation & Skeleton Pipeline (Complete)
- [x] Create project layout and core docs (`README.md`, `INSTRUCTIONS.md`, `RESULTS.md`, `COMMUNICATION.md`).
- [x] Define `requirements.txt` and `config.py` with default zone definitions, safety thresholds, and 1 FPS streaming rate.
- [x] Set up isolated Python 3.12 virtual environment via `uv` with PyTorch CPU, OpenCV, and Scipy.
- [x] Implement `core/dvr_streamer.py` for 1 FPS DVR/CCTV sequence simulation and video ingestion.
- [x] Implement `core/density_engine.py` with plug-and-play abstraction (`BaseDensityModel`, `FastSimModel`, `MCNNModel`, `CSRNetModel`).
- [x] Implement `core/spatial_analyzer.py` for zone-based density mapping, Fruin risk escalation, cluster contours, and peak dots.
- [x] Implement `core/alert_system.py` for multi-frame hysteresis, flash-surge triggers, async webhook dispatching, and JSONL logging.
- [x] Implement `core/visualizer.py` with high-freedom multi-layer OpenCV GUI overlays and keyboard hotkeys.
- [x] Implement `webhook_receiver.py` local testing server.
- [x] Implement `app.py` end-to-end CLI & GUI runner.
- [x] Add automated unit test suite `tests/test_pipeline.py` (8/8 passing).

### Phase 2: Integration of Real Weights & Mall Dataset Validation
- [ ] Wire up real MCNN / CSRNet PyTorch model definitions.
- [ ] Load pre-trained weights from `../models/` if available, or configure automated download/loading.
- [ ] Test inference across test images (`../datasets/Mall/` or sample video).
- [ ] Log benchmark figures to `RESULTS.md`.

### Phase 3: Verification & Performance Optimization
- [ ] Benchmark CPU vs GPU FPS and memory utilization.
- [ ] Add unit tests (`tests/test_pipeline.py`) validating:
  - Model forward pass output dimensions.
  - Correct calculation of density per $m^2$.
  - State machine transitions of alerts (trigger, sustain, resolve).

---

## 4. Engineering Rules & Coding Standards
1. **Never crash on missing weights or inputs:** Always provide graceful fallbacks and actionable error messages.
2. **Deterministic Configuration:** All configurable hyperparameters (thresholds, zone geometries, camera mappings) must reside in `config.py` rather than hardcoded magic numbers.
3. **Traceability:** Every major benchmark or experiment must be appended to [RESULTS.md](file:///run/media/mobeen/977e7ff6-efb4-4c0c-80f3-6ebc7ecfbbfb/crowdsafe-fyp/mvp/RESULTS.md).
4. **Communication:** Report progress, blockers, and questions according to [COMMUNICATION.md](file:///run/media/mobeen/977e7ff6-efb4-4c0c-80f3-6ebc7ecfbbfb/crowdsafe-fyp/mvp/COMMUNICATION.md).
