# CrowdSafe MVP — Real-Time Crowd Safety & Density Monitoring System

## 1. Overview
**CrowdSafe MVP** is an automated video analytics pipeline designed for real-time crowd density monitoring, bottleneck detection, and stampede risk prevention in public venues (malls, transit hubs, stadiums, and event grounds).

This MVP directory contains the modular pipeline codebase, configuration, instructions, benchmark logs, and collaboration protocols.

---

## 2. Directory Layout
```text
mvp/
├── README.md                 # Project overview and quickstart (this file)
├── INSTRUCTIONS.md           # Implementation blueprint and developer instructions
├── RESULTS.md                # Quantitative evaluation, metrics, and benchmark tracking
├── COMMUNICATION.md          # Antigravity <-> User communication protocol & workflow
├── requirements.txt          # Python dependencies
├── config.py                 # Central system parameters, zones, and safety thresholds
├── core/                     # Core pipeline modules
│   ├── __init__.py
│   ├── density_engine.py     # Density estimation model abstraction (MCNN/CSRNet)
│   ├── spatial_analyzer.py   # ROI zoning, crowd density per m², and clustering
│   └── alert_system.py       # Multi-level threshold alert triggers & logging
├── app.py                    # MVP main entrypoint / video processor
└── output/                   # Generated density maps, alerts, and performance logs
```

---

## 3. Quickstart Guide

### 3.1 Environment Setup
```bash
# Ensure Python 3.9+ is active
pip install -r requirements.txt
```

### 3.2 Running the Pipeline Demo
```bash
# Run with default synthetic/sample test stream
python app.py --source sample --visualize

# Run with custom video or camera stream
python app.py --source /path/to/video.mp4 --device cpu
```

---

## 4. Key Documentation Links
- **[Developer Blueprint & Execution Guide](file:///run/media/mobeen/977e7ff6-efb4-4c0c-80f3-6ebc7ecfbbfb/crowdsafe-fyp/mvp/INSTRUCTIONS.md)**
- **[Results & Benchmark Log](file:///run/media/mobeen/977e7ff6-efb4-4c0c-80f3-6ebc7ecfbbfb/crowdsafe-fyp/mvp/RESULTS.md)**
- **[Agent Communication Protocol](file:///run/media/mobeen/977e7ff6-efb4-4c0c-80f3-6ebc7ecfbbfb/crowdsafe-fyp/mvp/COMMUNICATION.md)**
- **[Central Configuration](file:///run/media/mobeen/977e7ff6-efb4-4c0c-80f3-6ebc7ecfbbfb/crowdsafe-fyp/mvp/config.py)**
