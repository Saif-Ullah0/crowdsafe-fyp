# CrowdSafe MVP — Agent & User Communication Protocol

This document defines the interaction model, reporting structure, decision checkpoints, and collaboration workflow between **Antigravity (AI Assistant)** and the **User (Project Lead / Engineer)** for building and evaluating the CrowdSafe MVP.

---

## 1. Principles of Collaboration

1. **Autonomous Execution with Transparent Checkpoints:** Antigravity builds modules, runs tests, and updates benchmarks incrementally, surfacing concise status updates with clickable file links.
2. **Deterministic & Reproducible:** Every significant code change, run metric, or error must be logged with instructions on how to reproduce it.
3. **No Hidden Assumptions:** Critical technical decisions (e.g., choice of model weights, camera calibration, risk threshold definitions) must be made explicit or prompted to the user when ambiguity exists.

---

## 2. Standard Progress Report Format

After completing each milestone or significant task batch, Antigravity provides an update using the following structure:

```markdown
### 🚀 Milestone / Task: [Task Name]
- **Status:** ✅ COMPLETED | ⚠️ BLOCKED | 🔄 IN PROGRESS
- **Key Changes Implemented:**
  - Created/Modified: [`path/to/file.py`](file:///run/media/mobeen/977e7ff6-efb4-4c0c-80f3-6ebc7ecfbbfb/FYP/crowdsafe-fyp/mvp/path/to/file.py)
  - Core logic added or changed.
- **Verification & Test Output:**
  - Command run: `python app.py ...`
  - Output summary: `[FPS: 45, MAE: 6.12, Critical Alerts: 0]`
- **Blockers / Decisions Required from User:**
  - None / [Specify question or choice required]
- **Next Planned Action:**
  - [Next task on roadmap]
```

---

## 3. Decision & Escalation Checkpoints

Antigravity must stop and request user feedback when encountering:

| Scenario | Action Required |
| :--- | :--- |
| **Model Weights Missing / Inaccessible** | Alert the user, use synthetic mock pipeline for test continuity, and request model weight path or permission. |
| **Safety Threshold Divergence** | If physical venue layout requires non-standard threshold levels ($p/m^2$), request confirmation on values in `config.py`. |
| **Hardware Constraints (OOM / CPU Throttling)** | Suggest batch size reductions, image downscaling, or switching between MCNN and CSRNet. |
| **External Service / Plugin Integrations** | Ask before setting up external webhooks, email dispatchers, or cloud database sync. |

---

## 4. User Interaction & Slash Command Guide

The user can guide Antigravity using standard natural language or slash commands:

- `/plan` — Use when designing or revising the MVP pipeline architecture or scheduling multi-step phases.
- `/goal` — Use when assigning Antigravity an autonomous, multi-phase implementation session without frequent interruptions.
- `/boost` — Use when debugging tricky PyTorch tensor shapes, model loss divergence, or complex spatial clustering algorithms.
- `/compact` — Use when context history becomes large to compress memory while preserving project states.

---

## 5. Artifacts and Results Tracking Rules

- **Results Log:** All empirical numbers, frame latencies, and counting errors must be recorded in [RESULTS.md](file:///run/media/mobeen/977e7ff6-efb4-4c0c-80f3-6ebc7ecfbbfb/crowdsafe-fyp/mvp/RESULTS.md).
- **Execution Runbook:** Operational procedures must remain up to date in [INSTRUCTIONS.md](file:///run/media/mobeen/977e7ff6-efb4-4c0c-80f3-6ebc7ecfbbfb/crowdsafe-fyp/mvp/INSTRUCTIONS.md).
- **File References:** Always use absolute clickable file links (`file:///...`) for files and code symbols.
