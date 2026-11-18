"""
CrowdSafe MVP — Main Pipeline Entrypoint & Executive Runner
Connects 1 FPS DVR/CCTV Ingestion, Plug-and-Play Density Estimation,
Spatial Risk Analysis, Non-blocking Webhook Alerts, and High-Freedom Visualizer.
"""
import sys
import argparse
import time
from pathlib import Path
import cv2

from config import AppConfig
from core.dvr_streamer import DVRStreamer
from core.density_engine import get_density_engine
from core.spatial_analyzer import SpatialAnalyzer
from core.alert_system import AlertSystem
from core.visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="CrowdSafe MVP — Real-Time CCTV Sentinel")
    parser.add_argument("--source", type=str, default="sample", help="Input source: 'sample', image directory, video path, or webcam ID")
    parser.add_argument("--model", type=str, default="fast_sim", choices=["fast_sim", "mcnn", "csrnet"], help="Model architecture")
    parser.add_argument("--fps", type=float, default=1.0, help="Ingestion rate in FPS (default: 1.0)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device: 'cpu' or 'cuda'")
    parser.add_argument("--webhook", type=str, default=None, help="HTTP POST Webhook URL for incident alerts")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (default: run continuously)")
    parser.add_argument("--save-output", action="store_true", default=True, help="Save preview snapshots to output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    config = AppConfig()

    # Override config with CLI arguments
    config.stream.source = args.source
    config.stream.target_fps = args.fps
    config.stream.max_frames = args.max_frames
    config.model.architecture = args.model
    config.model.device = args.device
    if args.webhook:
        config.alert.webhook_url = args.webhook
    config.visualizer.headless = args.headless

    print("=" * 70)
    print("      CrowdSafe MVP — Real-Time CCTV Safety & Density Sentinel      ")
    print("=" * 70)
    print(f" Source          : {config.stream.source}")
    print(f" Ingestion Rate  : {config.stream.target_fps} FPS (DVR Simulation)")
    print(f" Model Engine    : {config.model.architecture.upper()}")
    print(f" Compute Device  : {config.model.device.upper()}")
    print(f" Webhook Target  : {config.alert.webhook_url or 'Disabled'}")
    print(f" Visualizer Mode : {'Headless (CLI Only)' if config.visualizer.headless else 'OpenCV High-Freedom GUI'}")
    print(f" Hotkeys (GUI)   : 'h' Heatmap | 'b' Zones | 'd' Dots | 'z' HUD | 's' Snap | 'Space' Pause | 'q' Exit")
    print("=" * 70 + "\n")

    # 1. Initialize Ingestion Streamer
    streamer = DVRStreamer(
        source=config.stream.source,
        target_fps=config.stream.target_fps,
        loop=config.stream.loop,
        max_frames=config.stream.max_frames
    )

    # 2. Initialize Density Model Engine
    density_engine = get_density_engine(
        architecture=config.model.architecture,
        weights_path=config.model.weights_path,
        device=config.model.device
    )

    # 3. Initialize Spatial & Alert Systems
    spatial_analyzer = SpatialAnalyzer(
        zones=config.zones,
        instant_critical_threshold=config.alert.instant_critical_threshold
    )
    alert_system = AlertSystem(config=config.alert)

    # 4. Initialize Visualizer
    visualizer = Visualizer(config=config.visualizer)

    latest_alert = None
    processed_count = 0
    total_latency_ms = 0.0

    try:
        while True:
            # Paced read to enforce target 1 FPS DVR rate
            success, frame, frame_idx = streamer.read(pace=not config.visualizer.headless)
            if not success or frame is None:
                print("\n[CrowdSafe] Reached end of input stream.")
                break

            loop_start = time.time()

            # Step 1: Crowd Density Estimation
            density_result = density_engine.predict(frame)
            latency_ms = density_result["latency_ms"]
            total_latency_ms += latency_ms

            # Step 2: Spatial Risk Analysis
            spatial_analysis = spatial_analyzer.analyze(
                density_result["density_map"],
                frame.shape[:2]
            )

            # Step 3: Safety Condition & Webhook Evaluation
            new_alerts = alert_system.evaluate(
                spatial_analysis["zones"],
                camera_id="CAM_NORTH_GATE_01",
                is_flash_spike=spatial_analysis["is_flash_spike"]
            )
            if new_alerts:
                latest_alert = new_alerts[-1]

            processed_count += 1
            fps_effective = round(1.0 / max((time.time() - loop_start), 0.001), 1)

            # Step 4: Render Visual Overlays
            annotated_frame = visualizer.render(
                frame_bgr=frame,
                density_map=density_result["density_map"],
                spatial_analysis=spatial_analysis,
                fps=fps_effective,
                latency_ms=latency_ms,
                active_alert=latest_alert
            )

            # Print concise terminal status line
            top_zone = max(spatial_analysis["zones"], key=lambda z: z["density_per_m2"])
            print(f"Frame [{frame_idx:04d}] | Est. Count: {density_result['total_count']:5.1f} | "
                  f"Peak: {top_zone['name']} ({top_zone['density_per_m2']:4.1f} p/m²) | "
                  f"Risk: {spatial_analysis['overall_risk']:8s} | Latency: {latency_ms:4.1f}ms")

            # Save snapshot preview for initial frames
            if config.visualizer.headless and config.output_dir:
                if frame_idx == 1 or (args.max_frames and frame_idx == args.max_frames):
                    snap_path = config.output_dir / f"preview_frame_{frame_idx}.png"
                    cv2.imwrite(str(snap_path), annotated_frame)
                    print(f"  └─ Saved snapshot to: {snap_path}")

            # Step 5: Handle GUI Keys and Refresh Window
            if not visualizer.handle_key_events(annotated_frame, wait_ms=20):
                print("\n[CrowdSafe] User requested shutdown ('q' or ESC pressed).")
                break

    except KeyboardInterrupt:
        print("\n[CrowdSafe] Process interrupted by operator.")

    finally:
        streamer.close()
        visualizer.close()
        alert_system.shutdown()

        avg_latency = round(total_latency_ms / max(processed_count, 1), 2)
        print("\n" + "=" * 70)
        print(" CrowdSafe Executive Run Summary:")
        print(f"  - Total Frames Ingested : {processed_count}")
        print(f"  - Average Model Latency : {avg_latency} ms")
        print(f"  - Incidents Logged File : {config.alert.alert_log_file}")
        print("=" * 70)


if __name__ == "__main__":
    main()
