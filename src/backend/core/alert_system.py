"""
Alert & Notification Dispatcher for CrowdSafe MVP
Manages multi-frame hysteresis, instant flash-surge triggers, persistent JSONL logging,
and non-blocking asynchronous Webhook/REST alert dispatches.
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
from config import AlertConfig

class AlertSystem:
    """Evaluates multi-frame safety conditions and dispatches incident alerts asynchronously."""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config if config is not None else AlertConfig()
        self.zone_counters: Dict[str, int] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        
        # Ensure log file directory exists
        self.config.alert_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread pool for non-blocking asynchronous HTTP Webhook dispatches
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="CrowdSafe-Alert")

    def evaluate(
        self,
        zone_metrics: List[Dict[str, Any]],
        camera_id: str = "cam_01",
        is_flash_spike: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Evaluates current zone states against hysteresis and instant flash trigger rules.

        Args:
            zone_metrics: List of zone results from SpatialAnalyzer
            camera_id: Identifier for the video stream source
            is_flash_spike: Flag indicating an immediate dangerous spike

        Returns:
            List of newly triggered alert objects.
        """
        dispatched_alerts = []

        for zone in zone_metrics:
            zid = zone["id"]
            risk = zone["risk_level"]
            density = zone["density_per_m2"]

            # 1. Instant Flash Surge Check
            if is_flash_spike and density >= self.config.instant_critical_threshold:
                if zid not in self.active_alerts:
                    alert_event = self._create_alert(camera_id, zone, trigger_type="INSTANT_FLASH_SPIKE")
                    self.active_alerts[zid] = alert_event
                    dispatched_alerts.append(alert_event)
                    self._dispatch_alert(alert_event)
                continue

            # 2. Standard Hysteresis Check for HIGH or CRITICAL
            if risk in ["HIGH", "CRITICAL"]:
                self.zone_counters[zid] = self.zone_counters.get(zid, 0) + 1
                
                if self.zone_counters[zid] >= self.config.consecutive_frames_trigger:
                    if zid not in self.active_alerts:
                        alert_event = self._create_alert(camera_id, zone, trigger_type="HYSTERESIS_SUSTAINED")
                        self.active_alerts[zid] = alert_event
                        dispatched_alerts.append(alert_event)
                        self._dispatch_alert(alert_event)
            else:
                # Reset counter and clear active alert if back in safe bounds
                if zid in self.active_alerts:
                    del self.active_alerts[zid]
                self.zone_counters[zid] = 0

        return dispatched_alerts

    def _create_alert(self, camera_id: str, zone: Dict[str, Any], trigger_type: str = "HYSTERESIS_SUSTAINED") -> Dict[str, Any]:
        """Constructs a standardized crowd safety incident alert payload."""
        timestamp = datetime.now(timezone.utc).isoformat()
        risk = zone["risk_level"]
        
        action = (
            "URGENT: Deploy crowd marshals and divert incoming traffic immediately" 
            if risk == "CRITICAL" 
            else "Caution: Monitor bottleneck and prepare auxiliary exit route"
        )

        alert_data = {
            "timestamp": timestamp,
            "camera_id": camera_id,
            "zone_id": zone["id"],
            "zone_name": zone["name"],
            "alert_level": risk,
            "trigger_type": trigger_type,
            "density_per_m2": zone["density_per_m2"],
            "head_count": zone["count"],
            "recommended_action": action
        }

        if self.config.enable_console_logging:
            print(f"\n🚨 [{trigger_type}] Level: {risk} | Zone: {zone['name']} | Density: {zone['density_per_m2']} p/m²")
            print(f"   Action: {action}")

        return alert_data

    def _dispatch_alert(self, alert_data: Dict[str, Any]):
        """Logs to JSONL file and submits async webhook request."""
        # 1. Synchronously append to persistent JSONL log
        self._log_to_file(alert_data)

        # 2. Asynchronously POST to webhook URL (if configured)
        if self.config.webhook_url:
            self._executor.submit(self._send_webhook, alert_data)

    def _log_to_file(self, alert_data: Dict[str, Any]):
        """Appends alert payload to persistent JSONL log file."""
        try:
            with open(self.config.alert_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert_data) + "\n")
        except Exception as e:
            print(f"[AlertSystem] Warning: Failed to log alert to file ({e})")

    def _send_webhook(self, alert_data: Dict[str, Any]):
        """Dispatches HTTP POST to external webhook endpoint without blocking pipeline."""
        try:
            headers = {"Content-Type": "application/json", "User-Agent": "CrowdSafe-Sentinel/1.0"}
            response = requests.post(
                self.config.webhook_url,
                json=alert_data,
                headers=headers,
                timeout=self.config.webhook_timeout
            )
            if self.config.enable_console_logging:
                print(f"   └─ Webhook dispatched to {self.config.webhook_url} (HTTP {response.status_code})")
        except Exception as e:
            if self.config.enable_console_logging:
                print(f"   └─ Webhook dispatch warning: Failed to reach {self.config.webhook_url} ({e})")

    def shutdown(self):
        """Cleanly shuts down async executor."""
        self._executor.shutdown(wait=False)
