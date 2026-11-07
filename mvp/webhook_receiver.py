"""
Lightweight Webhook & REST Receiver for CrowdSafe MVP
Runs a local HTTP server to test, receive, and display incident alerts in real time.
Usage:
    python webhook_receiver.py --port 8080
"""
import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

class AlertWebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        
        try:
            alert_payload = json.loads(post_data.decode("utf-8"))
            timestamp = alert_payload.get("timestamp", datetime.now().isoformat())
            level = alert_payload.get("alert_level", "UNKNOWN")
            zone = alert_payload.get("zone_name", "Unknown Zone")
            density = alert_payload.get("density_per_m2", 0.0)
            count = alert_payload.get("head_count", 0.0)
            action = alert_payload.get("recommended_action", "None")
            trigger = alert_payload.get("trigger_type", "STANDARD")

            print("\n" + "=" * 65)
            print(f"📡 [WEBHOOK RECEIVED] {timestamp}")
            print(f"   Level     : {level} [{trigger}]")
            print(f"   Zone      : {zone}")
            print(f"   Density   : {density} people/m² (Headcount: {count})")
            print(f"   Directive : {action}")
            print("=" * 65)

            # Return HTTP 200 OK
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {"status": "success", "message": "Alert received and logged"}
            self.wfile.write(json.dumps(response).encode("utf-8"))

        except Exception as e:
            print(f"[WebhookReceiver] Error parsing payload: {e}")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))

    def log_message(self, format, *args):
        # Suppress standard HTTP request logging
        return


def main():
    parser = argparse.ArgumentParser(description="CrowdSafe Alert Webhook Receiver")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    args = parser.parse_args()

    server_address = ("", args.port)
    httpd = HTTPServer(server_address, AlertWebhookHandler)
    print("=" * 65)
    print(f"  CrowdSafe Incident Webhook Receiver Active on Port {args.port}")
    print(f"  Listening for POST alerts at: http://localhost:{args.port}/alert")
    print("  Press Ctrl+C to stop.")
    print("=" * 65 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[WebhookReceiver] Shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    main()

