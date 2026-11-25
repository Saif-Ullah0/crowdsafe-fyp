"use client";

import { useEffect, useState, useMemo, useRef, useCallback } from "react";

type RiskLevel = "SAFE" | "MODERATE" | "CRITICAL";
type DashboardMode = "SIMULATION" | "LIVE";
type SimScenario = "NORMAL" | "BUILDUP" | "CRITICAL";
type LegalModal = "TOS" | "PRIVACY" | null;

interface TelemetryPayload {
  timestamp: string;
  camera_id: string;
  zone_id: string;
  zone_name: string;
  status: string;
  alert_level: RiskLevel;
  trigger_type: string;
  density_per_m2: number;
  head_count: number;
  recommended_action: string;
}

interface AlertEvent extends TelemetryPayload {
  id: string;
  acknowledged: boolean;
}

interface ZoneRow {
  id: string;
  name: string;
  density: number;
  limit: number;
  exitState: string;
}

export default function Dashboard() {
  const [booting, setBooting] = useState<boolean>(true);
  const [mode, setMode] = useState<DashboardMode>("SIMULATION");
  const [simScenario, setSimScenario] = useState<SimScenario>("NORMAL");
  const [legalModal, setLegalModal] = useState<LegalModal>(null);
  const [clock, setClock] = useState<string>("00:00:00");

  const [latestAlert, setLatestAlert] = useState<TelemetryPayload | null>(null);
  const [zones, setZones] = useState<ZoneRow[]>([
    { id: "Z-01", name: "B-Block Main Entrance", density: 1.0, limit: 4.5, exitState: "OPEN" },
    { id: "Z-02", name: "B-Block Main Corridor", density: 1.2, limit: 4.5, exitState: "MONITORED" },
    { id: "Z-03", name: "Exit B Evacuation Route", density: 0.5, limit: 4.5, exitState: "CLEAR" },
  ]);
  const [alertHistory, setAlertHistory] = useState<AlertEvent[]>([]);
  const [densityHistory, setDensityHistory] = useState<number[]>(Array(30).fill(1.2));
  const [peakDensity, setPeakDensity] = useState<{ val: number; time: string }>({
    val: 1.2,
    time: "00:00:00",
  });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const buildupStepRef = useRef<number>(1.8);
  const lastTimestampRef = useRef<string>("");

  // Initial skeleton loader state
  useEffect(() => {
    const timer = setTimeout(() => setBooting(false), 450);
    return () => clearTimeout(timer);
  }, []);

  // Hydration-safe live clock
  useEffect(() => {
    const tick = () => setClock(new Date().toLocaleTimeString("en-GB", { hour12: false }));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  const ingestTelemetry = useCallback((data: TelemetryPayload, nextZones?: ZoneRow[]) => {
    const timeStr = new Date(data.timestamp).toLocaleTimeString("en-GB", { hour12: false });

    setLatestAlert(data);
    if (nextZones) setZones(nextZones);

    setPeakDensity((prev) =>
      data.density_per_m2 > prev.val ? { val: data.density_per_m2, time: timeStr } : prev
    );

    setDensityHistory((curr) => [...curr, data.density_per_m2].slice(-30));

    if (lastTimestampRef.current !== data.timestamp) {
      lastTimestampRef.current = data.timestamp;
      if (data.status !== "SAFE" && data.alert_level !== "SAFE") {
        setAlertHistory((history) => {
          const entry: AlertEvent = {
            ...data,
            id: `${data.timestamp}-${Math.random().toString(36).slice(2, 6)}`,
            acknowledged: false,
          };
          return [entry, ...history].slice(0, 10);
        });
      }
    }
  }, []);

  // Telemetry engine: Simulation vs Live Webhook
  useEffect(() => {
    if (mode === "SIMULATION") {
      if (simScenario !== "BUILDUP") {
        buildupStepRef.current = 1.8;
      }

      const emitSim = () => {
        const jitter = () => (Math.random() - 0.5) * 0.3;
        let corridor = 1.2 + jitter();
        let entrance = 1.0 + jitter() * 0.6;
        let exit = 0.5 + jitter() * 0.4;

        if (simScenario === "BUILDUP") {
          buildupStepRef.current = Math.min(4.4, buildupStepRef.current + 0.24);
          corridor = buildupStepRef.current + jitter() * 0.35;
          entrance = 2.2 + jitter() * 0.4;
          exit = 1.1 + jitter() * 0.3;
        } else if (simScenario === "CRITICAL") {
          corridor = 5.1 + jitter() * 0.7;
          entrance = 3.2 + jitter() * 0.5;
          exit = 1.4 + jitter() * 0.4;
        }

        corridor = Math.max(0.2, Number(corridor.toFixed(1)));
        entrance = Math.max(0.2, Number(entrance.toFixed(1)));
        exit = Math.max(0.1, Number(exit.toFixed(1)));

        const alertLevel: RiskLevel =
          corridor >= 4.5 ? "CRITICAL" : corridor >= 2.5 ? "MODERATE" : "SAFE";

        const headCount = Math.round(corridor * 14 + entrance * 8 + exit * 6);

        const action =
          alertLevel === "CRITICAL"
            ? "EVACUATE: Open Exit B barriers and halt Main Entrance inflow."
            : alertLevel === "MODERATE"
            ? "THROTTLE: Deploy marshals to B-Block Corridor to slow inflow."
            : "NOMINAL: Baseline density monitoring active.";

        ingestTelemetry(
          {
            timestamp: new Date().toISOString(),
            camera_id: "CAM-01",
            zone_id: "Z-02",
            zone_name: "B-Block Main Corridor",
            status: alertLevel === "SAFE" ? "SAFE" : "ALERT",
            alert_level: alertLevel,
            trigger_type: alertLevel === "CRITICAL" ? "FLASH_SURGE" : "THRESHOLD",
            density_per_m2: corridor,
            head_count: headCount,
            recommended_action: action,
          },
          [
            {
              id: "Z-01",
              name: "B-Block Main Entrance",
              density: entrance,
              limit: 4.5,
              exitState: alertLevel === "CRITICAL" ? "HOLD INFLOW" : "OPEN",
            },
            {
              id: "Z-02",
              name: "B-Block Main Corridor",
              density: corridor,
              limit: 4.5,
              exitState: alertLevel === "CRITICAL" ? "CONGESTED" : "MONITORED",
            },
            {
              id: "Z-03",
              name: "Exit B Evacuation Route",
              density: exit,
              limit: 4.5,
              exitState: "CLEAR",
            },
          ]
        );
      };

      emitSim();
      const id = setInterval(emitSim, 1800);
      return () => clearInterval(id);
    }

    const pollWebhook = async () => {
      try {
        const res = await fetch("/api/webhook");
        const json = await res.json();
        if (json.data && json.data.timestamp) {
          const d = json.data as TelemetryPayload;
          const base = Number((d.density_per_m2 || 0).toFixed(1));
          ingestTelemetry(d, [
            {
              id: "Z-01",
              name: "B-Block Main Entrance",
              density: Number(Math.max(0.3, base * 0.55).toFixed(1)),
              limit: 4.5,
              exitState: d.alert_level === "CRITICAL" ? "HOLD INFLOW" : "OPEN",
            },
            {
              id: "Z-02",
              name: d.zone_name || "B-Block Main Corridor",
              density: base,
              limit: 4.5,
              exitState: d.alert_level === "CRITICAL" ? "CONGESTED" : "MONITORED",
            },
            {
              id: "Z-03",
              name: "Exit B Evacuation Route",
              density: Number(Math.max(0.2, base * 0.3).toFixed(1)),
              limit: 4.5,
              exitState: "CLEAR",
            },
          ]);
        }
      } catch (err) {
        console.error("Webhook poll error:", err);
      }
    };

    pollWebhook();
    const id = setInterval(pollWebhook, 2000);
    return () => clearInterval(id);
  }, [mode, simScenario, ingestTelemetry]);

  const currentLevel: RiskLevel = latestAlert?.alert_level || "SAFE";
  const currentDensity = latestAlert?.density_per_m2 ?? 0;
  const isCritical = currentLevel === "CRITICAL";

  // Crisp Grid-Cell MCNN Spatial Density Matrix aligned to actual crowd positions
  useEffect(() => {
    if (booting) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cols = 20;
    const rows = 10;
    const w = canvas.width;
    const h = canvas.height;
    const cellW = w / cols;
    const cellH = h / rows;

    ctx.clearRect(0, 0, w, h);

    // Three hotspots matching the actual crowd locations in cctv-corridor.jpg
    const hotspots = [
      { cx: 6.0, cy: 6.2, rx: 2.8, ry: 2.4, weight: 0.95 },  // Left group
      { cx: 13.2, cy: 6.2, rx: 2.8, ry: 2.4, weight: 0.95 }, // Right group
      { cx: 10.0, cy: 3.4, rx: 3.0, ry: 1.5, weight: 1.05 }, // Dense back corridor
    ];

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        let maxFalloff = 0;
        for (const spot of hotspots) {
          const dx = (c - spot.cx) / spot.rx;
          const dy = (r - spot.cy) / spot.ry;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const f = Math.max(0, 1 - dist * 0.85) * spot.weight;
          if (f > maxFalloff) maxFalloff = f;
        }

        const cellDensity =
          maxFalloff > 0.15
            ? Math.max(0, currentDensity * maxFalloff + Math.sin(c * 1.7 + r * 2.3) * 0.2)
            : 0;

        let fill = "rgba(0, 0, 0, 0)";
        if (cellDensity >= 4.5) fill = "rgba(185, 38, 38, 0.48)";
        else if (cellDensity >= 3.2) fill = "rgba(180, 95, 25, 0.40)";
        else if (cellDensity >= 2.2) fill = "rgba(165, 130, 28, 0.32)";
        else if (cellDensity >= 1.2) fill = "rgba(26, 115, 85, 0.24)";

        if (fill !== "rgba(0, 0, 0, 0)") {
          ctx.fillStyle = fill;
          ctx.fillRect(c * cellW + 1, r * cellH + 1, cellW - 2, cellH - 2);
        }

        if (cellDensity >= 2.5) {
          ctx.fillStyle = "#F3F4F6";
          ctx.font = "bold 10px ui-monospace, SFMono-Regular, Menlo, monospace";
          ctx.fillText(cellDensity.toFixed(1), c * cellW + 6, r * cellH + cellH / 2 + 3);
        }
      }
    }

    // Outer Monitored Corridor Boundary
    ctx.strokeStyle = isCritical ? "#E05252" : currentLevel === "MODERATE" ? "#D99B38" : "#38A17D";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(cellW * 1.5, cellH * 1.8, cellW * 17, cellH * 7.5);

    // Cluster Bounding Boxes aligned to actual crowd pockets when density >= 2.5
    if (currentDensity >= 2.5) {
      const totalPax = latestAlert?.head_count || 0;
      const boxes = [
        { x: 3.5, y: 4.0, w: 5.0, h: 4.6, label: "CLUSTER-L", share: 0.32 },
        { x: 10.8, y: 4.0, w: 5.2, h: 4.6, label: "CLUSTER-R", share: 0.33 },
        { x: 7.2, y: 2.0, w: 5.6, h: 2.6, label: "BOTTLENECK", share: 0.35 },
      ];

      boxes.forEach((b) => {
        const bx = cellW * b.x;
        const by = cellH * b.y;
        const bw = cellW * b.w;
        const bh = cellH * b.h;

        ctx.strokeStyle = isCritical ? "#E05252" : "#D99B38";
        ctx.lineWidth = 1.75;
        ctx.strokeRect(bx, by, bw, bh);

        ctx.fillStyle = isCritical ? "#8C2626" : "#6E4E16";
        ctx.fillRect(bx, by - 15, 134, 15);
        ctx.fillStyle = "#F3F4F6";
        ctx.font = "bold 10px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillText(
          `${b.label} | ~${Math.round(totalPax * b.share)} PAX`,
          bx + 5,
          by - 4
        );
      });
    }
  }, [booting, currentDensity, currentLevel, isCritical, latestAlert?.head_count]);
  const rateOfChange = useMemo(() => {
    if (densityHistory.length < 2) return "0.0";
    const curr = densityHistory[densityHistory.length - 1];
    const prev = densityHistory[densityHistory.length - 2];
    return ((curr - prev) * 30).toFixed(1);
  }, [densityHistory]);

  const unackedCount = alertHistory.filter((a) => !a.acknowledged).length;

  const acknowledgeAlert = (id: string) => {
    setAlertHistory((prev) =>
      prev.map((a) => (a.id === id ? { ...a, acknowledged: true } : a))
    );
  };

  const maxScale = Math.max(...densityHistory, 6.0);
  const graphPoints = densityHistory
    .map(
      (val, idx) =>
        `${(idx / (densityHistory.length - 1)) * 100},${100 - (val / maxScale) * 100}`
    )
    .join(" ");

  const levelColor = (level: RiskLevel) => {
    if (level === "CRITICAL") return "text-[#E05252]";
    if (level === "MODERATE") return "text-[#D99B38]";
    return "text-[#38A17D]";
  };

  const levelBg = (level: RiskLevel) => {
    if (level === "CRITICAL") return "bg-[#8C2626] text-[#F9FAFB]";
    if (level === "MODERATE") return "bg-[#6E4E16] text-[#F9FAFB]";
    return "bg-[#1A4D3B] text-[#E6E8EB]";
  };

  return (
    <div
      style={{
        fontFamily: "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      }}
      className="h-screen bg-[#0B0D10] text-[#E6E8EB] flex flex-col overflow-hidden select-none"
    >
      {/* TOP UTILITY BAR */}
      <header className="h-10 bg-[#101318] border-b border-[#1F242D] px-4 flex items-center justify-between text-xs shrink-0">
        <div className="flex items-center gap-4">
          <span className="font-bold tracking-wider text-[#F3F4F6]">CROWDSAFE</span>
          <span className="text-[#555E6D]">|</span>
          <span className="text-[#8A94A6] font-mono text-[11px]">SOC TERMINAL / B-BLOCK</span>
          <span className="text-[#555E6D]">|</span>
          <span className="font-mono tabular-nums text-[#C5CBD3]">{clock} PKT</span>
        </div>

        <div className="flex items-center gap-4">
          {/* Feed Mode Selector */}
          <div className="flex items-center border border-[#1F242D] bg-[#0B0D10]">
            <button
              onClick={() => setMode("SIMULATION")}
              className={`px-2.5 py-1 text-[11px] font-mono ${
                mode === "SIMULATION"
                  ? "bg-[#243447] text-[#F3F4F6]"
                  : "text-[#7D8590]"
              }`}
            >
              SIMULATION
            </button>
            <button
              onClick={() => setMode("LIVE")}
              className={`px-2.5 py-1 text-[11px] font-mono border-l border-[#1F242D] ${
                mode === "LIVE"
                  ? "bg-[#1A4D3B] text-[#F3F4F6]"
                  : "text-[#7D8590]"
              }`}
            >
              LIVE WEBHOOK
            </button>
          </div>

          {/* Scenario Controls */}
          {mode === "SIMULATION" && (
            <div className="flex items-center border border-[#1F242D] bg-[#0B0D10]">
              {(["NORMAL", "BUILDUP", "CRITICAL"] as SimScenario[]).map((sc, i) => (
                <button
                  key={sc}
                  onClick={() => setSimScenario(sc)}
                  className={`px-2.5 py-1 text-[11px] font-mono ${
                    i > 0 ? "border-l border-[#1F242D]" : ""
                  } ${
                    simScenario === sc
                      ? sc === "CRITICAL"
                        ? "bg-[#8C2626] text-[#FFFFFF]"
                        : sc === "BUILDUP"
                        ? "bg-[#6E4E16] text-[#FFFFFF]"
                        : "bg-[#1F2937] text-[#FFFFFF]"
                      : "text-[#7D8590]"
                  }`}
                >
                  {sc === "CRITICAL" ? "SURGE" : sc}
                </button>
              ))}
            </div>
          )}

          {/* Overall Risk Indicator */}
          <div className={`px-2.5 py-1 text-[11px] font-mono font-semibold ${levelBg(currentLevel)}`}>
            STATUS: {currentLevel}
          </div>
        </div>
      </header>

      {/* CRITICAL INCIDENT BAR (Only rendered when CRITICAL) */}
      {isCritical && !booting && (
        <div className="bg-[#8C2626] text-[#F9FAFB] px-4 py-1.5 text-xs font-mono flex items-center justify-between border-b border-[#B33939] shrink-0">
          <span>
            CRITICAL THRESHOLD BREACHED: {latestAlert?.zone_name} AT {currentDensity.toFixed(1)} P/M2
          </span>
          <span>ACTION: {latestAlert?.recommended_action}</span>
        </div>
      )}

      {/* KEY METRICS LEDGER TAPE (4-column hairline strip, no cards) */}
      <section className="grid grid-cols-4 border-b border-[#1F242D] bg-[#0E1116] shrink-0">
        {booting ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className={`px-4 py-3 ${i > 0 ? "border-l border-[#1F242D]" : ""}`}
            >
              <div className="h-2.5 w-24 bg-[#181D26] mb-2" />
              <div className="h-6 w-20 bg-[#181D26]" />
            </div>
          ))
        ) : (
          <>
            <div className="px-4 py-2.5">
              <div className="text-[11px] text-[#7D8590] uppercase tracking-wide">
                Estimated Headcount
              </div>
              <div className="text-xl font-mono tabular-nums font-semibold text-[#F3F4F6] mt-0.5">
                {Math.round(latestAlert?.head_count || 0)}{" "}
                <span className="text-xs font-normal text-[#7D8590]">PAX</span>
              </div>
            </div>

            <div className="px-4 py-2.5 border-l border-[#1F242D]">
              <div className="text-[11px] text-[#7D8590] uppercase tracking-wide">
                Peak Density (Active)
              </div>
              <div className={`text-xl font-mono tabular-nums font-semibold mt-0.5 ${levelColor(currentLevel)}`}>
                {currentDensity.toFixed(1)}{" "}
                <span className="text-xs font-normal text-[#7D8590]">P/M2</span>
              </div>
            </div>

            <div className="px-4 py-2.5 border-l border-[#1F242D]">
              <div className="text-[11px] text-[#7D8590] uppercase tracking-wide">
                Density Velocity
              </div>
              <div className="text-xl font-mono tabular-nums font-semibold text-[#E6E8EB] mt-0.5">
                {Number(rateOfChange) > 0 ? `+${rateOfChange}` : rateOfChange}{" "}
                <span className="text-xs font-normal text-[#7D8590]">P/M2/MIN</span>
              </div>
            </div>

            <div className="px-4 py-2.5 border-l border-[#1F242D]">
              <div className="text-[11px] text-[#7D8590] uppercase tracking-wide">
                Unacknowledged Alerts
              </div>
              <div className="text-xl font-mono tabular-nums font-semibold text-[#F3F4F6] mt-0.5">
                {unackedCount}{" "}
                <span className="text-xs font-normal text-[#7D8590]">
                  / PEAK {peakDensity.val.toFixed(1)} @ {peakDensity.time}
                </span>
              </div>
            </div>
          </>
        )}
      </section>

      {/* MAIN ASYMMETRIC SPLIT WORKSPACE */}
      <main className="flex-1 grid grid-cols-12 min-h-0">
        {/* LEFT PANE (7 cols): Spatial Matrix + Zone Telemetry Table */}
        <div className="col-span-7 border-r border-[#1F242D] flex flex-col min-h-0">
          {/* Spatial Matrix Panel */}
          <div className="flex-1 flex flex-col min-h-0 border-b border-[#1F242D]">
            <div className="px-4 py-2 border-b border-[#1F242D] bg-[#101318] flex items-center justify-between text-[11px] font-mono shrink-0">
              <span className="text-[#C5CBD3] font-semibold">
                CAM-01: MCNN SPATIAL DENSITY MATRIX
              </span>
              <span className="text-[#7D8590]">
                {mode === "SIMULATION" ? "SOURCE: SIMULATED STREAM" : "SOURCE: /API/WEBHOOK"} | GRID: 20X10
              </span>
            </div>

            <div className="flex-1 bg-[#080A0E] p-3 flex items-center justify-center min-h-0">
              {booting ? (
                <div className="w-full h-full bg-[#141820]" />
              ) : (
                <div className="relative w-full h-full border border-[#1F242D] overflow-hidden bg-[#080A0E]">
                  <img
                    src="/cctv-corridor.jpg"
                    alt="B-Block Main Corridor CCTV Feed"
                    className="w-full h-full object-cover opacity-80"
                  />
                  <canvas
                    ref={canvasRef}
                    width={760}
                    height={340}
                    className="absolute inset-0 w-full h-full"
                  />
                </div>
              )}
            </div>
          </div>

          {/* Zone Telemetry Table (Institutional Ledger) */}
          <div className="h-44 flex flex-col shrink-0 bg-[#0D1015]">
            <div className="px-4 py-1.5 border-b border-[#1F242D] bg-[#101318] flex items-center justify-between text-[11px] font-mono">
              <span className="text-[#C5CBD3] font-semibold">ZONE TELEMETRY LEDGER</span>
              <span className="text-[#7D8590]">FRUIN LOS LIMIT: 4.5 P/M2</span>
            </div>

            <div className="overflow-y-auto flex-1">
              <table className="w-full text-left border-collapse font-mono text-xs tabular-nums">
                <thead>
                  <tr className="border-b border-[#1F242D] text-[10px] text-[#7D8590] bg-[#0B0D10]">
                    <th className="py-1.5 px-4 font-normal">ZONE ID</th>
                    <th className="py-1.5 px-4 font-normal">LOCATION</th>
                    <th className="py-1.5 px-4 font-normal text-right">DENSITY</th>
                    <th className="py-1.5 px-4 font-normal text-right">LIMIT</th>
                    <th className="py-1.5 px-4 font-normal text-right">ROUTE STATE</th>
                  </tr>
                </thead>
                <tbody>
                  {zones.map((z) => {
                    const zRisk: RiskLevel =
                      z.density >= 4.5 ? "CRITICAL" : z.density >= 2.5 ? "MODERATE" : "SAFE";
                    return (
                      <tr key={z.id} className="border-b border-[#161A22]">
                        <td className="py-2 px-4 text-[#8A94A6]">{z.id}</td>
                        <td className="py-2 px-4 text-[#E6E8EB]">{z.name}</td>
                        <td className={`py-2 px-4 text-right font-semibold ${levelColor(zRisk)}`}>
                          {z.density.toFixed(1)} p/m2
                        </td>
                        <td className="py-2 px-4 text-right text-[#7D8590]">
                          {z.limit.toFixed(1)} p/m2
                        </td>
                        <td className="py-2 px-4 text-right text-[#C5CBD3]">{z.exitState}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* RIGHT PANE (5 cols): Time-Series Chart + Incident Queue */}
        <div className="col-span-5 flex flex-col min-h-0 bg-[#0D1015]">
          {/* Time-Series Density Chart */}
          <div className="h-44 border-b border-[#1F242D] flex flex-col shrink-0">
            <div className="px-4 py-2 border-b border-[#1F242D] bg-[#101318] flex items-center justify-between text-[11px] font-mono">
              <span className="text-[#C5CBD3] font-semibold">DENSITY TIME-SERIES (30 SAMPLES)</span>
              <span className="text-[#7D8590]">CRITICAL: 4.5 P/M2</span>
            </div>

            <div className="flex-1 p-3 flex gap-2 min-h-0">
              <div className="flex flex-col justify-between text-[10px] font-mono text-[#7D8590] tabular-nums pr-1">
                <span>{maxScale.toFixed(1)}</span>
                <span>{(maxScale / 2).toFixed(1)}</span>
                <span>0.0</span>
              </div>

              <div className="flex-1 relative border border-[#1F242D] bg-[#080A0E]">
                {booting ? (
                  <div className="w-full h-full bg-[#141820]" />
                ) : (
                  <svg
                    className="w-full h-full"
                    preserveAspectRatio="none"
                    viewBox="0 0 100 100"
                  >
                    <line
                      x1="0"
                      y1={100 - (4.5 / maxScale) * 100}
                      x2="100"
                      y2={100 - (4.5 / maxScale) * 100}
                      stroke="#C93B3B"
                      strokeWidth="1"
                      strokeDasharray="2 2"
                      vectorEffect="non-scaling-stroke"
                    />
                    <line
                      x1="0"
                      y1={100 - (2.5 / maxScale) * 100}
                      x2="100"
                      y2={100 - (2.5 / maxScale) * 100}
                      stroke="#C88A2E"
                      strokeWidth="1"
                      strokeDasharray="2 2"
                      vectorEffect="non-scaling-stroke"
                    />
                    <polyline
                      fill="none"
                      stroke={isCritical ? "#E05252" : "#5B8EC9"}
                      strokeWidth="1.75"
                      vectorEffect="non-scaling-stroke"
                      points={graphPoints}
                    />
                  </svg>
                )}
              </div>
            </div>
          </div>

          {/* Incident Queue Ledger */}
          <div className="flex-1 flex flex-col min-h-0">
            <div className="px-4 py-2 border-b border-[#1F242D] bg-[#101318] flex items-center justify-between text-[11px] font-mono shrink-0">
              <span className="text-[#C5CBD3] font-semibold">INCIDENT QUEUE (MAX 10)</span>
              <span className="text-[#7D8590]">{unackedCount} OPEN</span>
            </div>

            <div className="flex-1 overflow-y-auto divide-y divide-[#161A22]">
              {alertHistory.length === 0 ? (
                <div className="p-6 text-xs font-mono text-[#7D8590] text-center">
                  NO ACTIVE INCIDENTS RECORDED. SELECT BUILDUP OR SURGE TO TEST DISPATCH.
                </div>
              ) : (
                alertHistory.map((alert) => (
                  <div
                    key={alert.id}
                    className={`px-4 py-2.5 text-xs ${
                      alert.acknowledged ? "opacity-45 bg-[#0B0D10]" : "bg-[#0E1117]"
                    }`}
                  >
                    <div className="flex items-center justify-between font-mono mb-1">
                      <div className="flex items-center gap-2">
                        <span className={`px-1.5 py-0.5 text-[10px] font-semibold ${levelBg(alert.alert_level)}`}>
                          {alert.alert_level}
                        </span>
                        <span className="text-[#8A94A6] tabular-nums">
                          {new Date(alert.timestamp).toLocaleTimeString("en-GB", { hour12: false })}
                        </span>
                        <span className="text-[#C5CBD3]">{alert.zone_id}</span>
                      </div>

                      {alert.acknowledged ? (
                        <span className="text-[10px] text-[#38A17D]">ACKED</span>
                      ) : (
                        <button
                          onClick={() => acknowledgeAlert(alert.id)}
                          className="border border-[#2D3542] bg-[#151A22] px-2 py-0.5 text-[10px] text-[#E6E8EB]"
                        >
                          ACK
                        </button>
                      )}
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-[#C5CBD3] mb-1">
                      <span>{alert.zone_name}</span>
                      <span className="font-mono tabular-nums">
                        {alert.density_per_m2.toFixed(1)} P/M2 | {Math.round(alert.head_count)} PAX
                      </span>
                    </div>

                    <div className="text-[11px] text-[#7D8590] font-mono truncate">
                      {alert.recommended_action}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </main>

      {/* INSTITUTIONAL FOOTER (System Specs + TOS / Privacy Policy) */}
      <footer className="h-8 bg-[#101318] border-t border-[#1F242D] px-4 flex items-center justify-between text-[11px] font-mono text-[#7D8590] shrink-0">
        <div className="flex items-center gap-4">
          <span>MODEL: MCNN v2.1 (0.13M PARAMS)</span>
          <span>|</span>
          <span>TARGET: 65 FPS</span>
          <span>|</span>
          <span>LATENCY: 15MS</span>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={() => setLegalModal("TOS")}
            className="text-[#8A94A6] underline underline-offset-2"
          >
            Terms of Service
          </button>
          <button
            onClick={() => setLegalModal("PRIVACY")}
            className="text-[#8A94A6] underline underline-offset-2"
          >
            Privacy Policy
          </button>
        </div>
      </footer>

      {/* LEGAL MODAL (TOS / Privacy Policy) */}
      {legalModal && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-50">
          <div className="bg-[#101318] border border-[#2D3542] max-w-lg w-full p-5">
            <div className="flex items-center justify-between border-b border-[#1F242D] pb-3 mb-3">
              <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-[#F3F4F6]">
                {legalModal === "TOS" ? "Terms of Service" : "Privacy Policy"}
              </h2>
              <button
                onClick={() => setLegalModal(null)}
                className="text-xs font-mono text-[#8A94A6] border border-[#2D3542] px-2 py-0.5"
              >
                CLOSE
              </button>
            </div>
            <div className="text-xs text-[#A0A8B5] leading-relaxed space-y-2">
              {legalModal === "TOS" ? (
                <>
                  <p>
                    CrowdSafe is an automated crowd density monitoring and decision-support interface
                    developed for Security Operations Center (SOC) personnel.
                  </p>
                  <p>
                    Density estimates and evacuation advisories are generated from computer vision
                    models and must be verified by authorized on-site safety officers prior to
                    executing physical crowd control measures.
                  </p>
                </>
              ) : (
                <>
                  <p>
                    CrowdSafe operates strictly on aggregate spatial density estimation (MCNN). The
                    system does not perform facial recognition, biometric identification, or
                    individual person tracking.
                  </p>
                  <p>
                    Only anonymized numerical telemetry (density per square meter, aggregate zone
                    headcount, and incident timestamps) is transmitted to this interface.
                  </p>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}