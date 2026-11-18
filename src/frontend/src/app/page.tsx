"use client";

import React, { useEffect, useState, useMemo, useCallback } from "react";
import {
  AlertTriangle,
  ShieldCheck,
  Activity,
  Video,
  History,
  Cpu,
  Network,
  Scan,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  MapPin,
  Radio,
  Sparkles,
} from "lucide-react";

// ============================================================================
// 1. STRICT TYPESCRIPT CONTRACTS & INITIAL STATE
// ============================================================================

export type RiskLevel = "SAFE" | "MODERATE" | "HIGH" | "CRITICAL";

export interface WebhookTelemetry {
  timestamp: string;
  camera_id: string;
  zone_id: string;
  zone_name: string;
  status?: "SAFE" | "WARNING" | "CRITICAL";
  alert_level: RiskLevel;
  trigger_type: string;
  density_per_m2: number;
  head_count: number;
  recommended_action: string;
}

export interface AlertEvent extends WebhookTelemetry {
  id: string;
  acknowledged: boolean;
}

export interface PeakDensityRecord {
  val: number;
  zone: string;
}

const INITIAL_HISTORY_LENGTH = 30;
const MAX_INCIDENT_QUEUE = 10;
const POLLING_INTERVAL_MS = 2000;
const CRITICAL_DENSITY_THRESHOLD = 4.0; // Stampede risk threshold (p/m²)

const INITIAL_PEAK_STATE: PeakDensityRecord = {
  val: 0,
  zone: "Awaiting Telemetry",
};

// ============================================================================
// 2. MAIN SOC DASHBOARD COMPONENT
// ============================================================================

export default function CrowdSafeSOCDashboard() {
  const [latestAlert, setLatestAlert] = useState<WebhookTelemetry | null>(null);
  const [alertHistory, setAlertHistory] = useState<AlertEvent[]>([]);
  const [densityHistory, setDensityHistory] = useState<number[]>(() =>
    Array(INITIAL_HISTORY_LENGTH).fill(0)
  );
  const [peakDensity, setPeakDensity] =
    useState<PeakDensityRecord>(INITIAL_PEAK_STATE);
  const [currentTime, setCurrentTime] = useState<string>("--:--:--");

  // Live SOC Wall Clock (avoids Next.js SSR hydration mismatch)
  useEffect(() => {
    const updateClock = () => setCurrentTime(new Date().toLocaleTimeString());
    updateClock();
    const timer = setInterval(updateClock, 1000);
    return () => clearInterval(timer);
  }, []);

  // Telemetry Polling Engine (Decoupled from peakDensity to prevent interval thrashing)
  useEffect(() => {
    let isMounted = true;

    const fetchLatestAlert = async () => {
      try {
        const res = await fetch("/api/webhook", { cache: "no-store" });
        if (!res.ok) return;
        const json = await res.json();

        if (!isMounted || !json?.data?.timestamp) return;
        const data: WebhookTelemetry = json.data;

        // 1. Update rolling 30-point density window
        setDensityHistory((curr) =>
          [...curr, data.density_per_m2 || 0].slice(-INITIAL_HISTORY_LENGTH)
        );

        // 2. Update all-time session peak independently
        setPeakDensity((prevPeak) =>
          data.density_per_m2 > prevPeak.val
            ? { val: data.density_per_m2, zone: data.zone_name || "Unknown" }
            : prevPeak
        );

        // 3. Evaluate incident state & update queue (strictly capped at 10)
        setLatestAlert((prevAlert) => {
          if (!prevAlert || prevAlert.timestamp !== data.timestamp) {
            const isSafeEvent =
              data.status === "SAFE" ||
              !data.alert_level ||
              data.alert_level === "SAFE";

            if (!isSafeEvent) {
              setAlertHistory((history) => {
                if (history.some((h) => h.id === data.timestamp)) return history;
                const newEvent: AlertEvent = {
                  ...data,
                  id: data.timestamp,
                  acknowledged: false,
                };
                return [newEvent, ...history].slice(0, MAX_INCIDENT_QUEUE);
              });
            }
            return data;
          }
          return prevAlert;
        });
      } catch (error) {
        console.error("[CrowdSafe SOC] Telemetry poll error:", error);
      }
    };

    fetchLatestAlert();
    const interval = setInterval(fetchLatestAlert, POLLING_INTERVAL_MS);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Derived Telemetry States
  const isSafe =
    !latestAlert ||
    latestAlert.status === "SAFE" ||
    !latestAlert.alert_level ||
    latestAlert.alert_level === "SAFE";
  const isCritical = latestAlert?.alert_level === "CRITICAL";

  const rateOfChange = useMemo(() => {
    if (densityHistory.length < 2) return "0.0";
    const current = densityHistory[densityHistory.length - 1];
    const prev = densityHistory[densityHistory.length - 2];
    return ((current - prev) * 30).toFixed(1);
  }, [densityHistory]);

  const activeAlertsCount = useMemo(
    () => alertHistory.filter((a) => !a.acknowledged).length,
    [alertHistory]
  );

  const acknowledgeAlert = useCallback((id: string) => {
    setAlertHistory((history) =>
      history.map((alert) =>
        alert.id === id ? { ...alert, acknowledged: true } : alert
      )
    );
  }, []);

  // SVG Math (Preserved + Area Polygon Extension)
  const maxScale = Math.max(...densityHistory, 5.0);
  const graphPoints = useMemo(() => {
    return densityHistory
      .map(
        (val, idx) =>
          `${(idx / (INITIAL_HISTORY_LENGTH - 1)) * 100},${
            100 - (val / maxScale) * 100
          }`
      )
      .join(" ");
  }, [densityHistory, maxScale]);

  const areaPoints = `0,100 ${graphPoints} 100,100`;
  const thresholdY = 100 - (CRITICAL_DENSITY_THRESHOLD / maxScale) * 100;

  // Palantir / Vercel Enterprise Badge Styling
  const getRiskBadgeStyle = (level?: RiskLevel) => {
    if (level === "CRITICAL") {
      return "text-red-400 bg-red-950/40 border-red-500/40 shadow-[0_0_25px_-5px_rgba(239,68,68,0.35)] animate-pulse";
    }
    if (level === "MODERATE" || level === "HIGH") {
      return "text-amber-400 bg-amber-950/30 border-amber-500/30 shadow-[0_0_20px_-5px_rgba(245,158,11,0.2)]";
    }
    return "text-emerald-400 bg-emerald-950/20 border-emerald-500/20";
  };

  return (
    <div className="min-h-screen lg:h-screen bg-[#09090b] text-zinc-300 p-3 sm:p-4 font-sans flex flex-col gap-3.5 lg:overflow-hidden selection:bg-blue-500/20 selection:text-blue-400">
      {/* Subtle Ambient Top Radial Glow */}
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_80%_40%_at_50%_-20%,rgba(59,130,246,0.07),transparent)] pointer-events-none" />

      {/* =====================================================================
          PANEL 1: TOP KPI COMMAND BAR
         ===================================================================== */}
      <header className="relative z-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 sm:gap-4 bg-zinc-900/50 backdrop-blur-xl border border-white/[0.07] p-3.5 rounded-xl items-center shadow-2xl shrink-0">
        {/* Brand & Clock */}
        <div className="col-span-1 flex items-center gap-3.5 lg:border-r border-white/[0.06] pr-4">
          <div className="p-2.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400 shadow-inner">
            <Scan size={22} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-semibold text-zinc-100 tracking-tight">
                CrowdSafe
              </h1>
              <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-white/[0.05] text-zinc-400 border border-white/[0.05]">
                SOC
              </span>
            </div>
            <p className="text-[11px] font-mono text-zinc-500 flex items-center gap-1.5 mt-0.5 tabular-nums">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping" />
              {currentTime} // LIVE
            </p>
          </div>
        </div>

        {/* Total View Count */}
        <div className="col-span-1 px-2 lg:px-4">
          <div className="text-[11px] font-medium uppercase tracking-wider text-zinc-500 mb-1">
            Total View Count
          </div>
          <div className="text-2xl sm:text-3xl font-semibold text-zinc-100 tracking-tight font-mono tabular-nums">
            {Math.round(latestAlert?.head_count || 0)}{" "}
            <span className="text-xs font-sans text-zinc-500 font-normal">
              people
            </span>
          </div>
        </div>

        {/* Peak Density (Current Frame) */}
        <div className="col-span-1 px-2 lg:px-4 lg:border-l border-white/[0.06]">
          <div className="text-[11px] font-medium uppercase tracking-wider text-zinc-500 mb-1">
            Current Density
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-semibold text-zinc-100 font-mono tabular-nums">
              {latestAlert?.density_per_m2?.toFixed(1) || "0.0"}
            </span>
            <span className="text-xs font-mono text-zinc-500">p/m²</span>
          </div>
          <div className="text-[10px] font-mono text-zinc-500 truncate mt-0.5 flex items-center gap-1">
            <MapPin size={10} className="text-zinc-600 shrink-0" />
            ZONE: {latestAlert?.zone_name || "N/A"}
          </div>
        </div>

        {/* Unacknowledged Active Alerts */}
        <div className="col-span-1 px-2 lg:px-4 lg:border-l border-white/[0.06]">
          <div className="text-[11px] font-medium uppercase tracking-wider text-zinc-500 mb-1">
            Active Alerts
          </div>
          <div className="text-2xl sm:text-3xl font-semibold text-zinc-100 font-mono tabular-nums flex items-center gap-2.5">
            {activeAlertsCount}
            {activeAlertsCount > 0 && (
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
              </span>
            )}
          </div>
        </div>

        {/* Overall Risk Pill */}
        <div
          className={`col-span-1 sm:col-span-2 lg:col-span-1 flex justify-center items-center rounded-lg border transition-all duration-500 h-full py-2.5 px-3 ${getRiskBadgeStyle(
            latestAlert?.alert_level
          )}`}
        >
          <div className="text-center">
            <div className="text-[10px] uppercase font-mono font-semibold tracking-widest opacity-75 mb-0.5">
              System Posture
            </div>
            <div className="font-semibold flex items-center justify-center gap-1.5 text-xs sm:text-sm uppercase tracking-wide">
              {isSafe ? <ShieldCheck size={16} /> : <AlertTriangle size={16} />}
              {isSafe ? "Normal Flow" : latestAlert?.alert_level}
            </div>
          </div>
        </div>
      </header>

      {/* =====================================================================
          WORKSPACE GRID (12-COLUMN RESPONSIVE BENTO)
         ===================================================================== */}
      <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-3.5 flex-1 min-h-0">
        {/* LEFT COLUMN: VISUAL TELEMETRY (7 Cols on Desktop) */}
        <div className="lg:col-span-7 flex flex-col gap-3.5 min-h-0">
          {/* PANEL 2: LIVE MCNN CAMERA FEED */}
          <div
            className={`bg-zinc-900/40 backdrop-blur-md border rounded-xl p-3.5 flex-1 flex flex-col min-h-[280px] lg:min-h-0 transition-all duration-500 ${
              isCritical
                ? "border-red-500/50 shadow-[0_0_30px_-10px_rgba(239,68,68,0.25)]"
                : "border-white/[0.07]"
            }`}
          >
            <div className="flex flex-wrap justify-between items-center gap-2 mb-3 shrink-0">
              <div className="flex items-center gap-2">
                <Video size={14} className="text-blue-400" />
                <span className="text-xs font-medium text-zinc-200 tracking-wide">
                  CAM-01 // B-Block Main Lobby
                </span>
                <span className="text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-1.5 py-0.5 rounded">
                  1080p60
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-mono text-zinc-400 bg-white/[0.03] border border-white/[0.07] px-2.5 py-0.5 rounded-md flex items-center gap-1.5">
                  <Sparkles size={11} className="text-blue-400" />
                  MCNN Density Heatmap: ON
                </span>
              </div>
            </div>

            {/* Video Viewport */}
            <div
              className={`relative flex-1 bg-[#050507] rounded-lg border overflow-hidden transition-colors duration-500 ${
                isCritical ? "border-red-500/40" : "border-white/[0.05]"
              }`}
            >
              {/* Subtle Tactical Grid Overlay */}
              <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:32px_32px] pointer-events-none" />

              {/* Dynamic Heatmap Radial Glow */}
              <div
                className={`absolute inset-0 transition-opacity duration-700 pointer-events-none ${
                  isCritical
                    ? "bg-[radial-gradient(circle_at_50%_50%,rgba(239,68,68,0.22),transparent_65%)] animate-pulse"
                    : "bg-[radial-gradient(circle_at_50%_50%,rgba(59,130,246,0.08),transparent_60%)]"
                }`}
              />

              {/* Top-Left HUD Overlay */}
              <div className="absolute top-3 left-3 bg-zinc-950/85 backdrop-blur-md px-2.5 py-1.5 border border-white/10 text-xs font-mono text-zinc-200 rounded-md flex items-center gap-2 shadow-lg">
                <Radio
                  size={12}
                  className={
                    isCritical
                      ? "text-red-500 animate-ping"
                      : "text-emerald-400"
                  }
                />
                <span>
                  HEADCOUNT:{" "}
                  <strong className="text-white tabular-nums">
                    {Math.round(latestAlert?.head_count || 0)}
                  </strong>
                </span>
              </div>

              {/* Action Recommendation Banner (Shows on non-SAFE state) */}
              {!isSafe && latestAlert?.recommended_action && (
                <div className="absolute bottom-3 inset-x-3 bg-red-950/90 backdrop-blur-md border border-red-500/40 text-red-200 px-3 py-2 rounded-lg text-xs font-mono flex items-center justify-between shadow-lg animate-in fade-in slide-in-from-bottom-2 duration-300">
                  <span className="flex items-center gap-2">
                    <AlertTriangle size={14} className="text-red-400 shrink-0" />
                    <span>
                      PROTOCOL: <strong>{latestAlert.recommended_action}</strong>
                    </span>
                  </span>
                  <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 bg-red-500/20 rounded">
                    Immediate Action
                  </span>
                </div>
              )}

              {/* Stream Placeholder Text */}
              <div className="flex h-full flex-col items-center justify-center text-zinc-600 text-xs font-mono text-center px-6 select-none">
                <Scan size={28} className="mb-2 text-zinc-700 stroke-[1.5]" />
                <span className="text-zinc-500 font-medium">
                  [ OpenCV RTSP / WebRTC Stream Target ]
                </span>
                <span className="text-[11px] text-zinc-600 mt-1">
                  MCNN density map &amp; bounding vectors render on this canvas
                </span>
              </div>
            </div>
          </div>

          {/* PANEL 3: 3-ZONE B-BLOCK FLOOR PLAN */}
          <div className="bg-zinc-900/40 backdrop-blur-md border border-white/[0.07] rounded-xl p-3.5 h-auto sm:h-44 shrink-0 flex flex-col">
            <div className="flex justify-between items-center mb-2.5">
              <span className="text-xs font-medium text-zinc-300 flex items-center gap-2">
                <MapPin size={14} className="text-blue-400" />
                Spatial Telemetry // B-Block Floor Plan
              </span>
              <span className="text-[10px] font-mono text-zinc-500">
                THRESHOLD: {CRITICAL_DENSITY_THRESHOLD.toFixed(1)} p/m²
              </span>
            </div>

            <div className="flex-1 bg-[#070709] border border-white/[0.05] rounded-lg p-2.5 grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {/* Zone 1: Entrance */}
              <div className="border border-white/[0.06] bg-zinc-900/50 rounded-md p-3 flex flex-col justify-between relative group hover:border-white/15 transition-colors">
                <div className="flex justify-between items-center">
                  <span className="text-[10px] font-mono text-zinc-500 tracking-wider">
                    01 // ENTRANCE
                  </span>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                </div>
                <div className="my-2 text-center">
                  <span className="text-xl font-mono font-semibold text-emerald-400 tabular-nums">
                    1.2
                  </span>
                  <span className="text-[10px] font-mono text-zinc-500 ml-1">
                    p/m²
                  </span>
                </div>
                <div className="text-[10px] font-mono text-zinc-600 text-center">
                  STATUS: OPTIMAL
                </div>
              </div>

              {/* Zone 2: Main Corridor (Dynamic Telemetry Target) */}
              <div
                className={`border rounded-md p-3 flex flex-col justify-between relative transition-all duration-500 ${
                  !isSafe
                    ? "border-red-500/50 bg-red-950/25 shadow-[inset_0_0_20px_rgba(239,68,68,0.12)]"
                    : "border-white/[0.06] bg-zinc-900/50"
                }`}
              >
                <div className="flex justify-between items-center">
                  <span className="text-[10px] font-mono text-zinc-400 tracking-wider font-semibold">
                    02 // MAIN CORRIDOR
                  </span>
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      !isSafe ? "bg-red-500 animate-ping" : "bg-emerald-500"
                    }`}
                  />
                </div>
                <div className="my-2 text-center">
                  <span
                    className={`text-xl font-mono font-semibold tabular-nums ${
                      !isSafe ? "text-red-400" : "text-zinc-100"
                    }`}
                  >
                    {latestAlert?.density_per_m2?.toFixed(1) || "0.0"}
                  </span>
                  <span className="text-[10px] font-mono text-zinc-500 ml-1">
                    p/m²
                  </span>
                </div>
                <div
                  className={`text-[10px] font-mono text-center uppercase ${
                    !isSafe ? "text-red-400 font-semibold" : "text-zinc-500"
                  }`}
                >
                  {isSafe ? "STATUS: NORMAL" : `ALERT: ${latestAlert?.alert_level}`}
                </div>
              </div>

              {/* Zone 3: Exit Route */}
              <div className="border border-white/[0.06] bg-zinc-900/50 rounded-md p-3 flex flex-col justify-between relative group hover:border-white/15 transition-colors">
                <div className="flex justify-between items-center">
                  <span className="text-[10px] font-mono text-zinc-500 tracking-wider">
                    03 // EXIT ROUTE
                  </span>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                </div>
                <div className="my-2 text-center">
                  <span className="text-xl font-mono font-semibold text-emerald-400 tabular-nums">
                    0.5
                  </span>
                  <span className="text-[10px] font-mono text-zinc-500 ml-1">
                    p/m²
                  </span>
                </div>
                <div className="text-[10px] font-mono text-emerald-500/80 text-center">
                  RELIEF GATES: OPEN
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: ANALYTICS & INCIDENT QUEUE (5 Cols on Desktop) */}
        <div className="lg:col-span-5 flex flex-col gap-3.5 min-h-0">
          {/* PANEL 4: SVG DENSITY TRENDS & VELOCITY */}
          <div className="bg-zinc-900/40 backdrop-blur-md border border-white/[0.07] rounded-xl p-4 flex flex-col shrink-0">
            <div className="flex justify-between items-center mb-3">
              <span className="text-xs font-medium text-zinc-300 flex items-center gap-2">
                <Activity size={14} className="text-blue-400" />
                Density Velocity &amp; Trend
              </span>
              <div
                className={`text-xs font-mono tabular-nums flex items-center gap-1 font-semibold px-2 py-0.5 rounded border ${
                  Number(rateOfChange) > 0
                    ? "text-amber-400 bg-amber-950/30 border-amber-500/20"
                    : "text-emerald-400 bg-emerald-950/20 border-emerald-500/20"
                }`}
              >
                {Number(rateOfChange) > 0 ? (
                  <TrendingUp size={13} />
                ) : (
                  <TrendingDown size={13} />
                )}
                {Number(rateOfChange) > 0 ? `+${rateOfChange}` : rateOfChange}{" "}
                p/min
              </div>
            </div>

            {/* SVG Sparkline with Area Gradient & Critical Threshold Line */}
            <div className="h-24 w-full relative mb-2.5 bg-[#070709] border border-white/[0.04] rounded-lg p-2 overflow-hidden">
              <svg
                className="w-full h-full overflow-visible"
                preserveAspectRatio="none"
                viewBox="0 0 100 100"
              >
                <defs>
                  <linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="0%"
                      stopColor={isCritical ? "#ef4444" : "#3b82f6"}
                      stopOpacity="0.35"
                    />
                    <stop
                      offset="100%"
                      stopColor={isCritical ? "#ef4444" : "#3b82f6"}
                      stopOpacity="0.0"
                    />
                  </linearGradient>
                </defs>

                {/* Critical Threshold Line (4.0 p/m²) */}
                <line
                  x1="0"
                  y1={thresholdY}
                  x2="100"
                  y2={thresholdY}
                  stroke="rgba(239,68,68,0.35)"
                  strokeDasharray="2,2"
                  strokeWidth="1"
                  vectorEffect="non-scaling-stroke"
                />

                {/* Area Fill */}
                <polygon points={areaPoints} fill="url(#trendGrad)" />

                {/* Primary Trend Line */}
                <polyline
                  fill="none"
                  stroke={isCritical ? "#ef4444" : "#3b82f6"}
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  vectorEffect="non-scaling-stroke"
                  points={graphPoints}
                  className="transition-all duration-300"
                />
              </svg>
            </div>

            <div className="flex justify-between items-center text-[11px] font-mono text-zinc-500">
              <span>WINDOW: 60s (30 SAMPLES)</span>
              <span className="text-zinc-400 tabular-nums">
                PEAK:{" "}
                <strong className="text-zinc-200">
                  {peakDensity.val.toFixed(1)} p/m²
                </strong>{" "}
                ({peakDensity.zone})
              </span>
            </div>
          </div>

          {/* PANEL 5: INCIDENT QUEUE (MAX 10 ITEMS) */}
          <div className="bg-zinc-900/40 backdrop-blur-md border border-white/[0.07] rounded-xl p-4 flex flex-col flex-1 min-h-[260px] lg:min-h-0">
            <div className="flex justify-between items-center mb-3 shrink-0">
              <span className="text-xs font-medium text-zinc-300 flex items-center gap-2">
                <History size={14} className="text-blue-400" />
                Incident Dispatch Queue
              </span>
              <span className="text-[10px] font-mono text-zinc-500 bg-white/[0.03] px-2 py-0.5 rounded border border-white/[0.05]">
                {alertHistory.length} / {MAX_INCIDENT_QUEUE} SLOTS
              </span>
            </div>

            {/* Scrollable Queue Container */}
            <div className="space-y-2.5 overflow-y-auto flex-1 pr-1">
              {alertHistory.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-xs font-mono text-zinc-600 py-8 border border-dashed border-white/[0.05] rounded-lg">
                  <ShieldCheck
                    size={22}
                    className="text-emerald-500/40 mb-1.5"
                  />
                  <span>NO ACTIVE INCIDENTS IN QUEUE</span>
                </div>
              ) : (
                alertHistory.map((alert) => {
                  const isAlertCrit = alert.alert_level === "CRITICAL";
                  return (
                    <div
                      key={alert.id}
                      className={`shrink-0 border rounded-lg p-3 text-xs transition-all duration-300 ${
                        alert.acknowledged
                          ? "opacity-45 border-white/[0.05] bg-zinc-950/40"
                          : isAlertCrit
                          ? "border-red-500/40 bg-red-950/20 shadow-[0_0_15px_-5px_rgba(239,68,68,0.2)]"
                          : "border-amber-500/30 bg-amber-950/15"
                      }`}
                    >
                      <div className="flex justify-between items-center mb-2">
                        <div className="flex items-center gap-2">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold tracking-wider uppercase ${
                              isAlertCrit
                                ? "bg-red-500 text-white"
                                : "bg-amber-500 text-zinc-950"
                            }`}
                          >
                            {alert.alert_level}
                          </span>
                          <span className="text-zinc-400 font-mono text-[11px] tabular-nums">
                            {new Date(alert.timestamp).toLocaleTimeString()}
                          </span>
                        </div>

                        {!alert.acknowledged ? (
                          <button
                            onClick={() => acknowledgeAlert(alert.id)}
                            className="text-[11px] font-mono text-zinc-200 hover:text-white flex items-center gap-1 bg-white/[0.07] hover:bg-white/[0.14] border border-white/10 px-2.5 py-1 rounded-md transition-all active:scale-95 cursor-pointer"
                          >
                            <CheckCircle2 size={12} className="text-emerald-400" />
                            ACK
                          </button>
                        ) : (
                          <span className="text-[10px] font-mono text-zinc-500 uppercase">
                            ✓ Acknowledged
                          </span>
                        )}
                      </div>

                      <div className="text-zinc-200 font-medium mb-1.5 flex items-center justify-between">
                        <span>{alert.zone_name}</span>
                        <span className="text-[10px] font-mono text-zinc-500">
                          CAM: {alert.camera_id || "CAM-01"}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-zinc-400 bg-black/30 p-2 rounded border border-white/[0.04]">
                        <div>
                          DENSITY:{" "}
                          <span className="text-zinc-100 font-semibold tabular-nums">
                            {alert.density_per_m2.toFixed(1)} p/m²
                          </span>
                        </div>
                        <div>
                          HEADCOUNT:{" "}
                          <span className="text-zinc-100 font-semibold tabular-nums">
                            {Math.round(alert.head_count)}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* PANEL 6: TELEMETRY & MODEL HEALTH BAR */}
          <div className="bg-zinc-900/40 backdrop-blur-md border border-white/[0.07] rounded-xl p-2.5 grid grid-cols-4 gap-2 text-center text-[10px] font-mono text-zinc-400 shrink-0">
            <div className="flex flex-col items-center justify-center border-r border-white/[0.06] py-1">
              <Cpu size={13} className="mb-1 text-blue-400" />
              <span className="text-zinc-300">MCNN v2.1</span>
            </div>
            <div className="flex flex-col items-center justify-center border-r border-white/[0.06] py-1">
              <Network size={13} className="mb-1 text-zinc-500" />
              <span className="tabular-nums">45 FPS</span>
            </div>
            <div className="flex flex-col items-center justify-center border-r border-white/[0.06] py-1">
              <Activity size={13} className="mb-1 text-zinc-500" />
              <span className="tabular-nums">22ms Inference</span>
            </div>
            <div className="flex flex-col items-center justify-center text-emerald-400 font-semibold py-1">
              <ShieldCheck size={13} className="mb-1" />
              <span>RTSP: ONLINE</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}