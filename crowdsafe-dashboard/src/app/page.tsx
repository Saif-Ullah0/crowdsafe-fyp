"use client";

import { useEffect, useState } from "react";
import { 
  AlertTriangle, ShieldCheck, Activity, Video, Users, MapPin, 
  History, Cpu, Network, Scan, Crosshair, Terminal
} from "lucide-react";

export default function Dashboard() {
  const [latestAlert, setLatestAlert] = useState<any>(null);
  const [alertHistory, setAlertHistory] = useState<any[]>([]);
  const [densityHistory, setDensityHistory] = useState<number[]>(Array(20).fill(0));

  useEffect(() => {
    const fetchLatestAlert = async () => {
      try {
        const res = await fetch("/api/webhook");
        const json = await res.json();
        
        if (json.data && json.data.timestamp) {
          setLatestAlert((prev: any) => {
            // Update history arrays
            setDensityHistory(curr => {
              const newHistory = [...curr, json.data.density_per_m2 || 0].slice(-20);
              return newHistory;
            });

            if (!prev || prev.timestamp !== json.data.timestamp) {
              if (json.data.status !== "SAFE") {
                setAlertHistory((history) => [json.data, ...history].slice(0, 8));
              }
              return json.data;
            }
            return prev;
          });
        }
      } catch (error) {
        console.error("Failed to fetch alert data", error);
      }
    };

    fetchLatestAlert();
    const interval = setInterval(fetchLatestAlert, 2000);
    return () => clearInterval(interval);
  }, []);

  const isSafe = !latestAlert || latestAlert.status === "SAFE" || !latestAlert.alert_level;
  const isCritical = latestAlert?.alert_level === "CRITICAL";
  
  const theme = isCritical 
    ? { bg: "bg-red-950/20", border: "border-red-500/50", text: "text-red-500", glow: "shadow-[0_0_15px_rgba(239,68,68,0.4)]", muted: "text-red-500/50" } 
    : isSafe 
      ? { bg: "bg-emerald-950/20", border: "border-emerald-500/50", text: "text-emerald-500", glow: "shadow-[0_0_15px_rgba(16,185,129,0.2)]", muted: "text-emerald-500/50" }
      : { bg: "bg-amber-950/20", border: "border-amber-500/50", text: "text-amber-500", glow: "shadow-[0_0_15px_rgba(245,158,11,0.3)]", muted: "text-amber-500/50" };

  // Calculate SVG Graph points
  const maxDensity = Math.max(...densityHistory, 5.0); // Minimum scale of 5.0
  const graphPoints = densityHistory.map((val, idx) => {
    const x = (idx / 19) * 100;
    const y = 100 - ((val / maxDensity) * 100);
    return `${x},${y}`;
  }).join(" ");

  return (
    <div className="min-h-screen bg-gray-950 text-gray-300 p-4 lg:p-6 font-mono selection:bg-gray-800">
      <div className="max-w-screen-2xl mx-auto space-y-4">
        
        {/* Top Navbar */}
        <header className={`flex items-center justify-between border-b ${theme.border} pb-4`}>
          <div className="flex items-center gap-4">
            <div className={`p-2 rounded border ${theme.border} ${theme.bg}`}>
              <Scan className={theme.text} size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-widest text-white uppercase">CrowdSafe OS</h1>
              <p className="text-xs text-gray-500">v2.0 // Real-Time Density Sentinel</p>
            </div>
          </div>
          
          <div className="flex items-center gap-6">
            <div className="hidden md:flex gap-6 text-xs text-gray-500">
              <span className="flex items-center gap-2"><Cpu size={14}/> ENGINE: MCNN</span>
              <span className="flex items-center gap-2"><Network size={14}/> LATENCY: 2ms</span>
            </div>
            <div className={`px-6 py-2 rounded border ${theme.border} ${theme.bg} ${theme.text} font-bold flex items-center gap-3 tracking-widest uppercase ${theme.glow}`}>
              {isSafe ? <ShieldCheck size={18} /> : <AlertTriangle size={18} className="animate-pulse" />}
              {isSafe ? "SYSTEM NOMINAL" : latestAlert.alert_level}
            </div>
          </div>
        </header>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          
          {/* Left Column: CCTV & Analytics (Spans 8 cols) */}
          <div className="lg:col-span-8 space-y-4">
            
            {/* Advanced CCTV Feed */}
            <div className={`relative aspect-video bg-black rounded border ${theme.border} overflow-hidden ${theme.glow}`}>
              {/* Grid Overlay */}
              <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] bg-[size:40px_40px]" />
              
              {/* Corner Brackets */}
              <div className={`absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 ${theme.border}`} />
              <div className={`absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 ${theme.border}`} />
              <div className={`absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 ${theme.border}`} />
              <div className={`absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 ${theme.border}`} />
              
              <div className="absolute top-4 left-4 ml-10 mt-1 flex flex-col gap-1">
                <span className="bg-black/80 px-2 py-1 text-xs border border-gray-800 text-gray-400 flex items-center gap-2">
                  <Video size={12}/> CAM-01 [MAIN ENTRANCE]
                </span>
                <span className="bg-black/80 px-2 py-1 text-[10px] border border-gray-800 text-gray-500">
                  FOV: 85° | HGT: 4.5m | FPS: 30
                </span>
              </div>

              {!isSafe && (
                <div className="absolute top-4 right-4 mr-10 mt-1 flex items-center gap-2 bg-red-950/80 border border-red-500 text-red-500 px-3 py-1 text-xs font-bold animate-pulse">
                  REC <span className="w-2 h-2 rounded-full bg-red-500"></span>
                </div>
              )}

              {/* Center Crosshair */}
              <div className="absolute inset-0 flex items-center justify-center opacity-20">
                <Crosshair size={64} className={theme.text} />
              </div>

              <div className="absolute bottom-4 left-4 ml-10 mb-1 text-xs text-gray-600">
                OPENCV STREAM OFFLINE / AWAITING RTSP
              </div>
            </div>

            {/* Live Analytics Row */}
            <div className="grid grid-cols-3 gap-4">
              <div className={`bg-black rounded border ${theme.border} p-4 flex flex-col justify-between`}>
                <div className="flex items-center gap-2 text-gray-500 text-xs mb-2"><Users size={14}/> LIVE COUNT</div>
                <div className="text-4xl font-light text-white">{latestAlert?.head_count?.toFixed(1) || "0.0"}</div>
              </div>
              <div className={`bg-black rounded border ${theme.border} p-4 flex flex-col justify-between`}>
                <div className="flex items-center gap-2 text-gray-500 text-xs mb-2"><MapPin size={14}/> ACTIVE ZONE</div>
                <div className="text-xl font-medium text-gray-300 truncate">{latestAlert?.zone_name || "STANDBY"}</div>
              </div>
              <div className={`bg-black rounded border ${theme.border} p-4 flex flex-col justify-between`}>
                <div className="flex items-center gap-2 text-gray-500 text-xs mb-2"><Activity size={14}/> DENSITY (p/m²)</div>
                <div className={`text-4xl font-bold ${theme.text}`}>{latestAlert?.density_per_m2 || "0.00"}</div>
              </div>
            </div>

            {/* Density Graph */}
            <div className={`bg-black rounded border ${theme.border} p-4`}>
              <div className="flex justify-between items-center mb-4">
                <span className="text-xs text-gray-500 flex items-center gap-2"><Activity size={14}/> DENSITY TREND (LAST 20 CYCLES)</span>
                <span className="text-[10px] text-gray-600">MAX CAPACITY: 5.0</span>
              </div>
              <div className="h-24 w-full relative">
                {/* Graph Background Grid */}
                <div className="absolute inset-0 border-b border-l border-gray-800 flex flex-col justify-between">
                  <div className="border-b border-gray-800/50 border-dashed w-full h-px" />
                  <div className="border-b border-gray-800/50 border-dashed w-full h-px" />
                  <div className="border-b border-gray-800/50 border-dashed w-full h-px" />
                </div>
                <svg className="w-full h-full overflow-visible" preserveAspectRatio="none" viewBox="0 0 100 100">
                  <polyline
                    fill="none"
                    stroke={isCritical ? "#ef4444" : isSafe ? "#10b981" : "#f59e0b"}
                    strokeWidth="2"
                    points={graphPoints}
                    className="transition-all duration-300"
                  />
                </svg>
              </div>
            </div>
          </div>

          {/* Right Column: Logs & Protocol (Spans 4 cols) */}
          <div className="lg:col-span-4 space-y-4 flex flex-col">
            
            {/* Action Protocol */}
            <div className={`bg-black rounded border ${theme.border} p-5 ${theme.bg}`}>
              <h3 className={`text-xs uppercase tracking-widest mb-3 flex items-center gap-2 ${theme.text}`}>
                <ShieldCheck size={16} /> ACTION PROTOCOL
              </h3>
              <p className={`text-sm leading-relaxed ${isSafe ? 'text-gray-500' : 'text-white'}`}>
                {latestAlert?.recommended_action || "> System monitoring active. Crowd levels within safe operational parameters."}
              </p>
            </div>

            {/* Terminal Log */}
            <div className={`bg-black rounded border ${theme.border} p-4 flex-1 flex flex-col overflow-hidden min-h-[400px]`}>
              <div className="flex items-center justify-between mb-4 border-b border-gray-800 pb-2">
                <h3 className="text-xs uppercase tracking-widest text-gray-500 flex items-center gap-2">
                  <Terminal size={14} /> SYSTEM LOG
                </h3>
              </div>
              
              <div className="space-y-2 overflow-y-auto flex-1 pr-2">
                {alertHistory.length === 0 ? (
                  <div className="text-xs text-gray-600 font-mono">_ waiting for anomalous events...</div>
                ) : (
                  alertHistory.map((alert, idx) => (
                    <div key={idx} className="border-l-2 border-gray-800 pl-3 py-1 text-xs">
                      <div className="flex justify-between items-start mb-1">
                        <span className={alert.alert_level === "CRITICAL" ? "text-red-500" : "text-amber-500"}>
                          [{alert.alert_level}]
                        </span>
                        <span className="text-gray-600">
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="text-gray-400">
                        {alert.zone_name} <br/>
                        <span className="text-gray-500">
                          &gt; {alert.head_count} p // {alert.density_per_m2} p/m²
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
            
          </div>
        </div>
      </div>
    </div>
  );
}