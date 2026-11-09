"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, ShieldCheck, Activity, Video, Users, MapPin, History } from "lucide-react";

export default function Dashboard() {
  const [latestAlert, setLatestAlert] = useState<any>(null);
  const [alertHistory, setAlertHistory] = useState<any[]>([]);

  useEffect(() => {
    const fetchLatestAlert = async () => {
      try {
        const res = await fetch("/api/webhook");
        const json = await res.json();
        
        if (json.data && json.data.timestamp) {
          setLatestAlert((prev: any) => {
            // Only add to history if it's a completely new event
            if (!prev || prev.timestamp !== json.data.timestamp) {
              if (json.data.status !== "SAFE") {
                setAlertHistory((history) => [json.data, ...history].slice(0, 10)); // Keep last 10
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
  
  // Theme variables based on status
  const theme = isCritical 
    ? { bg: "bg-red-950", text: "text-red-400", border: "border-red-900", card: "bg-red-900/20", glow: "shadow-[0_0_30px_rgba(239,68,68,0.3)]" } 
    : isSafe 
      ? { bg: "bg-gray-950", text: "text-emerald-400", border: "border-gray-800", card: "bg-gray-900", glow: "shadow-none" }
      : { bg: "bg-amber-950", text: "text-amber-400", border: "border-amber-900", card: "bg-amber-900/20", glow: "shadow-[0_0_30px_rgba(245,158,11,0.2)]" };

  return (
    <div className={`min-h-screen ${theme.bg} transition-all duration-700 text-white p-6 font-sans`}>
      {/* Top Navbar */}
      <nav className={`flex items-center justify-between border-b ${theme.border} pb-4 mb-6`}>
        <div className="flex items-center gap-3">
          <Activity className={theme.text} size={28} />
          <h1 className="text-2xl font-bold tracking-tight">CrowdSafe Sentinel</h1>
        </div>
        <div className={`px-4 py-1.5 rounded-full border ${theme.border} ${theme.text} text-sm font-semibold flex items-center gap-2 tracking-wide uppercase bg-black/40`}>
          {isSafe ? <ShieldCheck size={16} /> : <AlertTriangle size={16} className="animate-pulse" />}
          {isSafe ? "SYSTEM SAFE" : latestAlert.alert_level}
        </div>
      </nav>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
        
        {/* Left Column: Live CCTV & Active Stats */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Simulated CCTV Feed */}
          <div className={`relative aspect-video bg-black rounded-xl border ${theme.border} overflow-hidden ${theme.glow}`}>
            <div className="absolute top-4 left-4 flex items-center gap-2 bg-black/60 px-3 py-1 rounded text-xs font-mono text-gray-300">
              <Video size={14} /> CAM-01: Main Entrance
            </div>
            {!isSafe && (
              <div className="absolute top-4 right-4 flex items-center gap-2 bg-red-500/20 border border-red-500/50 text-red-500 px-3 py-1 rounded text-xs font-mono animate-pulse">
                REC <span className="w-2 h-2 rounded-full bg-red-500"></span>
              </div>
            )}
            <div className="flex h-full items-center justify-center text-gray-600 font-mono text-sm border-2 border-dashed border-gray-800 m-8 rounded">
               [ Live OpenCV Feed Stream Placeholder ]<br/>
               (WebRTC / RTSP Integration Pending)
            </div>
          </div>

          {/* Telemetry Cards */}
          <div className="grid grid-cols-3 gap-4">
            <div className={`${theme.card} border ${theme.border} rounded-xl p-5`}>
              <div className="flex items-center gap-2 text-gray-400 mb-2 text-sm"><Users size={16}/> Live Count</div>
              <div className="text-3xl font-bold">{latestAlert?.head_count || 0}</div>
            </div>
            <div className={`${theme.card} border ${theme.border} rounded-xl p-5`}>
              <div className="flex items-center gap-2 text-gray-400 mb-2 text-sm"><MapPin size={16}/> Active Zone</div>
              <div className="text-xl font-bold truncate">{latestAlert?.zone_name || "N/A"}</div>
            </div>
            <div className={`${theme.card} border ${theme.border} rounded-xl p-5`}>
              <div className="flex items-center gap-2 text-gray-400 mb-2 text-sm"><Activity size={16}/> Density (m²)</div>
              <div className={`text-3xl font-bold ${theme.text}`}>{latestAlert?.density_per_m2 || "0.00"}</div>
            </div>
          </div>
        </div>

        {/* Right Column: Alert History & Actions */}
        <div className="space-y-6">
          
          {/* Action Protocol */}
          <div className={`${theme.card} border ${theme.border} rounded-xl p-6 h-48 flex flex-col justify-center`}>
            <h3 className="text-gray-400 text-xs uppercase tracking-wider mb-2">Recommended Protocol</h3>
            <p className={`text-lg leading-relaxed font-medium ${isSafe ? 'text-gray-500' : 'text-white'}`}>
              {latestAlert?.recommended_action || "Standard monitoring protocol. No action required."}
            </p>
          </div>

          {/* Incident Log */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 h-[calc(100%-13rem)] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-gray-400 text-xs uppercase tracking-wider flex items-center gap-2">
                <History size={14} /> Incident Log
              </h3>
              <span className="text-xs text-gray-600 font-mono">Last 10 events</span>
            </div>
            
            <div className="space-y-3">
              {alertHistory.length === 0 ? (
                <div className="text-sm text-gray-600 text-center py-8">No recent incidents</div>
              ) : (
                alertHistory.map((alert, idx) => (
                  <div key={idx} className="bg-black/40 border border-gray-800 rounded p-3 text-sm">
                    <div className="flex justify-between items-start mb-1">
                      <span className={alert.alert_level === "CRITICAL" ? "text-red-400 font-semibold" : "text-amber-400 font-semibold"}>
                        {alert.alert_level}
                      </span>
                      <span className="text-gray-500 text-xs font-mono">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="text-gray-300">{alert.zone_name} — {alert.head_count} people ({alert.density_per_m2}/m²)</div>
                  </div>
                ))
              )}
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
}