"use client";

import { useEffect, useState, useMemo } from "react";
import { 
  AlertTriangle, ShieldCheck, Activity, Video, Users, MapPin, 
  History, Cpu, Network, Scan, Map, CheckCircle2, TrendingUp, TrendingDown
} from "lucide-react";

interface AlertEvent {
  id: string;
  timestamp: string;
  camera_id: string;
  zone_id: string;
  zone_name: string;
  alert_level: string;
  trigger_type: string;
  density_per_m2: number;
  head_count: number;
  recommended_action: string;
  acknowledged: boolean;
}

export default function Dashboard() {
  const [latestAlert, setLatestAlert] = useState<any>(null);
  const [alertHistory, setAlertHistory] = useState<AlertEvent[]>([]);
  const [densityHistory, setDensityHistory] = useState<number[]>(Array(30).fill(0));
  const [peakDensity, setPeakDensity] = useState<{val: number, zone: string}>({val: 0, zone: "N/A"});

  useEffect(() => {
    const fetchLatestAlert = async () => {
      try {
        const res = await fetch("/api/webhook");
        const json = await res.json();
        
        if (json.data && json.data.timestamp) {
          const data = json.data;
          
          setLatestAlert((prev: any) => {
            if (data.density_per_m2 > peakDensity.val) {
              setPeakDensity({ val: data.density_per_m2, zone: data.zone_name });
            }

            setDensityHistory(curr => [...curr, data.density_per_m2 || 0].slice(-30));

            if (!prev || prev.timestamp !== data.timestamp) {
              if (data.status !== "SAFE" && data.alert_level) {
                setAlertHistory(history => {
                  const newEvent = { ...data, id: data.timestamp, acknowledged: false };
                  if (history.find(h => h.id === newEvent.id)) return history;
                  // Strictly limit to 10 items in memory to prevent bloat
                  return [newEvent, ...history].slice(0, 10);
                });
              }
              return data;
            }
            return prev;
          });
        }
      } catch (error) {
        console.error("Failed to fetch telemetry", error);
      }
    };

    fetchLatestAlert();
    const interval = setInterval(fetchLatestAlert, 2000);
    return () => clearInterval(interval);
  }, [peakDensity]);

  const isSafe = !latestAlert || latestAlert.status === "SAFE" || !latestAlert.alert_level;
  const isCritical = latestAlert?.alert_level === "CRITICAL";
  
  const rateOfChange = useMemo(() => {
    if (densityHistory.length < 2) return 0;
    const current = densityHistory[densityHistory.length - 1];
    const prev = densityHistory[densityHistory.length - 2];
    return ((current - prev) * 30).toFixed(1);
  }, [densityHistory]);

  const activeAlertsCount = alertHistory.filter(a => !a.acknowledged).length;
  
  const acknowledgeAlert = (id: string) => {
    setAlertHistory(history => 
      history.map(alert => alert.id === id ? { ...alert, acknowledged: true } : alert)
    );
  };

  const getRiskColor = (level: string) => {
    if (level === "CRITICAL") return "text-red-500 bg-red-500/10 border-red-500/30";
    if (level === "MODERATE" || level === "HIGH") return "text-amber-500 bg-amber-500/10 border-amber-500/30";
    return "text-emerald-500 bg-emerald-500/10 border-emerald-500/30";
  };

  const maxScale = Math.max(...densityHistory, 5.0);
  const graphPoints = densityHistory.map((val, idx) => `${(idx / 29) * 100},${100 - ((val / maxScale) * 100)}`).join(" ");

  return (
    // Hard lock the screen height to 100vh so the body never scrolls
    <div className="h-screen bg-[#09090b] text-slate-300 p-4 font-sans flex flex-col gap-4 overflow-hidden">
      
      {/* PANEL 1: Top Bar */}
      <header className="grid grid-cols-5 gap-4 bg-[#11141d] border border-slate-800 p-4 rounded-xl items-center shadow-sm shrink-0">
        <div className="col-span-1 flex items-center gap-3 border-r border-slate-800 pr-4">
          <Scan size={28} className="text-blue-500" />
          <div>
            <h1 className="text-lg font-bold text-white leading-tight">CrowdSafe</h1>
            <p className="text-xs text-slate-500">{new Date().toLocaleTimeString()} // Live</p>
          </div>
        </div>
        
        <div className="col-span-1 px-4">
          <div className="text-xs text-slate-500 mb-1">Total View Count</div>
          <div className="text-3xl font-semibold text-white">{Math.round(latestAlert?.head_count || 0)} <span className="text-sm text-slate-600 font-normal">people</span></div>
        </div>

        <div className="col-span-1 px-4 border-l border-slate-800">
          <div className="text-xs text-slate-500 mb-1">Peak Density (Right Now)</div>
          <div className="text-xl font-semibold text-slate-200">
            {latestAlert?.density_per_m2?.toFixed(1) || "0.0"} <span className="text-xs text-slate-500">p/m²</span>
          </div>
          <div className="text-[10px] text-slate-500 truncate mt-0.5">ZONE: {latestAlert?.zone_name || "N/A"}</div>
        </div>

        <div className="col-span-1 px-4 border-l border-slate-800">
          <div className="text-xs text-slate-500 mb-1">Active Alerts</div>
          <div className="text-3xl font-semibold text-white flex items-center gap-2">
            {activeAlertsCount} 
            {activeAlertsCount > 0 && <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />}
          </div>
        </div>

        <div className={`col-span-1 flex justify-center items-center rounded-lg border ${getRiskColor(latestAlert?.alert_level)} h-full p-2`}>
          <div className="text-center">
            <div className="text-[10px] uppercase font-bold tracking-widest opacity-80 mb-1">Overall Risk</div>
            <div className="font-bold flex items-center justify-center gap-2 text-sm uppercase">
              {isSafe ? <ShieldCheck size={18} /> : <AlertTriangle size={18} />}
              {isSafe ? "Normal Flow" : latestAlert.alert_level}
            </div>
          </div>
        </div>
      </header>

      {/* Workspace Grid - Takes up remaining height without overflowing */}
      <div className="grid grid-cols-12 gap-4 flex-1 min-h-0">
        
        {/* LEFT COLUMN: Visuals */}
        <div className="col-span-7 flex flex-col gap-4 min-h-0">
          
          {/* PANEL 2: Live Camera Feed */}
          <div className="bg-[#11141d] border border-slate-800 rounded-xl p-3 flex-1 flex flex-col min-h-0">
            <div className="flex justify-between items-center mb-3 shrink-0">
              <span className="text-xs font-semibold text-slate-400 flex items-center gap-2"><Video size={14}/> CAM-01: B-Block Lobby</span>
              <span className="text-xs text-slate-500 border border-slate-700 px-2 rounded">Heatmap Overlay: ON</span>
            </div>
            <div className={`relative flex-1 bg-black rounded border ${isCritical ? 'border-red-900/50' : 'border-slate-800'} overflow-hidden`}>
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(239,68,68,0.1),transparent_60%)] pointer-events-none" />
              <div className="absolute top-4 left-4 bg-black/80 px-2 py-1 border border-slate-700 text-xs font-mono text-white rounded">
                Zone Count: {Math.round(latestAlert?.head_count || 0)}
              </div>
              <div className="flex h-full items-center justify-center text-slate-700 text-sm font-mono text-center px-10">
                [ OpenCV Video Stream Placeholder ]<br/>
                Congestion cluster boxes will render here.
              </div>
            </div>
          </div>

          {/* PANEL 3: Floor Plan Map */}
          <div className="bg-[#11141d] border border-slate-800 rounded-xl p-3 h-48 shrink-0 flex flex-col">
            <span className="text-xs font-semibold text-slate-400 flex items-center gap-2 mb-3"> Facility Map: B-Block</span>
            <div className="flex-1 bg-[#0a0c10] border border-slate-800 rounded relative overflow-hidden p-4 grid grid-cols-3 gap-2">
              <div className="col-span-1 border border-slate-700 bg-slate-800/20 rounded flex flex-col items-center justify-center relative">
                <span className="text-[10px] text-slate-500 absolute top-2 left-2">ENTRANCE</span>
                <span className="text-emerald-500 font-bold">1.2 <span className="text-[10px]">p/m²</span></span>
              </div>
              <div className={`col-span-1 border rounded flex flex-col items-center justify-center relative ${!isSafe ? 'border-red-500/50 bg-red-500/10' : 'border-slate-700 bg-slate-800/20'}`}>
                <span className="text-[10px] text-slate-500 absolute top-2 left-2">MAIN CORRIDOR</span>
                <span className={`${!isSafe ? 'text-red-500' : 'text-slate-300'} font-bold`}>{latestAlert?.density_per_m2?.toFixed(1) || "0.0"} <span className="text-[10px]">p/m²</span></span>
              </div>
              <div className="col-span-1 border border-slate-700 bg-slate-800/20 rounded flex flex-col items-center justify-center relative">
                <span className="text-[10px] text-slate-500 absolute top-2 left-2">EXIT ROUTE (OPEN)</span>
                <span className="text-emerald-500 font-bold">0.5 <span className="text-[10px]">p/m²</span></span>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Analytics & Actions */}
        <div className="col-span-5 flex flex-col gap-4 min-h-0">
          
          {/* PANEL 5: Trends & Rate of Change */}
          <div className="bg-[#11141d] border border-slate-800 rounded-xl p-4 flex flex-col shrink-0">
            <div className="flex justify-between items-center mb-4">
              <span className="text-xs font-semibold text-slate-400 flex items-center gap-2"><Activity size={14}/> Density Trends</span>
              <div className={`text-xs flex items-center gap-1 font-bold ${Number(rateOfChange) > 0 ? 'text-amber-500' : 'text-emerald-500'}`}>
                {Number(rateOfChange) > 0 ? <TrendingUp size={14}/> : <TrendingDown size={14}/>}
                {rateOfChange} p/min
              </div>
            </div>
            <div className="h-20 w-full relative mb-2">
              <svg className="w-full h-full overflow-visible" preserveAspectRatio="none" viewBox="0 0 100 100">
                <polyline fill="none" stroke={isCritical ? "#ef4444" : "#3b82f6"} strokeWidth="2" vectorEffect="non-scaling-stroke" points={graphPoints} />
              </svg>
              <div className="absolute bottom-0 left-0 w-full border-t border-slate-800 border-dashed" />
            </div>
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>Past 10 mins</span>
              <span>Today's Peak: {peakDensity.val.toFixed(1)} p/m² ({peakDensity.zone})</span>
            </div>
          </div>

          {/* PANEL 4: Alerts & Incidents - Hard constrained height */}
          <div className="bg-[#11141d] border border-slate-800 rounded-xl p-4 flex flex-col flex-1 min-h-0">
            <span className="text-xs font-semibold text-slate-400 flex items-center gap-2 mb-3 shrink-0">
              <History size={14}/> Incident Queue (Max 10)
            </span>
            
            {/* The overflow-y-auto enables the internal scrollbar */}
            <div className="space-y-2 overflow-y-auto flex-1 pr-1 custom-scrollbar">
              {alertHistory.length === 0 ? (
                <div className="text-xs text-slate-500 text-center py-8 shrink-0">Queue empty. No active alerts.</div>
              ) : (
                alertHistory.map((alert) => (
                  <div key={alert.id} className={`shrink-0 border rounded-lg p-3 text-xs transition-opacity ${alert.acknowledged ? 'opacity-50 border-slate-800 bg-slate-900/50' : 'border-red-900/50 bg-[#1a1415]'}`}>
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex items-center gap-2">
                        <span className={`px-1.5 py-0.5 rounded font-bold ${alert.alert_level === "CRITICAL" ? "bg-red-500 text-white" : "bg-amber-500 text-black"}`}>
                          {alert.alert_level}
                        </span>
                        <span className="text-slate-400 font-mono">{new Date(alert.timestamp).toLocaleTimeString()}</span>
                      </div>
                      {!alert.acknowledged && (
                        <button 
                          onClick={() => acknowledgeAlert(alert.id)}
                          className="text-slate-400 hover:text-white flex items-center gap-1 bg-slate-800 hover:bg-slate-700 px-2 py-1 rounded transition-colors"
                        >
                          <CheckCircle2 size={12} /> ACK
                        </button>
                      )}
                    </div>
                    
                    <div className="text-slate-300 font-medium mb-1">{alert.zone_name}</div>
                    
                    <div className="grid grid-cols-2 gap-2 text-slate-400 mb-2">
                      <div>Trigger Density: <span className="text-slate-200">{alert.density_per_m2.toFixed(1)} p/m²</span></div>
                      <div>Trigger Count: <span className="text-slate-200">{Math.round(alert.head_count)}</span></div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* PANEL 7: System Health */}
          <div className="bg-[#11141d] border border-slate-800 rounded-xl p-3 grid grid-cols-4 gap-2 text-center text-[10px] text-slate-400 shrink-0">
            <div className="flex flex-col items-center justify-center border-r border-slate-800">
              <Cpu size={14} className="mb-1 text-slate-500"/>
              MCNN v2.1
            </div>
            <div className="flex flex-col items-center justify-center border-r border-slate-800">
              <Network size={14} className="mb-1 text-slate-500"/>
              45 FPS
            </div>
            <div className="flex flex-col items-center justify-center border-r border-slate-800">
              <Activity size={14} className="mb-1 text-slate-500"/>
              22ms Latency
            </div>
            <div className="flex flex-col items-center justify-center text-emerald-500 font-semibold">
              <ShieldCheck size={14} className="mb-1"/>
              CAM: ONLINE
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}