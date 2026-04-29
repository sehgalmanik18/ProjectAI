/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { 
  FaceLandmarker, 
  FilesetResolver,
  FaceLandmarkerResult
} from '@mediapipe/tasks-vision';
import { 
  AlertTriangle, 
  Camera, 
  Eye, 
  Layout, 
  Volume2,
  VolumeX,
  Settings,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useInterval } from 'react-use';

// --- CONSTANTS ---
const DEFAULT_EAR_THRESHOLD = 0.20;
const DEFAULT_DROWSY_TIME_MS = 1500; // 1.5 seconds

// Landmarks based on standard face mesh indices
const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];

// --- UTILS ---
const calculateDistance = (p1: { x: number; y: number }, p2: { x: number; y: number }) => {
  return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
};

const calculateEAR = (landmarks: any[], eyeIndices: number[]) => {
  const p1 = landmarks[eyeIndices[0]];
  const p2 = landmarks[eyeIndices[1]];
  const p3 = landmarks[eyeIndices[2]];
  const p4 = landmarks[eyeIndices[3]];
  const p5 = landmarks[eyeIndices[4]];
  const p6 = landmarks[eyeIndices[5]];

  const vertical1 = calculateDistance(p2, p6);
  const vertical2 = calculateDistance(p3, p5);
  const horizontal = calculateDistance(p1, p4);

  return (vertical1 + vertical2) / (2.0 * horizontal);
};

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [earThreshold, setEarThreshold] = useState(DEFAULT_EAR_THRESHOLD);
  const [triggerDelay, setTriggerDelay] = useState(DEFAULT_DROWSY_TIME_MS);
  const [ear, setEar] = useState(0);
  const [isDrowsy, setIsDrowsy] = useState(false);
  const [focusStatus, setFocusStatus] = useState<'Forward' | 'Left' | 'Right'>('Forward');
  const [isAlarmEnabled, setIsAlarmEnabled] = useState(true);
  const [drowsyStartTime, setDrowsyStartTime] = useState<number | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const oscillatorRef = useRef<OscillatorNode | null>(null);

  // --- AUDIO ALERT SYSTEM ---
  const playAlarm = useCallback(() => {
    if (!isAlarmEnabled) return;
    if (!audioCtxRef.current) {
      audioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume();
    }
    if (!oscillatorRef.current) {
      const osc = audioCtxRef.current.createOscillator();
      const gain = audioCtxRef.current.createGain();
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(880, audioCtxRef.current.currentTime);
      gain.gain.setValueAtTime(0.1, audioCtxRef.current.currentTime);
      osc.connect(gain);
      gain.connect(audioCtxRef.current.destination);
      osc.start();
      oscillatorRef.current = osc;
    }
  }, [isAlarmEnabled]);

  const stopAlarm = useCallback(() => {
    if (oscillatorRef.current) {
      oscillatorRef.current.stop();
      oscillatorRef.current = null;
    }
  }, []);

  // --- INITIALIZATION ---
  useEffect(() => {
    const setupFaceMesh = async () => {
      const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );
      const faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode: "VIDEO",
        numFaces: 1
      });
      landmarkerRef.current = faceLandmarker;
      setIsLoaded(true);
    };
    setupFaceMesh();
  }, []);

  // --- CAMERA SETUP ---
  const startCamera = async () => {
    if (!videoRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720, facingMode: 'user' } 
      });
      videoRef.current.srcObject = stream;
      videoRef.current.play();
      setCameraActive(true);
    } catch (err) {
      console.error("Camera error:", err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setCameraActive(false);
      setIsDrowsy(false);
      setDrowsyStartTime(null);
      stopAlarm();
    }
  };

  // --- PROCESSING LOOP ---
  const processFrame = useCallback(() => {
    if (!videoRef.current || !landmarkerRef.current || !cameraActive) return;

    const startTimeMs = performance.now();
    const results = landmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
      const landmarks = results.faceLandmarks[0];

      // EAR Calculation
      const leftEar = calculateEAR(landmarks, LEFT_EYE);
      const rightEar = calculateEAR(landmarks, RIGHT_EYE);
      const avgEar = (leftEar + rightEar) / 2;
      setEar(avgEar);

      // Drowsiness Detection
      if (avgEar < earThreshold) {
        if (drowsyStartTime === null) {
          setDrowsyStartTime(Date.now());
        } else {
          const elapsed = Date.now() - drowsyStartTime;
          if (elapsed > triggerDelay) {
            setIsDrowsy(true);
            playAlarm();
          }
        }
      } else {
        setDrowsyStartTime(null);
        setIsDrowsy(false);
        stopAlarm();
      }

      // Focus Detection (Head orientation proxy)
      const nose = landmarks[1];
      const leftEyeInner = landmarks[463];
      const rightEyeInner = landmarks[243];
      
      // Simple logic: if nose is too far from midpoint of eyes relative to horizontal scale
      // Actually, nose.x is normalized 0-1.
      if (nose.x < 0.42) {
        setFocusStatus('Right');
      } else if (nose.x > 0.58) {
        setFocusStatus('Left');
      } else {
        setFocusStatus('Forward');
      }
    } else {
      // No face detected
      setIsDrowsy(false);
      stopAlarm();
    }
  }, [cameraActive, drowsyStartTime, playAlarm, stopAlarm]);

  useInterval(processFrame, 50);

  return (
    <div className="min-h-screen bg-[#0A0A0B] text-slate-200 font-sans p-6 overflow-x-hidden flex flex-col gap-6 selection:bg-[#10b981] selection:text-black">
      {/* Header Section */}
      <header className="flex justify-between items-center bg-[#161618] border border-white/10 rounded-2xl px-6 py-4 shadow-xl">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${cameraActive ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></div>
          <h1 className="text-xl font-bold tracking-tight text-white uppercase">Guardian<span className="text-emerald-500">Drive</span> <span className="text-[10px] font-mono text-slate-500 align-top opacity-60">DMS v1.4</span></h1>
        </div>
        <div className="flex gap-6 items-center">
          <div className="hidden md:block text-right">
            <p className="text-[10px] uppercase text-slate-500 font-semibold tracking-widest">System Status</p>
            <p className={`text-sm font-mono uppercase ${cameraActive ? 'text-emerald-400' : 'text-red-400'}`}>
              {cameraActive ? 'Operational / Live' : 'Standby / Offline'}
            </p>
          </div>
          <div className="hidden md:block h-8 w-[1px] bg-white/10"></div>
          <div className="flex items-center gap-3">
            <button 
              onClick={() => setIsAlarmEnabled(!isAlarmEnabled)}
              className="p-2 hover:bg-white/5 rounded-xl transition-colors text-slate-400 hover:text-white border border-white/5"
            >
              {isAlarmEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
            </button>
            <button className="p-2 hover:bg-white/5 rounded-xl transition-colors text-slate-400 hover:text-white border border-white/5">
              <Settings size={18} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 flex-grow">
        {/* Live Camera Feed (Large Bento Box) */}
        <div className="md:col-span-8 md:row-span-4 bg-[#161618] border border-white/10 rounded-3xl relative overflow-hidden flex items-center justify-center shadow-2xl min-h-[400px]">
          {!cameraActive && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/60 backdrop-blur-md">
              <div className="w-20 h-20 bg-emerald-500/20 rounded-full flex items-center justify-center mb-6 animate-pulse border border-emerald-500/20">
                <Camera className="w-10 h-10 text-emerald-500" />
              </div>
              <h2 className="text-2xl font-black mb-2 italic uppercase">Guardian Ready</h2>
              <p className="text-slate-400 text-sm mb-8 text-center px-12 max-w-md">Initialize optical telemetry tracking to begin driver vigilance monitoring.</p>
              <button 
                onClick={startCamera}
                disabled={!isLoaded}
                className="px-10 py-4 bg-emerald-500 hover:bg-emerald-400 text-black font-black uppercase tracking-tighter rounded-2xl transition-all hover:scale-105 disabled:opacity-50 disabled:grayscale shadow-[0_0_30px_rgba(16,185,129,0.3)]"
              >
                {isLoaded ? 'Initialize System' : 'Syncing Neural Kernels...'}
              </button>
            </div>
          )}

          <video 
            ref={videoRef} 
            className={`w-full h-full object-cover scale-x-[-1] transition-opacity duration-1000 ${cameraActive ? 'opacity-100' : 'opacity-0'}`}
            playsInline
          />

          <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent pointer-events-none z-[5]"></div>

          {cameraActive && (
            <>
              <div className="absolute top-6 left-6 z-20 flex gap-2">
                <span className="px-3 py-1 bg-black/50 backdrop-blur-md rounded-full text-[10px] font-bold border border-white/20 uppercase">IR-SENSOR-OPTIC</span>
                <span className="px-3 py-1 bg-emerald-500/20 text-emerald-400 backdrop-blur-md rounded-full text-[10px] font-bold border border-emerald-500/40 uppercase">FACE TRACKING ACTIVE</span>
              </div>
              
              <div className="absolute bottom-8 left-8 z-20 flex flex-col gap-0">
                <p className="text-xs font-mono text-slate-400 uppercase tracking-widest mb-1">Focus State</p>
                <div className="flex items-center gap-2">
                   <p className="text-3xl font-black text-white italic uppercase tracking-tighter">
                    Looking {focusStatus}
                  </p>
                  {focusStatus === 'Forward' ? (
                    <span className="text-emerald-400 text-3xl font-black drop-shadow-[0_0_10px_#10b981]">✓</span>
                  ) : (
                    <span className="text-red-500 text-3xl font-black drop-shadow-[0_0_10px_#ef4444]">×</span>
                  )}
                </div>
              </div>

               <AnimatePresence>
                {isDrowsy && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="absolute inset-0 z-30 flex items-center justify-center bg-red-600/20 backdrop-blur-[2px]"
                  >
                    <div className="bg-red-600 text-white px-10 py-5 rounded-3xl flex flex-col items-center gap-2 shadow-[0_0_60px_rgba(220,38,38,0.6)] border-4 border-red-400 animate-bounce">
                      <AlertTriangle className="w-12 h-12" />
                      <span className="text-2xl font-black italic uppercase tracking-tighter">DROWSINESS ALERT</span>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <button 
                onClick={stopCamera}
                className="absolute top-6 right-6 z-20 p-3 bg-white/5 hover:bg-red-500/20 text-slate-400 hover:text-red-500 rounded-2xl transition-all border border-white/10 hover:border-red-500/30"
              >
                <Layout className="w-5 h-5" />
              </button>
            </>
          )}

          {/* Scantron Effect */}
          <div className="absolute inset-0 pointer-events-none opacity-[0.03] bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_2px,3px_100%] z-40" />
        </div>

        {/* Sensitivity Controls */}
        <div className="md:col-span-4 md:row-span-2 bg-[#161618] border border-white/10 rounded-3xl p-6 shadow-xl">
          <h3 className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-6">Sensitivity Settings</h3>
          <div className="space-y-6">
            <div>
              <div className="flex justify-between text-[11px] mb-2">
                <span className="text-slate-300">EAR Threshold</span>
                <span className="text-white font-mono">{earThreshold.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                min="0.10" 
                max="0.30" 
                step="0.01" 
                value={earThreshold}
                onChange={(e) => setEarThreshold(parseFloat(e.target.value))}
                className="w-full h-1 bg-slate-800 rounded-full appearance-none accent-emerald-500 cursor-pointer"
              />
            </div>
            <div>
              <div className="flex justify-between text-[11px] mb-2">
                <span className="text-slate-300">Trigger Delay</span>
                <span className="text-white font-mono">{(triggerDelay / 1000).toFixed(1)}s</span>
              </div>
              <input 
                type="range" 
                min="500" 
                max="3000" 
                step="100" 
                value={triggerDelay}
                onChange={(e) => setTriggerDelay(parseInt(e.target.value))}
                className="w-full h-1 bg-slate-800 rounded-full appearance-none accent-emerald-500 cursor-pointer"
              />
            </div>
            <p className="text-[9px] text-slate-500 font-mono italic">Adjust thresholds to match your ocular dimensions and blink patterns.</p>
          </div>
        </div>

        {/* Metric: EAR (Eye Aspect Ratio) */}
        <div className="md:col-span-4 md:row-span-2 bg-[#161618] border border-white/10 rounded-3xl p-6 flex flex-col justify-between shadow-xl">
          <div className="flex justify-between items-start">
            <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">Oculometric Aperture (EAR)</span>
            <Eye className={`w-5 h-5 ${ear < earThreshold ? 'text-red-500' : 'text-emerald-500'}`} />
          </div>
          <div>
            <div className={`text-6xl font-mono font-black tracking-tighter mb-4 ${ear < earThreshold ? 'text-red-500' : 'text-white'}`}>
              {ear === 0 ? '---' : ear.toFixed(2)}
            </div>
            <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden mb-3">
              <motion.div 
                className={`h-full ${ear < earThreshold ? 'bg-red-500' : 'bg-emerald-500'}`}
                animate={{ width: `${Math.min(ear * 300, 100)}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] font-mono font-bold">
              <span className="text-slate-500 uppercase">Threshold: {earThreshold.toFixed(2)}</span>
              <span className={ear < earThreshold ? 'text-red-500' : 'text-emerald-500'}>
                {ear === 0 ? 'SCANNING...' : ear < earThreshold ? 'CRITICAL' : 'OPTIMAL'}
              </span>
            </div>
          </div>
        </div>

        {/* Metric: Alert Status */}
        <div className={`md:col-span-4 md:row-span-2 border rounded-3xl p-6 flex flex-col justify-between transition-all duration-500 shadow-xl ${isDrowsy ? 'bg-red-500/10 border-red-500 shadow-red-500/20' : 'bg-[#161618] border-white/10'}`}>
          <div className="flex justify-between items-start">
            <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">Vigilance Analytics</span>
            <div className={`w-3 h-3 rounded-full ${isDrowsy ? 'bg-red-500 animate-ping' : 'bg-emerald-500 opacity-30 shadow-[0_0_10px_#10b981]'}`}></div>
          </div>
          <div>
            <p className="text-xs text-slate-500 font-bold uppercase mb-1">Fatigue Level</p>
            <p className={`text-4xl font-black italic uppercase tracking-tighter ${isDrowsy ? 'text-red-500' : 'text-slate-200'}`}>
              {isDrowsy ? 'Critical Risk' : 'Low Risk'}
            </p>
            <p className="text-[10px] font-mono text-slate-500 mt-2 uppercase">
              Sensor uptime: {cameraActive ? 'Live Sync' : 'Static'}
            </p>
          </div>
        </div>

        {/* Event Stream (Alert Log) */}
        <div className="md:col-span-6 row-span-2 bg-[#161618] border border-white/10 rounded-3xl p-6 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-slate-400 text-xs font-semibold uppercase tracking-wider">Neural Event Stream</h3>
            <span className="text-[9px] px-2 py-0.5 bg-white/5 rounded-full text-slate-500 font-bold uppercase">Auto-purge v2</span>
          </div>
          <div className="space-y-4 font-mono text-[11px] max-h-[140px] overflow-y-auto pr-2 custom-scrollbar">
            <LogEntry time="07:54:12" msg="System Initialized" type="info" />
            <LogEntry time="07:54:20" msg="Neural Buffers Synchronized" type="info" />
            {isDrowsy && <LogEntry time="LIVE" msg="ALERT: Fatigued Pattern Detected" type="danger" />}
            {focusStatus !== 'Forward' && <LogEntry time="LIVE" msg={`ATTENTION: Peripheral view ${focusStatus}`} type="warning" />}
            <div className="opacity-40"><LogEntry time="07:50:00" msg="Background Calibration complete" type="info" /></div>
          </div>
        </div>

        {/* Device Stats (System Info) */}
        <div className="md:col-span-6 row-span-2 bg-[#161618] border border-white/10 rounded-3xl p-6 flex items-stretch gap-6 shadow-xl">
          <div className="flex-1 bg-emerald-500 rounded-2xl p-4 flex flex-col justify-between text-black group relative overflow-hidden">
            <div className="z-10">
              <h3 className="text-[10px] font-black uppercase tracking-widest opacity-60 mb-3">Unit Telemetry</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[9px] font-bold opacity-60 uppercase">D-Engine Load</p>
                  <p className="text-xl font-black font-mono tracking-tighter">14.2%</p>
                </div>
                <div>
                  <p className="text-[9px] font-bold opacity-60 uppercase">Stream Rate</p>
                  <p className="text-xl font-black font-mono tracking-tighter">60.2</p>
                </div>
              </div>
            </div>
            <div className="absolute -bottom-4 -right-4 w-16 h-16 bg-white/20 rounded-full blur-xl group-hover:scale-150 transition-transform duration-700" />
          </div>
          
          <div className="flex-1 flex flex-col justify-center gap-4">
            <div className="flex flex-col gap-1">
              <span className="text-[10px] font-mono text-slate-500 uppercase font-bold tracking-widest">Calibration</span>
              <div className="flex gap-1">
                {[1,2,3,4,5,6].map(i => (
                  <div key={i} className={`h-1.5 flex-1 rounded-full ${cameraActive ? 'bg-emerald-500' : 'bg-slate-800'}`} />
                ))}
              </div>
            </div>
            <button 
              onClick={() => {
                playAlarm();
                setTimeout(stopAlarm, 1000);
              }}
              className="w-full py-3 bg-white/5 hover:bg-white/10 text-emerald-500 font-black rounded-xl text-[10px] uppercase tracking-widest border border-white/5 transition-all active:scale-95"
            >
              Force Alarm Test
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function LogEntry({ time, msg, type }: { time: string; msg: string; type: 'info' | 'warning' | 'danger' }) {
  const colors = {
    info: 'text-slate-500',
    warning: 'text-amber-400',
    danger: 'text-red-500 font-black animate-pulse'
  };

  return (
    <div className="flex gap-4 border-b border-white/5 pb-2">
      <span className="text-slate-600 shrink-0">[{time}]</span>
      <span className={`text-[11px] font-bold tracking-tight uppercase ${colors[type]}`}>{msg}</span>
    </div>
  );
}
