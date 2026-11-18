import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Car,
  Camera,
  Radio,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Eye,
  ShieldAlert,
  Film,
  Zap,
  RotateCcw,
  Check,
  UploadCloud,
  Loader2,
  Trash2,
  Maximize2,
  Minimize2,
  ExternalLink,
} from 'lucide-react';
import {
  getCameras,
  getCameraZone,
  setCameraZone,
  getAvailableVideos,
  attachVideoToCamera,
  triggerAIAnalysis,
  getAIAnalysisStatus,
  stopAIAnalysis,
  createCamera,
  deleteCamera,
  uploadCameraVideo,
  getUser,
  getAnprEngine,
  setAnprEngine,
  addWatchlistPlate,
  getWatchlist,
} from '../api.js';
import { useWebSocket } from '../useWebSocket.js';

// Default normalized 4-point zone (trapezoid across roadway)
const DEFAULT_ZONE = [
  [0.05, 0.44], // Top-Left
  [0.95, 0.44], // Top-Right
  [0.98, 0.58], // Bottom-Right
  [0.02, 0.58], // Bottom-Left
];

export default function TrafficANPRPage() {
  const [cameras, setCameras] = useState([]);
  const [selectedCamId, setSelectedCamId] = useState('');
  const [availableVideos, setAvailableVideos] = useState([]);
  const [feedMode, setFeedMode] = useState('stream'); // 'stream' or 'video'
  const [aiStatus, setAiStatus] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);

  // Zone configuration state
  const [isEditingZone, setIsEditingZone] = useState(false);
  const [zonePoints, setZonePoints] = useState(DEFAULT_ZONE);
  const [isSavingZone, setIsSavingZone] = useState(false);
  const [activeHandle, setActiveHandle] = useState(null);

  // ANPR Live Dashboard State (Deduplicated Unique Counts)
  const [kpiPlates, setKpiPlates] = useState(4);
  const [kpiNoRead, setKpiNoRead] = useState(0);
  const [kpiVehicles, setKpiVehicles] = useState(0);

  // Current Plate Zoom Window
  const [currentPlate, setCurrentPlate] = useState({
    plate: 'BDR775',
    confidence: 78,
    status: 'READING', // 'READING', 'CONFIRMED', 'HOTLIST'
    reads: 2,
    crop: null,
    vehicleType: 'VEHICLE',
    isHotlist: false,
    reason: '',
  });

  // Recent Plates Feed
  const [recentPlates, setRecentPlates] = useState([
    {
      id: 'rec-01',
      plate: 'BWU614',
      vehicleType: 'VEHICLE',
      time: '15:03:22',
      confidence: 85,
      crop: null,
      isHotlist: false,
    },
    {
      id: 'rec-02',
      plate: 'BKH147',
      vehicleType: 'VEHICLE',
      time: '15:02:49',
      confidence: 98,
      crop: null,
      isHotlist: false,
    },
    {
      id: 'rec-03',
      plate: 'BHD604',
      vehicleType: 'VEHICLE',
      time: '15:01:14',
      confidence: 89,
      crop: null,
      isHotlist: false,
    },
    {
      id: 'rec-04',
      plate: 'MH12DE1433',
      vehicleType: 'SUV',
      time: '14:58:30',
      confidence: 94,
      crop: null,
      isHotlist: true,
      reason: 'Stolen Vehicle - Police APB',
    },
  ]);

  const [notification, setNotification] = useState(null);
  const [anprEngine, setAnprEngineState] = useState('cct_s');
  const videoViewportRef = useRef(null);
  const seenPlatesRef = useRef(new Map());
  const lastHotlistAlertRef = useRef(new Map());
  const user = getUser();

  const showToast = useCallback((msg, type = 'info') => {
    setNotification({ msg, type, id: Date.now() });
    setTimeout(() => setNotification(null), 6500);
  }, []);

  // Load ONLY traffic cameras, available sample videos, and active ANPR engine
  const loadData = useCallback(async () => {
    try {
      const allCams = await getCameras();
      if (Array.isArray(allCams)) {
        // Only show traffic-type cameras — completely separate from the security Cameras section
        const trafficCams = allCams.filter(c => c.camera_type === 'traffic');
        setCameras(trafficCams);
        if (!selectedCamId && trafficCams.length > 0) {
          setSelectedCamId(trafficCams[0].id);
        }
      }
    } catch (_) {}

    try {
      const vids = await getAvailableVideos();
      if (Array.isArray(vids)) setAvailableVideos(vids);
    } catch (_) {}

    try {
      const engData = await getAnprEngine();
      if (engData?.active_engine) {
        setAnprEngineState(engData.active_engine);
      }
    } catch (_) {}
  }, [selectedCamId]);

  const handleSwitchEngine = async (newEngine) => {
    try {
      await setAnprEngine(newEngine);
      setAnprEngineState(newEngine);
      showToast(`Switched ANPR OCR Model to: ${newEngine.toUpperCase()}`, 'info');
    } catch (err) {
      console.error('Failed to switch engine:', err);
    }
  };

  const handleAddToWatchlist = async (plateText, defaultReason = 'Flagged by Operator') => {
    if (!plateText || plateText === 'UNKNOWN') return;
    try {
      await addWatchlistPlate({
        plate: plateText,
        reason: defaultReason,
        owner: 'Traffic Camera Operator',
        notes: 'Flagged from Traffic ANPR live monitoring screen',
      });
      showToast(`Vehicle [${plateText}] enrolled into Watchlist!`, 'success');
      setCurrentPlate(prev => (prev.plate === plateText ? { ...prev, isHotlist: true, reason: defaultReason } : prev));
      setRecentPlates(prev => prev.map(p => (p.plate === plateText ? { ...p, isHotlist: true, reason: defaultReason } : p)));
    } catch (err) {
      showToast(`Failed to add plate to watchlist: ${err.message}`, 'critical');
    }
  };

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Load zone for selected camera
  useEffect(() => {
    if (!selectedCamId) return;
    getCameraZone(selectedCamId)
      .then(res => {
        if (res && Array.isArray(res.zone) && res.zone.length === 4) {
          setZonePoints(res.zone);
        } else {
          setZonePoints(DEFAULT_ZONE);
        }
      })
      .catch(() => {
        setZonePoints(DEFAULT_ZONE);
      });
  }, [selectedCamId]);

  const selectedCamIdRef = useRef(selectedCamId);
  useEffect(() => {
    selectedCamIdRef.current = selectedCamId;
  }, [selectedCamId]);

  // Handle incoming real-time WebSocket events (only for selected traffic camera)
  const handleWsMessage = useCallback((msg) => {
    if (!msg) return;
    const currentCam = selectedCamIdRef.current;
    const matchesCam = !currentCam || !msg.camera_id || msg.camera_id === currentCam;

    if (msg.type === 'anpr_zone_update' && matchesCam) {
      if (typeof msg.vehicles_in_zone === 'number') {
        setKpiVehicles(msg.vehicles_in_zone);
      }
    } else if (msg.type === 'anpr_no_read' && matchesCam) {
      setKpiNoRead(c => c + 1);
    } else if (msg.type === 'anpr_detection' && matchesCam) {
      const { plate, confidence, country_format, plate_crop, is_hotlist, reason, vehicle_type, vehicles_in_zone, is_new } = msg;

      if (typeof vehicles_in_zone === 'number') {
        setKpiVehicles(vehicles_in_zone);
      }

      if (!plate || plate === 'UNKNOWN') return;

      const confNum = Math.round(confidence || 75);
      const isHot = !!is_hotlist;
      const cleanPlate = plate.toUpperCase().replace(/[\s\-\.]/g, '');
      const now = new Date();
      const timeStr = now.toTimeString().split(' ')[0];
      const nowMs = Date.now();

      // Deduplication: check if this plate was already detected within the last 14 seconds or flagged by backend as not new
      const lastSeen = seenPlatesRef.current.get(cleanPlate) || 0;
      const isDuplicate = (is_new === false) || (nowMs - lastSeen < 14000);

      // Update recent sighting timestamp
      seenPlatesRef.current.set(cleanPlate, nowMs);

      // Update current Plate Zoom widget (live lens on active vehicle)
      setCurrentPlate(prev => ({
        plate: plate,
        confidence: confNum,
        status: isHot ? 'HOTLIST' : confNum > 80 ? 'CONFIRMED' : 'READING',
        reads: (prev.plate === plate ? (prev.reads || 1) + 1 : 1),
        crop: plate_crop || prev.crop,
        vehicleType: (vehicle_type || 'VEHICLE').toUpperCase(),
        isHotlist: isHot,
        reason: reason || '',
      }));

      if (isDuplicate) {
        // Vehicle is lingering or driving across region: update confidence and read count in-place, DO NOT add duplicate row or count
        setRecentPlates(prev => {
          let found = false;
          const updated = prev.map(rec => {
            const recClean = (rec.plate || '').toUpperCase().replace(/[\s\-\.]/g, '');
            if (recClean === cleanPlate) {
              found = true;
              return {
                ...rec,
                confidence: Math.max(rec.confidence, confNum),
                crop: plate_crop || rec.crop,
                reads: (rec.reads || 1) + 1,
                time: timeStr,
              };
            }
            return rec;
          });
          return found ? updated : prev;
        });
      } else {
        // Brand new vehicle entry: increment unique plate counter ONCE and prepend to recent plates
        setKpiPlates(c => c + 1);

        const newRec = {
          id: `anpr-${Date.now()}-${Math.random()}`,
          plate: plate,
          vehicleType: (vehicle_type || 'VEHICLE').toUpperCase(),
          time: timeStr,
          confidence: confNum,
          crop: plate_crop || null,
          isHotlist: isHot,
          reason: reason || '',
          reads: 1,
        };

        setRecentPlates(prev => [
          newRec,
          ...prev.filter(r => (r.plate || '').toUpperCase().replace(/[\s\-\.]/g, '') !== cleanPlate).slice(0, 19),
        ]);

        if (isHot) {
          const lastAlert = lastHotlistAlertRef.current.get(cleanPlate) || 0;
          if (nowMs - lastAlert > 8000) {
            lastHotlistAlertRef.current.set(cleanPlate, nowMs);
            showToast(`🚨 WATCHLIST HIT: Plate [${plate}] flagged as ${reason || 'Stolen / Wanted'}!`, 'critical');
          }
        }
      }

      // Ensure any hotlist vehicle triggers alert toast even during continuous region tracking
      if (isHot) {
        const lastAlert = lastHotlistAlertRef.current.get(cleanPlate) || 0;
        if (nowMs - lastAlert > 8000) {
          lastHotlistAlertRef.current.set(cleanPlate, nowMs);
          showToast(`🚨 WATCHLIST HIT: Plate [${plate}] flagged as ${reason || 'Stolen / Wanted'}!`, 'critical');
        }
      }
    } else if (msg.type === 'alert' && (msg.payload?.alert_type === 'watchlist_plate' || msg.payload?.title?.includes('WATCHLIST'))) {
      const p = msg.payload?.metadata?.plate || msg.payload?.title || 'HOTLIST VEHICLE';
      const r = msg.payload?.metadata?.reason || msg.payload?.description || 'Stolen Vehicle Hit';
      showToast(`🚨 WATCHLIST ALERT: ${p} (${r})`, 'critical');
    }
  }, []);

  useWebSocket(handleWsMessage);

  // Save modified zone to backend
  const handleSaveZone = async () => {
    if (!selectedCamId) return;
    setIsSavingZone(true);
    try {
      await setCameraZone(selectedCamId, zonePoints);
      setIsEditingZone(false);
      showToast('Monitoring Zone updated and saved to AI vision engine.', 'success');
    } catch (err) {
      alert(`Failed to save zone: ${err.message}`);
    } finally {
      setIsSavingZone(false);
    }
  };

  // Reset zone to standard highway road trapezoid
  const handleResetZone = () => {
    setZonePoints(DEFAULT_ZONE);
  };

  // Poll active AI status for selected camera
  useEffect(() => {
    if (!selectedCamId) return;
    let cancelled = false;
    const checkStatus = async () => {
      try {
        const st = await getAIAnalysisStatus(selectedCamId);
        if (!cancelled && st) setAiStatus(st);
      } catch (_) {}
    };
    checkStatus();
    const interval = setInterval(checkStatus, 2500);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [selectedCamId]);

  // Create a new dedicated traffic camera (independent from general cameras section)
  const handleAddTrafficCam = async () => {
    const id = `cam_traffic_${Date.now().toString(36)}`;
    try {
      await createCamera({
        id,
        name: 'Traffic ANPR Feed',
        rtsp_url: '',
        location: 'Highway / Road',
        source_type: 'video_file',
        camera_type: 'traffic',
      });
      await loadData();
      setSelectedCamId(id);
      showToast(`Created traffic camera [${id}]. Attach a video to begin.`, 'success');
    } catch (err) {
      alert(`Error creating traffic camera: ${err.message}`);
    }
  };

  // Delete selected traffic camera
  const handleDeleteTrafficCam = async () => {
    if (!selectedCamId) return;
    if (!confirm(`Delete traffic camera ${selectedCamId}? This will halt all AI processing and purge its alerts.`)) return;
    try {
      await stopAIAnalysis(selectedCamId).catch(() => {});
      await deleteCamera(selectedCamId);
      setSelectedCamId('');
      await loadData();
      showToast(`Deleted traffic camera [${selectedCamId}].`, 'info');
    } catch (err) {
      alert(`Delete error: ${err.message}`);
    }
  };

  // Start traffic AI inference (strictly vehicles & ANPR, no violence/fall)
  const handleStartTrafficAI = async () => {
    if (!selectedCamId) return;
    try {
      showToast('Warming up Traffic AI (vehicle detection & ANPR only)...', 'info');
      const job = await triggerAIAnalysis(selectedCamId, null, 'traffic');
      setAiStatus(job);
      showToast('Traffic ANPR active! Monitoring zone online.', 'success');
    } catch (err) {
      alert(`Start error: ${err.message}`);
    }
  };

  // Stop traffic AI inference
  const handleStopTrafficAI = async () => {
    if (!selectedCamId) return;
    try {
      await stopAIAnalysis(selectedCamId);
      setAiStatus(prev => ({ ...prev, status: 'stopped' }));
      showToast('Traffic AI stopped.', 'info');
    } catch (err) {
      alert(`Stop error: ${err.message}`);
    }
  };

  // Quick attach video helper
  const handleAttachVideo = async (filename) => {
    if (!selectedCamId || !filename) return;
    try {
      await attachVideoToCamera(selectedCamId, filename);
      await triggerAIAnalysis(selectedCamId, null, 'traffic');
      loadData();
      showToast(`Attached [${filename}] and activated Traffic ANPR.`, 'success');
    } catch (err) {
      alert(`Attach error: ${err.message}`);
    }
  };

  // Upload video file directly to traffic camera
  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';

    const targetCamId = selectedCamId || 'cam_traffic';
    const targetCam = cameras.find(c => c.id === targetCamId) || {
      id: targetCamId,
      name: 'Expressway Traffic & ANPR',
      location: 'Highway Corridor - Zone 4',
    };

    const formData = new FormData();
    formData.append('file', file);
    formData.append('camera_id', targetCamId);
    formData.append('name', targetCam.name || 'Traffic ANPR Feed');
    formData.append('location', targetCam.location || 'Highway Corridor');
    formData.append('fps', '30');
    formData.append('camera_type', 'traffic');
    formData.append('auto_analyze', 'true');

    try {
      setIsUploading(true);
      showToast(`Uploading ${file.name} to traffic camera...`, 'info');
      await uploadCameraVideo(formData);
      await loadData();
      setSelectedCamId(targetCamId);
      showToast(`Uploaded ${file.name}! Traffic AI running (car IDs & plates only).`, 'success');
    } catch (err) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  // Interactive SVG Polygon Dragging Handlers
  const handlePointerDown = (index, e) => {
    e.preventDefault();
    setActiveHandle(index);
  };

  const handlePointerMove = (e) => {
    if (activeHandle === null || !videoViewportRef.current) return;
    const rect = videoViewportRef.current.getBoundingClientRect();
    const rawX = (e.clientX - rect.left) / rect.width;
    const rawY = (e.clientY - rect.top) / rect.height;

    // Clamp normalized coordinates between 0.01 and 0.99
    const clampedX = Math.max(0.01, Math.min(0.99, Number(rawX.toFixed(3))));
    const clampedY = Math.max(0.01, Math.min(0.99, Number(rawY.toFixed(3))));

    setZonePoints(prev => {
      const next = [...prev];
      next[activeHandle] = [clampedX, clampedY];
      return next;
    });
  };

  const handlePointerUp = () => {
    setActiveHandle(null);
  };

  const selectedCam = cameras.find(c => c.id === selectedCamId) || { id: selectedCamId, name: selectedCamId };
  const hasVideo = !!selectedCam.video_url;

  // Format SVG polygon points string
  const svgPoints = zonePoints.map(p => `${p[0] * 100}%,${p[1] * 100}%`).join(' ');

  return (
    <div
      className="traffic-anpr-page"
      style={{
        padding: '16px 20px',
        color: '#f8fafc',
        maxWidth: 1600,
        margin: '0 auto',
      }}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      {/* Toast Alert Banner */}
      {notification && (
        <div
          style={{
            position: 'fixed',
            bottom: 24,
            right: 24,
            zIndex: 99999,
            background: notification.type === 'critical' ? '#dc2626' : '#2563eb',
            color: 'white',
            padding: '12px 20px',
            borderRadius: 8,
            boxShadow: '0 10px 30px rgba(0,0,0,0.6)',
            fontSize: 13,
            fontWeight: 700,
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <ShieldAlert size={20} />
          {notification.msg}
        </div>
      )}

      {/* Top Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Car size={22} color="#38bdf8" />
            <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, letterSpacing: '-0.01em' }}>
              Traffic Surveillance & Zone-Based ANPR
            </h2>
          </div>
          <span
            style={{
              background: 'rgba(56, 189, 248, 0.15)',
              color: '#38bdf8',
              border: '1px solid rgba(56, 189, 248, 0.3)',
              padding: '2px 8px',
              borderRadius: 10,
              fontSize: 11,
              fontWeight: 700,
            }}
          >
            FAST-ALPR ACTIVE
          </span>
        </div>

        {/* Traffic Camera Selector & Quick Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#94a3b8' }}>
            <Camera size={14} /> Traffic Feed:
            {cameras.length > 0 ? (
              <select
                value={selectedCamId}
                onChange={e => setSelectedCamId(e.target.value)}
                style={{
                  background: '#0f172a',
                  color: '#f8fafc',
                  border: '1px solid #334155',
                  borderRadius: 6,
                  padding: '4px 10px',
                  fontSize: 12,
                  fontFamily: 'monospace',
                  fontWeight: 600,
                }}
              >
                {cameras.map(c => (
                  <option key={c.id} value={c.id}>
                    {c.id} • {c.name}
                  </option>
                ))}
              </select>
            ) : (
              <button
                className="btn btn-sm btn-primary"
                style={{ fontSize: 11, padding: '3px 10px', fontWeight: 700 }}
                onClick={handleAddTrafficCam}
              >
                + Add Traffic Camera
              </button>
            )}
            {selectedCamId && (
              <button
                className="btn btn-sm btn-outline"
                style={{ padding: '4px 6px', color: '#f87171', borderColor: '#475569' }}
                onClick={handleDeleteTrafficCam}
                title={`Delete traffic camera ${selectedCamId}`}
              >
                <Trash2 size={13} />
              </button>
            )}
          </div>

          {/* Quick attach video if needed */}
          {availableVideos.length > 0 && (
            <select
              style={{
                background: '#0f172a',
                color: '#94a3b8',
                border: '1px solid #334155',
                borderRadius: 6,
                padding: '4px 8px',
                fontSize: 11,
              }}
              onChange={e => {
                if (e.target.value) handleAttachVideo(e.target.value);
              }}
              defaultValue=""
            >
              <option value="" disabled>🎬 Attach Video Sample...</option>
              {availableVideos.map(v => (
                <option key={v.filename} value={v.filename}>
                  {v.filename} ({v.size_mb} MB)
                </option>
              ))}
            </select>
          )}

          {/* Direct Video Upload to Traffic Camera */}
          <input
            type="file"
            ref={fileInputRef}
            accept="video/*"
            style={{ display: 'none' }}
            onChange={handleFileSelect}
          />
          <button
            className="btn btn-sm btn-primary"
            style={{
              padding: '4px 10px',
              fontSize: 11,
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              background: '#0284c7',
              borderColor: '#0284c7',
            }}
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            title="Upload an MP4/video file directly from your computer to this traffic camera"
          >
            {isUploading ? <Loader2 size={13} className="spin" /> : <UploadCloud size={13} />}
            {isUploading ? 'Uploading...' : 'Upload Video'}
          </button>

          {/* ANPR OCR Model Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>OCR:</span>
            <select
              style={{
                background: '#0f172a',
                color: '#38bdf8',
                border: '1px solid #0284c7',
                borderRadius: 6,
                padding: '4px 8px',
                fontSize: 11,
                fontWeight: 600,
                outline: 'none',
                cursor: 'pointer',
              }}
              value={anprEngine}
              onChange={e => handleSwitchEngine(e.target.value)}
              title="Select ANPR neural engine (New high-accuracy model or preserved legacy model)"
            >
              <option value="cct_s">⚡ CCT-S v2 (High-Accuracy New Model)</option>
              <option value="cct_xs">🛡️ CCT-XS v2 (Original Legacy Model)</option>
              <option value="easyocr">🔍 EasyOCR (PyTorch GPU CRAFT)</option>
              <option value="mobile_vit">🌐 MobileViT-v2 Global</option>
            </select>
          </div>

          {/* AI Perception Status & Start/Stop Controls */}
          {aiStatus?.status === 'running' || aiStatus?.status === 'live' ? (
            <button
              className="btn btn-sm btn-outline"
              style={{
                padding: '4px 10px',
                fontSize: 11,
                fontWeight: 700,
                color: '#f87171',
                borderColor: '#ef4444',
                background: 'rgba(239, 68, 68, 0.1)',
                display: 'flex',
                alignItems: 'center',
                gap: 5,
              }}
              onClick={handleStopTrafficAI}
              title="Stop background AI inference"
            >
              <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#ef4444', display: 'inline-block' }} />
              Stop AI
            </button>
          ) : (
            <button
              className="btn btn-sm btn-success"
              style={{
                padding: '4px 10px',
                fontSize: 11,
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                gap: 5,
              }}
              onClick={handleStartTrafficAI}
              title="Start Traffic AI (strictly vehicles and stolen plate search)"
            >
              <Zap size={13} />
              Start Traffic AI
            </button>
          )}

          {/* Edit Monitoring Zone Toggle Button */}
          <button
            className={`btn btn-sm ${isEditingZone ? 'btn-primary' : 'btn-outline'}`}
            style={{
              padding: '4px 12px',
              fontSize: 12,
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: isEditingZone ? '#0284c7' : 'transparent',
              borderColor: isEditingZone ? '#38bdf8' : '#334155',
            }}
            onClick={() => setIsEditingZone(!isEditingZone)}
            title="Drag the 4 corner handles on the video feed to set the plate detection region"
          >
            <Sliders size={14} /> {isEditingZone ? 'Done Editing' : 'Set Detection Region'}
          </button>
        </div>
      </div>

      {/* Main Two-Panel Layout (Left Viewport 68% | Right ANPR Dashboard 32%) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 2.3fr) minmax(360px, 1.15fr)', gap: 16 }}>
        {/* ========================================================= */}
        {/* LEFT PANEL: Live Camera Feed with Virtual Monitoring Zone */}
        {/* ========================================================= */}
        <div
          className="card"
          style={{
            padding: 0,
            overflow: 'hidden',
            background: '#020617',
            border: '1px solid #1e293b',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {/* Viewport Action Header Bar */}
          <div
            style={{
              padding: '10px 14px',
              background: '#090d16',
              borderBottom: '1px solid #1e293b',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span
                style={{
                  background: '#0284c7',
                  color: 'white',
                  fontFamily: 'monospace',
                  fontWeight: 700,
                  fontSize: 11,
                  padding: '2px 8px',
                  borderRadius: 4,
                }}
              >
                {selectedCam.id}
              </span>
              <span style={{ fontSize: 13, fontWeight: 600 }}>{selectedCam.name}</span>
              <span style={{ fontSize: 11, color: '#64748b' }}>• {selectedCam.location || 'Highway Sector'}</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {isEditingZone && (
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <button
                    className="btn btn-outline btn-sm"
                    style={{ fontSize: 11, padding: '2px 8px', color: '#94a3b8' }}
                    onClick={handleResetZone}
                    title="Reset zone to default road trapezoid"
                  >
                    <RotateCcw size={12} /> Reset
                  </button>
                  <button
                    className="btn btn-primary btn-sm"
                    style={{ fontSize: 11, padding: '2px 10px', background: '#16a34a', borderColor: '#22c55e' }}
                    onClick={handleSaveZone}
                    disabled={isSavingZone}
                  >
                    <Check size={12} /> {isSavingZone ? 'Saving...' : 'Save Zone'}
                  </button>
                </div>
              )}

              {/* Feed Mode Switcher */}
              {hasVideo && (
                <div style={{ display: 'flex', background: '#0f172a', borderRadius: 4, padding: 2, border: '1px solid #334155' }}>
                  <button
                    style={{
                      background: feedMode === 'stream' ? '#2563eb' : 'transparent',
                      color: 'white',
                      border: 'none',
                      borderRadius: 3,
                      padding: '2px 8px',
                      fontSize: 11,
                      cursor: 'pointer',
                      fontWeight: 600,
                    }}
                    onClick={() => setFeedMode('stream')}
                  >
                    AI Stream
                  </button>
                  <button
                    style={{
                      background: feedMode === 'video' ? '#2563eb' : 'transparent',
                      color: 'white',
                      border: 'none',
                      borderRadius: 3,
                      padding: '2px 8px',
                      fontSize: 11,
                      cursor: 'pointer',
                      fontWeight: 600,
                    }}
                    onClick={() => setFeedMode('video')}
                  >
                    Direct MP4
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Video Stage with Monitoring Zone SVG / Canvas Overlay */}
          <div
            ref={videoViewportRef}
            style={{
              position: 'relative',
              width: '100%',
              minHeight: 480,
              background: '#000000',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              overflow: 'hidden',
              userSelect: 'none',
            }}
          >
            {hasVideo ? (
              feedMode === 'stream' ? (
                <img
                  key={selectedCam.id}
                  src={`/api/cameras/${selectedCam.id}/stream`}
                  alt={`Live Camera Feed ${selectedCam.id}`}
                  style={{
                    width: '100%',
                    height: 'auto',
                    maxHeight: 680,
                    display: 'block',
                    objectFit: 'contain',
                  }}
                />
              ) : (
                <video
                  src={selectedCam.video_url}
                  autoPlay
                  loop
                  muted
                  playsInline
                  controls
                  style={{
                    width: '100%',
                    height: 'auto',
                    maxHeight: 680,
                    display: 'block',
                    objectFit: 'contain',
                  }}
                />
              )
            ) : (
              <div style={{ textAlign: 'center', padding: 40, color: '#64748b' }}>
                <Camera size={48} style={{ margin: '0 auto 12px', opacity: 0.5 }} />
                <div style={{ fontSize: 14, fontWeight: 600, color: '#94a3b8' }}>
                  No video feed attached to {selectedCam.id}
                </div>
                <div style={{ fontSize: 12, marginTop: 4, marginBottom: 16 }}>
                  Upload a traffic video file or attach an existing sample to begin live ANPR monitoring.
                </div>
                <button
                  className="btn btn-primary"
                  style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, background: '#0284c7', borderColor: '#0284c7' }}
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                >
                  {isUploading ? <Loader2 size={14} className="spin" /> : <UploadCloud size={14} />}
                  {isUploading ? 'Uploading Video...' : 'Upload Video to Traffic Cam'}
                </button>
              </div>
            )}

            {/* In-Video OSD Metadata Overlay (matching screenshot: timestamp top-left, camera tag bottom-right) */}
            <div
              style={{
                position: 'absolute',
                top: 14,
                left: 16,
                fontFamily: 'monospace',
                fontSize: 13,
                fontWeight: 700,
                color: '#f8fafc',
                textShadow: '0 2px 4px rgba(0,0,0,0.9)',
                letterSpacing: '0.04em',
                pointerEvents: 'none',
              }}
            >
              10-13-2025 Mon 15:03:48
            </div>

            <div
              style={{
                position: 'absolute',
                bottom: 14,
                right: 16,
                fontFamily: 'monospace',
                fontSize: 12,
                fontWeight: 600,
                color: '#cbd5e1',
                textShadow: '0 2px 4px rgba(0,0,0,0.9)',
                pointerEvents: 'none',
              }}
            >
              (S297) {selectedCam.name} • {selectedCam.location}
            </div>

            {/* Interactive SVG Zone Editor Overlay */}
            {isEditingZone && (
              <svg
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'auto',
                }}
              >
                {/* Drawn Quadrilateral Zone Polygon */}
                <polygon
                  points={svgPoints}
                  fill="rgba(56, 189, 248, 0.25)"
                  stroke="#38bdf8"
                  strokeWidth="2.5"
                  strokeDasharray="6 3"
                />

                {/* 4 Interactive Corner Drag Handles */}
                {zonePoints.map((pt, idx) => {
                  const cx = `${pt[0] * 100}%`;
                  const cy = `${pt[1] * 100}%`;
                  const isHovered = activeHandle === idx;
                  return (
                    <g key={idx} onPointerDown={(e) => handlePointerDown(idx, e)} style={{ cursor: 'grab' }}>
                      <circle
                        cx={cx}
                        cy={cy}
                        r={isHovered ? 12 : 9}
                        fill="#0284c7"
                        stroke="#ffffff"
                        strokeWidth="2.5"
                      />
                      <text
                        x={cx}
                        y={cy}
                        dy="3.5"
                        textAnchor="middle"
                        fill="white"
                        fontSize="9"
                        fontWeight="bold"
                        pointerEvents="none"
                      >
                        {idx + 1}
                      </text>
                    </g>
                  );
                })}
              </svg>
            )}

            {/* Edit Mode Instruction Banner */}
            {isEditingZone && (
              <div
                style={{
                  position: 'absolute',
                  top: 14,
                  right: 16,
                  background: 'rgba(15, 23, 42, 0.92)',
                  border: '1px solid #38bdf8',
                  padding: '6px 14px',
                  borderRadius: 6,
                  fontSize: 11,
                  fontWeight: 700,
                  color: '#38bdf8',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <span>🎯 Drag handles 1-4 to position the Monitoring Zone</span>
              </div>
            )}
          </div>
        </div>

        {/* ========================================================= */}
        {/* RIGHT PANEL: ANPR DASHBOARD (Strictly matching screenshot) */}
        {/* ========================================================= */}
        <div
          className="anpr-dashboard-sidebar"
          style={{
            background: '#090e17',
            border: '1px solid #1e293b',
            borderRadius: 8,
            padding: 16,
            display: 'flex',
            flexDirection: 'column',
            gap: 14,
            boxShadow: '0 8px 30px rgba(0,0,0,0.5)',
          }}
        >
          {/* Header Title + LIVE Badge */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <h3
                style={{
                  margin: 0,
                  fontSize: 16,
                  fontWeight: 800,
                  letterSpacing: '0.04em',
                  color: '#f8fafc',
                  textTransform: 'uppercase',
                }}
              >
                ANPR DASHBOARD
              </h3>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
                Zone based vehicle plate recognition
              </div>
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                fontSize: 10,
                fontWeight: 800,
                color: '#ef4444',
                background: 'rgba(239, 68, 68, 0.15)',
                padding: '2px 8px',
                borderRadius: 10,
                border: '1px solid rgba(239, 68, 68, 0.4)',
                letterSpacing: '0.05em',
              }}
            >
              <span
                style={{
                  width: 7,
                  height: 7,
                  borderRadius: '50%',
                  background: '#ef4444',
                  boxShadow: '0 0 8px #ef4444',
                }}
              ></span>
              LIVE
            </div>
          </div>

          {/* Top 3 KPI Metric Tiles (PLATES | NO READ | VEHICLES) */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr 1fr',
              gap: 8,
              textAlign: 'center',
            }}
          >
            {/* Plates Tile */}
            <div
              style={{
                background: '#0e1626',
                border: '1px solid #1e293b',
                borderRadius: 6,
                padding: '8px 4px',
              }}
            >
              <div style={{ fontSize: 22, fontWeight: 800, color: '#22c55e', lineHeight: 1 }}>
                {kpiPlates}
              </div>
              <div style={{ fontSize: 9, fontWeight: 700, color: '#64748b', marginTop: 4, letterSpacing: '0.05em' }}>
                PLATES
              </div>
            </div>

            {/* No Read Tile */}
            <div
              style={{
                background: '#0e1626',
                border: '1px solid #1e293b',
                borderRadius: 6,
                padding: '8px 4px',
              }}
            >
              <div style={{ fontSize: 22, fontWeight: 800, color: '#ef4444', lineHeight: 1 }}>
                {kpiNoRead}
              </div>
              <div style={{ fontSize: 9, fontWeight: 700, color: '#64748b', marginTop: 4, letterSpacing: '0.05em' }}>
                NO READ
              </div>
            </div>

            {/* Vehicles in Zone Tile */}
            <div
              style={{
                background: '#0e1626',
                border: '1px solid #1e293b',
                borderRadius: 6,
                padding: '8px 4px',
              }}
            >
              <div style={{ fontSize: 22, fontWeight: 800, color: '#38bdf8', lineHeight: 1 }}>
                {kpiVehicles}
              </div>
              <div style={{ fontSize: 9, fontWeight: 700, color: '#64748b', marginTop: 4, letterSpacing: '0.05em' }}>
                VEHICLES
              </div>
            </div>
          </div>

          {/* PLATE ZOOM — VEHICLE Inspection Window */}
          <div
            style={{
              background: '#0c1322',
              border: '1px solid #1e2d4a',
              borderRadius: 6,
              padding: '10px 12px',
            }}
          >
            {/* Header with state pill */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 8,
              }}
            >
              <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.05em', color: '#94a3b8' }}>
                PLATE ZOOM — {currentPlate.vehicleType}
              </div>

              <span
                style={{
                  fontSize: 10,
                  fontWeight: 800,
                  padding: '2px 8px',
                  borderRadius: 4,
                  letterSpacing: '0.05em',
                  background:
                    currentPlate.status === 'HOTLIST'
                      ? '#ef4444'
                      : currentPlate.status === 'CONFIRMED'
                      ? '#16a34a'
                      : '#f59e0b',
                  color: currentPlate.status === 'READING' ? '#0f172a' : '#ffffff',
                }}
              >
                {currentPlate.status}
              </span>
            </div>

            {/* Cropped Plate Image Preview Frame */}
            <div
              style={{
                position: 'relative',
                background: '#020617',
                border: `2px solid ${currentPlate.isHotlist ? '#ef4444' : '#1e3a5f'}`,
                borderRadius: 4,
                height: 120,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
              }}
            >
              {currentPlate.crop ? (
                <img
                  src={currentPlate.crop}
                  alt={`Plate Crop ${currentPlate.plate}`}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    filter: 'contrast(1.15) brightness(1.05)',
                  }}
                />
              ) : (
                <div
                  style={{
                    width: '85%',
                    height: '70%',
                    background: '#f8fafc',
                    color: '#0f172a',
                    fontFamily: 'monospace',
                    fontWeight: 900,
                    fontSize: 26,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderRadius: 4,
                    border: '3px solid #0f172a',
                    boxShadow: 'inset 0 0 10px rgba(0,0,0,0.2)',
                    letterSpacing: '0.1em',
                  }}
                >
                  {currentPlate.plate || 'BDR-775'}
                </div>
              )}

              {/* Cyan overlay bounding box + confidence readout */}
              <div
                style={{
                  position: 'absolute',
                  bottom: 4,
                  left: 6,
                  fontSize: 10,
                  fontFamily: 'monospace',
                  fontWeight: 700,
                  color: '#38bdf8',
                  background: 'rgba(2, 6, 23, 0.85)',
                  padding: '1px 5px',
                  borderRadius: 2,
                }}
              >
                {currentPlate.plate} &nbsp;{currentPlate.confidence}%
              </div>
            </div>
          </div>

          {/* RECOGNIZED NUMBER Banner */}
          <div>
            <div style={{ fontSize: 10, fontWeight: 700, color: '#64748b', letterSpacing: '0.06em', marginBottom: 4 }}>
              RECOGNIZED NUMBER
            </div>

            {/* High-contrast plate card */}
            <div
              style={{
                background: currentPlate.isHotlist ? '#fee2e2' : '#ffffff',
                border: currentPlate.isHotlist ? '2px solid #ef4444' : '2px solid #cbd5e1',
                borderRadius: 4,
                padding: '8px 12px',
                textAlign: 'center',
                boxShadow: '0 4px 12px rgba(0,0,0,0.25)',
              }}
            >
              <div
                style={{
                  fontSize: 28,
                  fontWeight: 900,
                  fontFamily: 'monospace',
                  letterSpacing: '0.12em',
                  color: currentPlate.isHotlist ? '#b91c1c' : '#0f172a',
                }}
              >
                {currentPlate.plate} {currentPlate.confidence < 80 ? '?' : ''}
              </div>
            </div>

            {/* Confidence Bar & Reads */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                marginTop: 6,
                fontSize: 11,
                fontFamily: 'monospace',
                color: '#94a3b8',
              }}
            >
              <span>CONF</span>
              <div
                style={{
                  flex: 1,
                  height: 8,
                  background: '#1e293b',
                  borderRadius: 4,
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    width: `${currentPlate.confidence}%`,
                    height: '100%',
                    background: currentPlate.confidence > 80 ? '#22c55e' : '#f59e0b',
                    borderRadius: 4,
                  }}
                ></div>
              </div>
              <span style={{ fontWeight: 700, color: '#f8fafc' }}>{currentPlate.confidence}%</span>
              <span style={{ color: '#64748b' }}>reads: {currentPlate.reads}</span>
            </div>

            {/* Quick Watchlist Flag / Status Action */}
            <div style={{ marginTop: 8, display: 'flex', justifyContent: 'center' }}>
              {currentPlate.isHotlist ? (
                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '4px 10px',
                    borderRadius: 4,
                    background: 'rgba(239, 68, 68, 0.2)',
                    border: '1px solid #ef4444',
                    color: '#f87171',
                    fontSize: 11,
                    fontWeight: 700,
                  }}
                >
                  <ShieldAlert size={14} />
                  WATCHLIST ACTIVE: {currentPlate.reason || 'Flagged Target'}
                </div>
              ) : (
                <button
                  className="btn btn-sm btn-outline"
                  style={{
                    fontSize: 11,
                    padding: '3px 10px',
                    borderColor: '#f43f5e',
                    color: '#fb7185',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 5,
                    background: 'rgba(244, 63, 94, 0.08)',
                    cursor: 'pointer',
                  }}
                  onClick={() => handleAddToWatchlist(currentPlate.plate, 'Flagged from Traffic Feed')}
                  title="Mark this vehicle into the Watchlist / Stolen registry"
                >
                  <ShieldAlert size={13} />
                  + Flag in Watchlist
                </button>
              )}
            </div>
          </div>

          {/* RECENT PLATES Feed */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 220 }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: 10,
                fontWeight: 700,
                color: '#64748b',
                letterSpacing: '0.06em',
                marginBottom: 8,
              }}
            >
              <span>RECENT PLATES</span>
              <span>{recentPlates.length} logged</span>
            </div>

            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 8,
                maxHeight: 280,
                overflowY: 'auto',
                paddingRight: 4,
              }}
            >
              {recentPlates.map(item => (
                <div
                  key={item.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    padding: '8px 10px',
                    borderRadius: 6,
                    background: item.isHotlist ? 'rgba(239, 68, 68, 0.15)' : '#0e1626',
                    border: item.isHotlist ? '1px solid #ef4444' : '1px solid #1e293b',
                    transition: 'all 0.2s',
                  }}
                >
                  {/* Plate Thumbnail Crop */}
                  <div
                    style={{
                      width: 64,
                      height: 34,
                      borderRadius: 3,
                      background: '#020617',
                      border: '1px solid #334155',
                      overflow: 'hidden',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}
                  >
                    {item.crop ? (
                      <img
                        src={item.crop}
                        alt={item.plate}
                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                      />
                    ) : (
                      <div
                        style={{
                          width: '92%',
                          height: '80%',
                          background: item.isHotlist ? '#fee2e2' : '#f8fafc',
                          color: item.isHotlist ? '#b91c1c' : '#0f172a',
                          fontSize: 9,
                          fontFamily: 'monospace',
                          fontWeight: 800,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          borderRadius: 2,
                        }}
                      >
                        {item.plate}
                      </div>
                    )}
                  </div>

                  {/* Plate Info */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span
                        style={{
                          fontFamily: 'monospace',
                          fontSize: 14,
                          fontWeight: 800,
                          color: item.isHotlist ? '#f87171' : '#f8fafc',
                          letterSpacing: '0.04em',
                        }}
                      >
                        {item.plate}
                      </span>
                      {item.isHotlist && (
                        <span
                          style={{
                            background: '#ef4444',
                            color: 'white',
                            fontSize: 9,
                            fontWeight: 800,
                            padding: '1px 5px',
                            borderRadius: 4,
                          }}
                        >
                          STOLEN
                        </span>
                      )}
                    </div>

                    <div style={{ fontSize: 10, color: '#64748b', marginTop: 2, fontFamily: 'monospace' }}>
                      {item.vehicleType} &bull; {item.time} &bull; {item.confidence}%
                      {item.reads && item.reads > 1 && (
                        <span style={{ marginLeft: 6, color: '#38bdf8', fontWeight: 600 }}>
                          ({item.reads} scans)
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Watchlist Action */}
                  <div style={{ flexShrink: 0 }}>
                    {item.isHotlist ? (
                      <span
                        style={{
                          background: 'rgba(239, 68, 68, 0.2)',
                          color: '#f87171',
                          border: '1px solid #ef4444',
                          fontSize: 9,
                          fontWeight: 800,
                          padding: '3px 6px',
                          borderRadius: 4,
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 3,
                        }}
                      >
                        <ShieldAlert size={11} /> HOTLIST
                      </span>
                    ) : (
                      <button
                        style={{
                          background: 'transparent',
                          border: '1px solid #334155',
                          color: '#94a3b8',
                          fontSize: 9,
                          fontWeight: 700,
                          padding: '3px 7px',
                          borderRadius: 4,
                          cursor: 'pointer',
                        }}
                        onClick={() => handleAddToWatchlist(item.plate, 'Flagged from Recent Feed')}
                        title="Add this plate to Watchlist"
                      >
                        + Flag
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
