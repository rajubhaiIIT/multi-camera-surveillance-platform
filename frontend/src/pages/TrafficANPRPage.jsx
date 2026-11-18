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
  getUser,
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

  // Zone configuration state
  const [isEditingZone, setIsEditingZone] = useState(false);
  const [zonePoints, setZonePoints] = useState(DEFAULT_ZONE);
  const [isSavingZone, setIsSavingZone] = useState(false);
  const [activeHandle, setActiveHandle] = useState(null);

  // ANPR Live Dashboard State
  const [kpiPlates, setKpiPlates] = useState(27);
  const [kpiNoRead, setKpiNoRead] = useState(2);
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
  const videoViewportRef = useRef(null);
  const user = getUser();

  // Load cameras and available sample videos
  const loadData = useCallback(async () => {
    try {
      const cams = await getCameras();
      if (Array.isArray(cams) && cams.length > 0) {
        setCameras(cams);
        if (!selectedCamId) {
          // Prefer camera with active video or cam1
          const withVid = cams.find(c => c.video_url || c.source_type === 'video_file') || cams[0];
          setSelectedCamId(withVid.id);
        }
      }
    } catch (_) {}

    try {
      const vids = await getAvailableVideos();
      if (Array.isArray(vids)) setAvailableVideos(vids);
    } catch (_) {}
  }, [selectedCamId]);

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

  // Handle incoming real-time WebSocket events
  const handleWsMessage = useCallback((msg) => {
    if (msg.type === 'anpr_detection') {
      const { plate, confidence, country_format, plate_crop, is_hotlist, reason, vehicle_type, vehicles_in_zone } = msg;

      if (typeof vehicles_in_zone === 'number') {
        setKpiVehicles(vehicles_in_zone);
      }

      const confNum = Math.round(confidence || 75);
      const isHot = !!is_hotlist;

      // Update current Plate Zoom widget
      setCurrentPlate(prev => ({
        plate: plate || 'UNKNOWN',
        confidence: confNum,
        status: isHot ? 'HOTLIST' : confNum > 80 ? 'CONFIRMED' : 'READING',
        reads: (prev.plate === plate ? prev.reads + 1 : 1),
        crop: plate_crop || prev.crop,
        vehicleType: (vehicle_type || 'VEHICLE').toUpperCase(),
        isHotlist: isHot,
        reason: reason || '',
      }));

      // Increment counters
      setKpiPlates(c => c + 1);

      // Prepend to Recent Plates list
      const now = new Date();
      const timeStr = now.toTimeString().split(' ')[0];
      const newRec = {
        id: `anpr-${Date.now()}-${Math.random()}`,
        plate: plate || 'UNKNOWN',
        vehicleType: (vehicle_type || 'VEHICLE').toUpperCase(),
        time: timeStr,
        confidence: confNum,
        crop: plate_crop || null,
        isHotlist: isHot,
        reason: reason || '',
      };

      setRecentPlates(prev => [newRec, ...prev.slice(0, 19)]);

      if (isHot) {
        showToast(`🚨 HOTLIST DETECTED: [${plate}] matches stolen vehicle registry!`, 'critical');
      }
    } else if (msg.type === 'alert' && msg.payload?.alert_type === 'watchlist_plate') {
      const p = msg.payload.metadata?.plate || 'STOLEN VEHICLE';
      showToast(`⚠ Stolen Vehicle Plate Alert: ${p}`, 'critical');
    }
  }, []);

  useWebSocket(handleWsMessage);

  const showToast = (msg, type = 'info') => {
    setNotification({ msg, type, id: Date.now() });
    setTimeout(() => setNotification(null), 6000);
  };

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

  // Quick attach video helper
  const handleAttachVideo = async (filename) => {
    if (!selectedCamId || !filename) return;
    try {
      await attachVideoToCamera(selectedCamId, filename);
      loadData();
      showToast(`Attached [${filename}] to ${selectedCamId}. Stream warming up.`, 'success');
    } catch (err) {
      alert(`Attach error: ${err.message}`);
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

        {/* Camera Selector & Quick Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#94a3b8' }}>
            <Camera size={14} /> Camera Feed:
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
                <div style={{ fontSize: 12, marginTop: 4 }}>
                  Attach a sample video from the top bar dropdown to begin live ANPR monitoring.
                </div>
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
                    </div>
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
