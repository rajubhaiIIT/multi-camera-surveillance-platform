import { useEffect, useState, useRef, useCallback } from 'react';
import {
  getCameras,
  createCamera,
  deleteCamera,
  uploadCameraVideo,
  getAvailableVideos,
  attachVideoToCamera,
  triggerAIAnalysis,
  getAIAnalysisStatus,
  stopAIAnalysis,
  getUser,
} from '../api.js';
import {
  Camera,
  Plus,
  Trash2,
  Video,
  MapPin,
  UploadCloud,
  Film,
  Zap,
  Square,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  RotateCw,
} from 'lucide-react';

export default function CamerasPage() {
  const [cameras, setCameras] = useState([]);
  const [availableVideos, setAvailableVideos] = useState([]);
  const [showAddRTSP, setShowAddRTSP] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [autoAnalyzeUpload, setAutoAnalyzeUpload] = useState(true);
  const [aiOverlayEnabled, setAiOverlayEnabled] = useState({});

  // Analysis status dictionary by camera ID
  const [analysisStatuses, setAnalysisStatuses] = useState({});
  const [triggeringCam, setTriggeringCam] = useState({});

  // Forms
  const [rtspForm, setRtspForm] = useState({ id: '', name: '', rtsp_url: '', location: 'Main Campus', fps: 30 });
  const [uploadForm, setUploadForm] = useState({ id: '', name: '', location: 'Indoor / Facility', fps: 30 });
  const [selectedFile, setSelectedFile] = useState(null);

  const fileInputRef = useRef(null);
  const user = getUser();
  const canEdit = user?.role === 'admin' || user?.role === 'operator';

  const loadCameras = useCallback(() => {
    getCameras().then(setCameras).catch(() => {});
    getAvailableVideos().then(setAvailableVideos).catch(() => {});
  }, []);

  useEffect(() => {
    loadCameras();
  }, [loadCameras]);

  // Robust status polling loop for active AI inference jobs
  useEffect(() => {
    let isCancelled = false;

    const poll = async () => {
      if (cameras.length === 0) return;
      for (const cam of cameras) {
        if (cam.video_url || cam.source_type === 'video_file') {
          try {
            const st = await getAIAnalysisStatus(cam.id);
            if (!isCancelled && st && st.status !== 'idle') {
              setAnalysisStatuses(prev => {
                const prevSt = prev[cam.id];
                // Only trigger React state change if values actually changed
                if (
                  prevSt &&
                  prevSt.status === st.status &&
                  prevSt.current_frame === st.current_frame &&
                  prevSt.alerts_created === st.alerts_created
                ) {
                  return prev;
                }
                return { ...prev, [cam.id]: st };
              });
            }
          } catch (_) {}
        }
      }
    };

    poll();
    const interval = setInterval(poll, 1000);
    return () => {
      isCancelled = true;
      clearInterval(interval);
    };
  }, [cameras]);

  // Trigger AI Perception Worker on a Camera
  async function handleTriggerAI(camId) {
    try {
      setTriggeringCam(prev => ({ ...prev, [camId]: true }));
      setAnalysisStatuses(prev => ({
        ...prev,
        [camId]: {
          status: 'initializing',
          status_msg: 'Warming up YOLO neural networks & video frames...',
          progress_pct: 0.0,
          current_frame: 0,
          total_frames: 0,
        },
      }));
      const job = await triggerAIAnalysis(camId, null, 'general');
      if (job) {
        setAnalysisStatuses(prev => ({ ...prev, [camId]: job }));
      }
    } catch (err) {
      alert(`Failed to start AI analysis: ${err.message}`);
      setAnalysisStatuses(prev => ({
        ...prev,
        [camId]: { status: 'error', error: err.message },
      }));
    } finally {
      setTriggeringCam(prev => ({ ...prev, [camId]: false }));
    }
  }

  // Stop running AI Perception Worker
  async function handleStopAI(camId) {
    try {
      await stopAIAnalysis(camId);
      setAnalysisStatuses(prev => ({
        ...prev,
        [camId]: { ...prev[camId], status: 'stopped' },
      }));
    } catch (err) {
      alert(`Failed to stop AI: ${err.message}`);
    }
  }

  // Add Camera via RTSP
  async function handleAddRTSP(e) {
    e.preventDefault();
    try {
      await createCamera({ ...rtspForm, is_active: true, source_type: 'rtsp' });
      setShowAddRTSP(false);
      setRtspForm({ id: '', name: '', rtsp_url: '', location: 'Main Campus', fps: 30 });
      loadCameras();
    } catch (err) { alert(err.message); }
  }

  // Upload Video File from browser
  async function handleUploadVideo(e) {
    e.preventDefault();
    if (!selectedFile) {
      alert('Please choose a video file (.mp4, .mov, .webm, .avi)');
      return;
    }
    const camId = uploadForm.id.trim() || `cam_${Date.now().toString().slice(-4)}`;
    const camName = uploadForm.name.trim() || `Video Feed (${selectedFile.name})`;

    const fd = new FormData();
    fd.append('file', selectedFile);
    fd.append('camera_id', camId);
    fd.append('name', camName);
    fd.append('location', uploadForm.location);
    fd.append('fps', uploadForm.fps);
    fd.append('auto_analyze', autoAnalyzeUpload ? 'true' : 'false');

    setIsUploading(true);
    try {
      await uploadCameraVideo(fd);
      setShowUploadModal(false);
      setSelectedFile(null);
      setUploadForm({ id: '', name: '', location: 'Indoor / Facility', fps: 30 });
      if (fileInputRef.current) fileInputRef.current.value = '';
      loadCameras();
      if (autoAnalyzeUpload) {
        setAnalysisStatuses(prev => ({
          ...prev,
          [camId]: {
            status: 'initializing',
            status_msg: 'Warming up YOLO & processing uploaded footage...',
            progress_pct: 0.0,
            current_frame: 0,
          },
        }));
      }
    } catch (err) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  }

  // Attach existing server video to camera
  async function handleAttachExisting(camId, filename) {
    if (!filename) return;
    try {
      await attachVideoToCamera(camId, filename);
      loadCameras();
    } catch (err) {
      alert(err.message);
    }
  }

  async function handleDelete(id) {
    if (!confirm(`Delete camera ${id}?`)) return;
    try { await deleteCamera(id); loadCameras(); } catch (err) { alert(err.message); }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Camera Management & Live Feeds</h2>
          <div className="subtitle">{cameras.length} camera streams registered &bull; In-browser Video & Background AI Perception</div>
        </div>
        {canEdit && (
          <div style={{ display: 'flex', gap: 10 }}>
            <button className="btn btn-primary" onClick={() => { setShowUploadModal(!showUploadModal); setShowAddRTSP(false); }}>
              <UploadCloud size={16} /> Upload Video Sample
            </button>
            <button className="btn btn-outline" onClick={() => { setShowAddRTSP(!showAddRTSP); setShowUploadModal(false); }}>
              <Plus size={16} /> Add RTSP Stream
            </button>
          </div>
        )}
      </div>

      {/* Upload Video Sample Modal / Card */}
      {showUploadModal && (
        <div className="card" style={{ marginBottom: 24, border: '1px solid #3b82f6' }}>
          <div className="card-header">
            <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <UploadCloud size={20} color="#3b82f6" /> Upload Video Sample as Camera Feed
            </h3>
            <button className="btn btn-outline btn-sm" onClick={() => setShowUploadModal(false)}>Close</button>
          </div>
          <form onSubmit={handleUploadVideo}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14, marginBottom: 14 }}>
              <div>
                <label style={{ display: 'block', fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>Camera ID</label>
                <input
                  placeholder="e.g. cam_lobby"
                  value={uploadForm.id}
                  onChange={e => setUploadForm({ ...uploadForm, id: e.target.value })}
                  required
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>Display Name</label>
                <input
                  placeholder="e.g. Main Lobby Camera"
                  value={uploadForm.name}
                  onChange={e => setUploadForm({ ...uploadForm, name: e.target.value })}
                  required
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>Location</label>
                <input
                  placeholder="e.g. Building A - Level 1"
                  value={uploadForm.location}
                  onChange={e => setUploadForm({ ...uploadForm, location: e.target.value })}
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>Target FPS</label>
                <input
                  type="number"
                  value={uploadForm.fps}
                  onChange={e => setUploadForm({ ...uploadForm, fps: +e.target.value })}
                  style={{ width: '100%' }}
                />
              </div>
            </div>

            <div
              style={{
                border: '2px dashed #334155',
                borderRadius: 8,
                padding: 24,
                textAlign: 'center',
                background: '#0f172a',
                marginBottom: 14,
                cursor: 'pointer',
              }}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4,video/webm,video/quicktime,video/x-msvideo"
                style={{ display: 'none' }}
                onChange={e => setSelectedFile(e.target.files?.[0] || null)}
              />
              <Film size={36} color="#60a5fa" style={{ margin: '0 auto 8px', display: 'block' }} />
              <div style={{ fontWeight: 600, fontSize: 14, color: '#f8fafc' }}>
                {selectedFile ? `Selected: ${selectedFile.name} (${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)` : 'Click or drop a video file here'}
              </div>
              <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
                Supports MP4, WebM, MOV, and AVI surveillance footage
              </div>
            </div>

            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', userSelect: 'none' }}>
                <input
                  type="checkbox"
                  checked={autoAnalyzeUpload}
                  onChange={e => setAutoAnalyzeUpload(e.target.checked)}
                  style={{ width: 16, height: 16, cursor: 'pointer', accentColor: '#3b82f6' }}
                />
                <span style={{ fontSize: 13, color: '#38bdf8', fontWeight: 600 }}>
                  ⚡ Automatically run real-time AI Threat, Fall & ANPR Detection after upload
                </span>
              </label>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button type="submit" className="btn btn-success" disabled={isUploading || !selectedFile}>
                {isUploading ? 'Uploading & Starting AI Feed...' : 'Upload & Start Feed'}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Add Camera via RTSP Form */}
      {showAddRTSP && (
        <div className="card" style={{ marginBottom: 20 }}>
          <h3 style={{ marginBottom: 12 }}>Register RTSP IP Camera</h3>
          <form onSubmit={handleAddRTSP} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr auto', gap: 12 }}>
            <input placeholder="Camera ID (e.g. cam3)" value={rtspForm.id} onChange={e => setRtspForm({ ...rtspForm, id: e.target.value })} required />
            <input placeholder="Display Name" value={rtspForm.name} onChange={e => setRtspForm({ ...rtspForm, name: e.target.value })} required />
            <input placeholder="RTSP URL (rtsp://...)" value={rtspForm.rtsp_url} onChange={e => setRtspForm({ ...rtspForm, rtsp_url: e.target.value })} required />
            <input placeholder="Location" value={rtspForm.location} onChange={e => setRtspForm({ ...rtspForm, location: e.target.value })} />
            <button type="submit" className="btn btn-success">Create</button>
          </form>
        </div>
      )}

      {/* Camera Grid with Live In-Browser Video Feeds */}
      <div className="camera-grid">
        {cameras.map(cam => {
          const hasVideo = !!cam.video_url;
          const job = analysisStatuses[cam.id];
          const isOverlay = aiOverlayEnabled[cam.id] !== false; // Default ON to show boxes and person IDs

          return (
            <div key={cam.id} className="camera-card">
              <div className="camera-feed" style={{ position: 'relative', overflow: 'hidden', height: 260, background: '#020617' }}>
                {hasVideo ? (
                  isOverlay ? (
                    <img
                      src={`/api/cameras/${cam.id}/stream`}
                      alt={`${cam.name} AI Detection Overlay Feed`}
                      style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block', background: '#020617' }}
                    />
                  ) : (
                    <video
                      src={cam.video_url}
                      autoPlay
                      loop
                      muted
                      playsInline
                      controls
                      style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    />
                  )
                ) : (
                  <div style={{ textAlign: 'center', padding: 20 }}>
                    <Camera size={44} color="#334155" />
                    <div style={{ marginTop: 8, fontSize: 13, color: '#94a3b8', fontWeight: 600 }}>
                      {cam.name} ({cam.id})
                    </div>
                    <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                      RTSP Stream: {cam.rtsp_url}
                    </div>

                    {/* Quick attach video helper */}
                    {availableVideos.length > 0 && canEdit && (
                      <div style={{ marginTop: 14 }}>
                        <select
                          style={{ fontSize: 12, padding: '4px 8px' }}
                          onChange={e => handleAttachExisting(cam.id, e.target.value)}
                          defaultValue=""
                        >
                          <option value="" disabled>🎬 Attach Video Sample...</option>
                          {availableVideos.map(v => (
                            <option key={v.filename} value={v.filename}>
                              {v.filename} ({v.size_mb} MB)
                            </option>
                          ))}
                        </select>
                      </div>
                    )}
                  </div>
                )}

                <div className="cam-overlay" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', padding: '8px 10px', boxSizing: 'border-box' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div className="live-dot"></div>
                    <span className={`badge badge-${cam.status === 'online' ? 'online' : 'offline'}`}>
                      {hasVideo ? (isOverlay ? 'AI PERCEPTION (BOXES & IDS)' : 'RAW VIDEO') : (cam.status === 'online' ? 'RTSP LIVE' : 'OFFLINE')}
                    </span>
                  </div>
                  {hasVideo && (
                    <button
                      type="button"
                      style={{
                        padding: '4px 10px',
                        fontSize: 11,
                        fontWeight: 700,
                        letterSpacing: '0.02em',
                        background: isOverlay ? '#16a34a' : '#334155',
                        color: '#ffffff',
                        border: '1px solid rgba(255,255,255,0.3)',
                        borderRadius: 4,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 5,
                        boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                      }}
                      onClick={() => setAiOverlayEnabled(prev => ({ ...prev, [cam.id]: !isOverlay }))}
                      title="Toggle AI bounding boxes and person ID overlays"
                    >
                      {isOverlay ? '🟩 AI Boxes: ON' : '⬜ AI Boxes: OFF'}
                    </button>
                  )}
                </div>
              </div>

              <div className="camera-info">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <h4>{cam.name}</h4>
                    <div className="cam-meta">
                      <span><MapPin size={11} /> {cam.location}</span>
                      <span><Video size={11} /> {cam.fps} FPS ({cam.source_type || 'rtsp'})</span>
                    </div>
                  </div>
                  {canEdit && user?.role === 'admin' && (
                    <button className="btn btn-danger btn-sm" onClick={() => handleDelete(cam.id)}>
                      <Trash2 size={12} />
                    </button>
                  )}
                </div>

                {/* AI Threat & Anomaly Perception Control Box */}
                {hasVideo && (
                  <div style={{
                    marginTop: 12,
                    padding: '10px 12px',
                    background: '#090d16',
                    borderRadius: 6,
                    border: '1px solid #1e293b',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 5 }}>
                        <Zap size={13} color="#eab308" /> AI Threat Perception
                      </span>
                      {job?.status === 'live' && (
                        <span className="badge badge-success" style={{ fontSize: 10, display: 'flex', alignItems: 'center', gap: 4, background: 'rgba(34,197,94,0.2)', color: '#4ade80' }}>
                          <span className="live-dot" style={{ width: 6, height: 6 }}></span> LIVE MONITORING
                        </span>
                      )}
                      {job?.status === 'initializing' && (
                        <span className="badge badge-warning" style={{ fontSize: 10, display: 'flex', alignItems: 'center', gap: 4 }}>
                          <Loader2 size={10} className="spin" /> WARMING UP
                        </span>
                      )}
                      {job?.status === 'running' && (
                        <span className="badge badge-warning" style={{ fontSize: 10, display: 'flex', alignItems: 'center', gap: 4, background: 'rgba(59,130,246,0.2)', color: '#60a5fa' }}>
                          <Loader2 size={10} className="spin" /> PROCESSING ({job.progress_pct}%)
                        </span>
                      )}
                      {job?.status === 'completed' && (
                        <span className="badge badge-success" style={{ fontSize: 10, display: 'flex', alignItems: 'center', gap: 4 }}>
                          <CheckCircle2 size={10} /> COMPLETED
                        </span>
                      )}
                      {job?.status === 'stopped' && (
                        <span className="badge badge-offline" style={{ fontSize: 10 }}>
                          PAUSED
                        </span>
                      )}
                      {job?.status === 'error' && (
                        <span className="badge badge-danger" style={{ fontSize: 10, display: 'flex', alignItems: 'center', gap: 4 }}>
                          <AlertTriangle size={10} /> ERROR
                        </span>
                      )}
                    </div>

                    {/* Initializing message */}
                    {job?.status === 'initializing' && (
                      <div style={{ fontSize: 11, color: '#f59e0b', marginBottom: 6, fontStyle: 'italic' }}>
                        {job.status_msg || 'Loading neural weights & preparing pipeline...'}
                      </div>
                    )}

                    {/* Progress Bar & Real-time Metrics if live, running, or initializing */}
                    {(job?.status === 'live' || job?.status === 'running' || job?.status === 'initializing') && (
                      <div style={{ marginBottom: 8 }}>
                        <div style={{ height: 6, width: '100%', background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
                          <div style={{
                            height: '100%',
                            width: `${Math.max(job.progress_pct || 0, job.status === 'initializing' ? 10 : 0)}%`,
                            background: 'linear-gradient(90deg, #10b981, #06b6d4, #3b82f6)',
                            transition: 'width 0.3s ease',
                          }} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#64748b', marginTop: 4 }}>
                          <span>Tracks: <b style={{ color: '#38bdf8' }}>{job.active_tracks || 0} active</b></span>
                          <span>{job.fps || 0} FPS</span>
                          <span style={{ color: '#ef4444', fontWeight: 600 }}>🚨 {job.alerts_created || 0} alerts</span>
                        </div>
                      </div>
                    )}

                    {/* Summary if paused/stopped */}
                    {job?.status === 'stopped' && (
                      <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8 }}>
                        Paused at frame {job.current_frame} &bull; {job.alerts_created || 0} alerts detected
                      </div>
                    )}

                    {/* Error display */}
                    {job?.status === 'error' && (
                      <div style={{ fontSize: 11, color: '#ef4444', marginBottom: 8, wordBreak: 'break-word' }}>
                        {job.error || 'Inference error'}
                      </div>
                    )}

                    {/* Action buttons (Pause / Resume Overrides) */}
                    {canEdit && (
                      <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                        {job?.status === 'live' || job?.status === 'running' || job?.status === 'initializing' ? (
                          <button
                            className="btn btn-outline btn-sm"
                            style={{ width: '100%', padding: '5px 8px', fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5 }}
                            onClick={() => handleStopAI(cam.id)}
                          >
                            <Square size={12} /> Pause AI Monitoring
                          </button>
                        ) : (
                          <button
                            className="btn btn-primary btn-sm"
                            style={{ width: '100%', padding: '5px 8px', fontSize: 11, background: '#2563eb', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5 }}
                            disabled={triggeringCam[cam.id]}
                            onClick={() => handleTriggerAI(cam.id)}
                          >
                            <Zap size={12} /> {job?.status === 'stopped' ? 'Resume Live AI Monitoring' : 'Start Live AI Monitoring'}
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
