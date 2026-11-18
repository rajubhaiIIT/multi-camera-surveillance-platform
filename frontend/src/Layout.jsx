import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useState, useCallback } from 'react';
import { getUser, logout } from './api.js';
import { useWebSocket } from './useWebSocket.js';
import {
  Shield,
  LayoutDashboard,
  Camera,
  Bell,
  Activity,
  Car,
  Radio,
  LogOut,
  X,
  Volume2,
  VolumeX,
  Minimize2,
  Maximize2,
  Trash2,
  ExternalLink,
} from 'lucide-react';

export default function Layout() {
  const user = getUser();
  const navigate = useNavigate();
  const location = useLocation();
  const isTrafficPage = location.pathname.startsWith('/traffic');
  const [toasts, setToasts] = useState([]);
  const [isMuted, setIsMuted] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);

  const handleWsMessage = useCallback((msg) => {
    if (msg.type === 'alert' && !isMuted) {
      const alert = msg.payload;
      const toast = {
        id: alert.id || Date.now() + Math.random(),
        title: alert.title,
        desc: alert.description,
        severity: alert.severity || 'high',
        camera_id: alert.camera_id,
        timestamp: alert.timestamp || Date.now() / 1000,
        alert_type: alert.alert_type,
      };
      // Keep at most 3 non-intrusive alert toasts at once
      setToasts(prev => [toast, ...prev.filter(t => t.id !== toast.id)].slice(0, 3));
      // Auto-dismiss after 9 seconds if not manually removed
      setTimeout(() => {
        setToasts(prev => prev.filter(t => t.id !== toast.id));
      }, 9000);
    }
  }, [isMuted]);

  const removeToast = (id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  };

  const clearAllToasts = () => {
    setToasts([]);
  };

  const { connected } = useWebSocket(handleWsMessage);

  const navItems = [
    { to: '/', icon: <LayoutDashboard size={18} />, label: 'Dashboard' },
    { to: '/cameras', icon: <Camera size={18} />, label: 'Cameras' },
    { to: '/traffic', icon: <Radio size={18} />, label: 'Traffic ANPR', badge: 'ZONE' },
    { to: '/alerts', icon: <Bell size={18} />, label: 'Alerts' },
    { to: '/events', icon: <Activity size={18} />, label: 'Events' },
    { to: '/watchlist', icon: <Car size={18} />, label: 'Vehicle Watchlist' },
  ];

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <Shield size={28} color="#3b82f6" />
          <div>
            <h1>Surveillance<br/>Command Center</h1>
            <span className="version">v1.0.0</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="section-label">Navigation</div>
          {navItems.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) => isActive ? 'active' : ''}
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                {item.icon} {item.label}
              </span>
              {item.badge && (
                <span
                  style={{
                    background: '#ef4444',
                    color: 'white',
                    fontSize: 9,
                    fontWeight: 800,
                    padding: '1px 6px',
                    borderRadius: 8,
                    letterSpacing: 0.5,
                  }}
                >
                  {item.badge}
                </span>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="user-info">
            <span>{user?.username || 'unknown'}</span>
            <span className="user-badge">{user?.role || 'viewer'}</span>
          </div>
          <div className="ws-indicator">
            <span className={`ws-dot ${connected ? 'connected' : 'disconnected'}`}></span>
            {connected ? 'Live Connected' : 'Reconnecting...'}
          </div>
          <button className="logout-btn" onClick={logout}>
            <LogOut size={12} /> Sign Out
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main
        className={`main-content ${isTrafficPage ? 'traffic-mode' : ''}`}
        style={
          isTrafficPage
            ? {
                height: '100vh',
                maxHeight: '100vh',
                overflow: 'hidden',
                padding: '10px 14px',
                display: 'flex',
                flexDirection: 'column',
              }
            : undefined
        }
      >
        <Outlet />
      </main>

      {/* Modern, Non-Intrusive Floating Alert Dock (Bottom-Right, Never Blocks Cameras) */}
      {toasts.length > 0 && (
        <div className="alert-dock-container">
          {isMinimized ? (
            <div
              className="alert-dock-min-badge"
              onClick={() => setIsMinimized(false)}
              title="Click to expand alerts"
            >
              <Bell size={15} />
              <span>{toasts.length} Active Alert{toasts.length > 1 ? 's' : ''}</span>
              <Maximize2 size={13} style={{ marginLeft: 4 }} />
            </div>
          ) : (
            <>
              {/* Header Bar with Quick Controls */}
              <div className="alert-dock-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Bell size={14} color="#ef4444" />
                  <span>Security Alerts ({toasts.length})</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <button
                    type="button"
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: isMuted ? '#f59e0b' : '#94a3b8',
                      padding: '2px 4px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 3,
                      fontSize: 11,
                    }}
                    onClick={() => setIsMuted(!isMuted)}
                    title={isMuted ? 'Unmute alert popups' : 'Mute alert popups'}
                  >
                    {isMuted ? <VolumeX size={13} /> : <Volume2 size={13} />}
                  </button>
                  <button
                    type="button"
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: '#94a3b8',
                      padding: '2px 6px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      fontSize: 11,
                    }}
                    onClick={clearAllToasts}
                    title="Dismiss all notifications"
                  >
                    <Trash2 size={12} /> Dismiss All
                  </button>
                  <button
                    type="button"
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: '#94a3b8',
                      padding: '2px 4px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                    }}
                    onClick={() => setIsMinimized(true)}
                    title="Minimize alert window"
                  >
                    <Minimize2 size={13} />
                  </button>
                </div>
              </div>

              {/* Toast Cards with Explicit Remove Button */}
              {toasts.map(t => (
                <div key={t.id} className={`alert-dock-card ${t.severity}`}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                        <span style={{ fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                          🚨 {t.title}
                        </span>
                        <span className={`badge badge-${t.severity}`} style={{ fontSize: 9, padding: '1px 5px' }}>
                          {t.severity}
                        </span>
                      </div>
                      <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.4, marginBottom: 6 }}>
                        {t.desc}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 11, color: '#94a3b8' }}>
                        <span>📷 {t.camera_id}</span>
                        <button
                          type="button"
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#38bdf8',
                            fontSize: 11,
                            fontWeight: 600,
                            padding: 0,
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 3,
                          }}
                          onClick={() => navigate('/alerts')}
                        >
                          View Feed <ExternalLink size={10} />
                        </button>
                      </div>
                    </div>

                    {/* Prominent Remove / Close Button */}
                    <button
                      type="button"
                      className="alert-dock-close-btn"
                      onClick={() => removeToast(t.id)}
                      title="Remove notification"
                    >
                      <X size={15} />
                    </button>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
