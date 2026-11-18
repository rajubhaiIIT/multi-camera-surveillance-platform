import { useEffect, useState } from 'react';
import { getStats, getAlerts, getCameras } from '../api.js';
import { Camera, Bell, ShieldAlert, Activity, Clock, Server } from 'lucide-react';

function timeAgo(ts) {
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export default function DashboardPage() {
  const [stats, setStats] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [cameras, setCameras] = useState([]);

  useEffect(() => {
    getStats().then(setStats).catch(() => {});
    getAlerts({ limit: 5, acknowledged: false }).then(setAlerts).catch(() => {});
    getCameras().then(setCameras).catch(() => {});

    const interval = setInterval(() => {
      getStats().then(setStats).catch(() => {});
      getAlerts({ limit: 5, acknowledged: false }).then(setAlerts).catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Operations Dashboard</h2>
          <div className="subtitle">Real-time surveillance system overview</div>
        </div>
        <div className="ws-indicator">
          <Clock size={14} />
          Uptime: {stats ? `${Math.floor(stats.uptime_seconds / 60)}m` : '—'}
        </div>
      </div>

      {/* Stats Grid */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(59,130,246,0.15)', color: '#3b82f6' }}>
            <Camera size={22} />
          </div>
          <div>
            <div className="stat-value">{stats?.cameras_online ?? '—'}<span style={{ fontSize: 16, color: '#64748b' }}>/{stats?.cameras_total ?? '—'}</span></div>
            <div className="stat-label">Cameras Online</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(239,68,68,0.15)', color: '#ef4444' }}>
            <ShieldAlert size={22} />
          </div>
          <div>
            <div className="stat-value">{stats?.active_alerts_count ?? '—'}</div>
            <div className="stat-label">Active Alerts</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(245,158,11,0.15)', color: '#f59e0b' }}>
            <Bell size={22} />
          </div>
          <div>
            <div className="stat-value">{stats?.alerts_by_severity?.critical ?? 0}</div>
            <div className="stat-label">Critical Threats</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(34,197,94,0.15)', color: '#22c55e' }}>
            <Activity size={22} />
          </div>
          <div>
            <div className="stat-value">{stats?.total_events ?? 0}</div>
            <div className="stat-label">Total Events</div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Recent Alerts */}
        <div className="card">
          <div className="card-header">
            <h3>🚨 Active Alerts</h3>
            <a href="/alerts" className="btn btn-outline btn-sm">View All</a>
          </div>
          <div className="alert-list">
            {alerts.length === 0 && <div style={{ color: '#64748b', fontSize: 13 }}>No active alerts</div>}
            {alerts.map(a => (
              <div key={a.id} className={`alert-card ${a.severity}`}>
                <div className="alert-body">
                  <div className="alert-title">{a.title}</div>
                  <div className="alert-desc">{a.description}</div>
                  <div className="alert-meta">
                    <span className={`badge badge-${a.severity}`}>{a.severity}</span>
                    <span>📷 {a.camera_id}</span>
                    <span>{timeAgo(a.timestamp)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Services Health */}
        <div className="card">
          <div className="card-header">
            <h3><Server size={16} /> Infrastructure Services</h3>
          </div>
          <div className="services-grid">
            {stats?.services_status && Object.entries(stats.services_status).map(([name, status]) => (
              <div key={name} className="service-card">
                <span className="service-dot" style={{
                  background: status.includes('healthy') ? '#22c55e' : '#f59e0b'
                }}></span>
                <div>
                  <div className="service-name">{name}</div>
                  <div className="service-status">{status}</div>
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: 20 }}>
            <h3 style={{ fontSize: 14, marginBottom: 10 }}>📷 Camera Fleet</h3>
            {cameras.map(c => (
              <div key={c.id} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--border)', fontSize: 13 }}>
                <span>{c.name} ({c.id})</span>
                <span className={`badge badge-${c.status === 'online' ? 'online' : 'offline'}`}>{c.status}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
