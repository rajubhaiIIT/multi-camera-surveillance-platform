import { useEffect, useState } from 'react';
import { getAlerts, acknowledgeAlert, deleteAlert, getUser } from '../api.js';
import { Bell, CheckCircle, Trash2, Filter } from 'lucide-react';

function timeAgo(ts) {
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export default function AlertsPage() {
  const [alerts, setAlerts] = useState([]);
  const [severityFilter, setSeverityFilter] = useState('');
  const [typeFilter, setTypeFilter] = useState('');
  const [ackFilter, setAckFilter] = useState('');
  const user = getUser();
  const canAck = user?.role === 'admin' || user?.role === 'operator';

  const loadAlerts = () => {
    const params = { limit: 100 };
    if (severityFilter) params.severity = severityFilter;
    if (typeFilter) params.alert_type = typeFilter;
    if (ackFilter !== '') params.acknowledged = ackFilter === 'true';

    getAlerts(params).then(setAlerts).catch(() => {});
  };

  useEffect(() => {
    loadAlerts();
    const interval = setInterval(loadAlerts, 4000);
    return () => clearInterval(interval);
  }, [severityFilter, typeFilter, ackFilter]);

  async function handleAck(id) {
    const note = prompt('Enter acknowledgment / resolution note (optional):', 'Verified on CCTV');
    if (note === null) return;
    try {
      await acknowledgeAlert(id, { note, operator: user?.username });
      loadAlerts();
    } catch (err) {
      alert(err.message);
    }
  }

  async function handleDelete(id) {
    if (!confirm(`Delete alert ${id}?`)) return;
    try {
      await deleteAlert(id);
      loadAlerts();
    } catch (err) {
      alert(err.message);
    }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Security Alerts Feed</h2>
          <div className="subtitle">Real-time threat notifications, falls, weapons, and ANPR hotlists</div>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: '#94a3b8' }}>
            <Filter size={16} /> Filters:
          </div>
          <select value={severityFilter} onChange={e => setSeverityFilter(e.target.value)}>
            <option value="">All Severities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
            <option value="info">Info</option>
          </select>
          <select value={typeFilter} onChange={e => setTypeFilter(e.target.value)}>
            <option value="">All Types</option>
            <option value="fall">Fall</option>
            <option value="fight">Fight</option>
            <option value="weapon">Weapon</option>
            <option value="watchlist_plate">Watchlist Plate</option>
            <option value="loitering">Loitering</option>
            <option value="perimeter">Perimeter</option>
          </select>
          <select value={ackFilter} onChange={e => setAckFilter(e.target.value)}>
            <option value="">All Statuses</option>
            <option value="false">Unacknowledged Only</option>
            <option value="true">Acknowledged Only</option>
          </select>
          <button className="btn btn-outline btn-sm" onClick={loadAlerts}>
            Refresh
          </button>
        </div>
      </div>

      {/* Alerts List */}
      <div className="alert-list">
        {alerts.length === 0 && (
          <div className="card" style={{ textAlign: 'center', color: '#64748b', padding: 40 }}>
            <Bell size={36} style={{ margin: '0 auto 12px', display: 'block', opacity: 0.5 }} />
            No alerts matching current filters.
          </div>
        )}
        {alerts.map(a => (
          <div key={a.id} className={`alert-card ${a.severity}`}>
            <div className="alert-body">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <span className="alert-title">{a.title}</span>
                <span className={`badge badge-${a.severity}`}>{a.severity}</span>
                <span className="badge badge-info">{a.alert_type}</span>
                {a.acknowledged ? (
                  <span className="badge badge-online">ACKNOWLEDGED</span>
                ) : (
                  <span className="badge badge-critical">UNRESOLVED</span>
                )}
              </div>
              <div className="alert-desc">{a.description}</div>
              <div className="alert-meta">
                <span>📷 Camera: {a.camera_id}</span>
                <span>⏱️ {timeAgo(a.timestamp)}</span>
                {a.track_id && <span>Track #{a.track_id}</span>}
                {a.global_id && <span>Global ID: {a.global_id}</span>}
                {a.acknowledged && (
                  <span style={{ color: '#22c55e' }}>
                    ✓ Ack by {a.acknowledged_by} ({timeAgo(a.acknowledged_at)})
                  </span>
                )}
                {a.metadata?.ack_note && (
                  <span style={{ fontStyle: 'italic' }}>Note: "{a.metadata.ack_note}"</span>
                )}
              </div>
            </div>
            <div className="alert-actions">
              {!a.acknowledged && canAck && (
                <button className="btn btn-success btn-sm" onClick={() => handleAck(a.id)}>
                  <CheckCircle size={14} /> Ack
                </button>
              )}
              {(user?.role === 'admin' || canAck) && (
                <button className="btn btn-danger btn-sm" onClick={() => handleDelete(a.id)} title="Delete / Remove Alert">
                  <Trash2 size={14} /> Remove
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
