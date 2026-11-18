import { useEffect, useState } from 'react';
import { getWatchlist, addWatchlistPlate, removeWatchlistPlate, getVehicleSightings, getUser } from '../api.js';
import { Car, Plus, Trash2, ShieldAlert, Eye } from 'lucide-react';

function timeAgo(ts) {
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export default function WatchlistPage() {
  const [watchlist, setWatchlist] = useState([]);
  const [sightings, setSightings] = useState([]);
  const [showAdd, setShowAdd] = useState(false);
  const [form, setForm] = useState({ plate: '', reason: 'Stolen Vehicle', owner: '', notes: '' });
  const user = getUser();
  const canEdit = user?.role === 'admin' || user?.role === 'operator';

  const load = () => {
    getWatchlist().then(setWatchlist).catch(() => {});
    getVehicleSightings({ limit: 20 }).then(setSightings).catch(() => {});
  };

  useEffect(() => { load(); }, []);

  async function handleAdd(e) {
    e.preventDefault();
    try {
      await addWatchlistPlate(form);
      setShowAdd(false);
      setForm({ plate: '', reason: 'Stolen Vehicle', owner: '', notes: '' });
      load();
    } catch (err) {
      alert(err.message);
    }
  }

  async function handleRemove(plate) {
    if (!confirm(`Remove plate ${plate} from hotlist?`)) return;
    try {
      await removeWatchlistPlate(plate);
      load();
    } catch (err) {
      alert(err.message);
    }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Vehicle Watchlist & ANPR Hotlist</h2>
          <div className="subtitle">Automatic license plate recognition watchlist and sighting logs</div>
        </div>
        {canEdit && (
          <button className="btn btn-primary" onClick={() => setShowAdd(!showAdd)}>
            <Plus size={16} /> Flag Vehicle
          </button>
        )}
      </div>

      {/* Add Plate Modal / Card */}
      {showAdd && (
        <div className="card" style={{ marginBottom: 20 }}>
          <h3 style={{ marginBottom: 12 }}>Add License Plate to Hotlist</h3>
          <form onSubmit={handleAdd} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr auto', gap: 12 }}>
            <input
              placeholder="Plate # (e.g. MH12DE1433)"
              value={form.plate}
              onChange={e => setForm({ ...form, plate: e.target.value })}
              required
            />
            <input
              placeholder="Reason (e.g. Stolen Vehicle)"
              value={form.reason}
              onChange={e => setForm({ ...form, reason: e.target.value })}
              required
            />
            <input
              placeholder="Owner / Agency"
              value={form.owner}
              onChange={e => setForm({ ...form, owner: e.target.value })}
            />
            <input
              placeholder="Additional Notes"
              value={form.notes}
              onChange={e => setForm({ ...form, notes: e.target.value })}
            />
            <button type="submit" className="btn btn-danger">Add to Hotlist</button>
          </form>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 16 }}>
        {/* Active Watchlist */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 15 }}>
              <ShieldAlert size={18} color="#ef4444" /> Flagged Plates ({watchlist.length})
            </h3>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>License Plate</th>
                <th>Reason</th>
                <th>Owner / Info</th>
                <th>Added</th>
                {canEdit && <th>Action</th>}
              </tr>
            </thead>
            <tbody>
              {watchlist.length === 0 && (
                <tr>
                  <td colSpan={canEdit ? 5 : 4} style={{ textAlign: 'center', padding: 30, color: '#64748b' }}>
                    No vehicles currently on the watchlist.
                  </td>
                </tr>
              )}
              {watchlist.map(item => (
                <tr key={item.plate}>
                  <td>
                    <span style={{
                      display: 'inline-block',
                      background: '#1e293b',
                      border: '2px solid #cbd5e1',
                      borderRadius: 4,
                      padding: '3px 8px',
                      fontWeight: 700,
                      letterSpacing: 1.5,
                      color: '#f8fafc',
                      fontFamily: 'monospace',
                      fontSize: 13,
                    }}>
                      {item.plate}
                    </span>
                  </td>
                  <td><span className="badge badge-critical">{item.reason}</span></td>
                  <td>
                    <div>{item.owner}</div>
                    {item.notes && <div style={{ fontSize: 11, color: '#64748b' }}>{item.notes}</div>}
                  </td>
                  <td style={{ whiteSpace: 'nowrap' }}>{timeAgo(item.added_at)}</td>
                  {canEdit && (
                    <td>
                      <button className="btn btn-outline btn-sm" onClick={() => handleRemove(item.plate)}>
                        <Trash2 size={12} />
                      </button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Recent Sightings */}
        <div className="card">
          <div className="card-header">
            <h3><Eye size={18} /> Recent Vehicle Sightings</h3>
          </div>
          <div className="alert-list">
            {sightings.length === 0 && (
              <div style={{ color: '#64748b', fontSize: 13, textAlign: 'center', padding: 20 }}>
                No vehicle sightings logged recently.
              </div>
            )}
            {sightings.map(s => (
              <div key={s.id} className="alert-card low" style={{ padding: 10 }}>
                <div className="alert-body">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, fontSize: 13 }}>
                      🚘 {s.metadata?.plate || 'Unknown Plate'}
                    </span>
                    <span className="badge badge-info">{s.camera_id}</span>
                  </div>
                  <div className="alert-meta" style={{ marginTop: 4 }}>
                    <span>Speed: {s.metadata?.speed_kmh ? `${s.metadata.speed_kmh} km/h` : 'Stationary'}</span>
                    <span>{timeAgo(s.timestamp)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
