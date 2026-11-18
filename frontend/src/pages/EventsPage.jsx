import { useEffect, useState } from 'react';
import { getEvents, getCameras } from '../api.js';
import { Activity, Search, Filter } from 'lucide-react';

function timeAgo(ts) {
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export default function EventsPage() {
  const [events, setEvents] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [searchCam, setSearchCam] = useState('');
  const [eventType, setEventType] = useState('');
  const [globalId, setGlobalId] = useState('');
  const [query, setQuery] = useState('');

  useEffect(() => {
    getCameras().then(setCameras).catch(() => {});
    loadEvents();
  }, []);

  const loadEvents = () => {
    const params = { limit: 100 };
    if (searchCam) params.camera_id = searchCam;
    if (eventType) params.event_type = eventType;
    if (globalId) params.global_id = +globalId;
    if (query) params.query = query;

    getEvents(params).then(setEvents).catch(() => {});
  };

  const handleSearch = (e) => {
    e.preventDefault();
    loadEvents();
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Forensic Events & Trajectories</h2>
          <div className="subtitle">Search multi-camera tracking logs, person identities, and sightings</div>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="card" style={{ marginBottom: 20 }}>
        <form onSubmit={handleSearch} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
          <select value={searchCam} onChange={e => setSearchCam(e.target.value)}>
            <option value="">All Cameras</option>
            {cameras.map(c => (
              <option key={c.id} value={c.id}>{c.name} ({c.id})</option>
            ))}
          </select>

          <select value={eventType} onChange={e => setEventType(e.target.value)}>
            <option value="">All Event Types</option>
            <option value="track">Person Tracklet</option>
            <option value="face">Face Match</option>
            <option value="pose">Pose / Motion</option>
            <option value="weapon">Weapon</option>
            <option value="vehicle">Vehicle / ANPR</option>
          </select>

          <input
            placeholder="Global Person ID (e.g. 102)"
            value={globalId}
            onChange={e => setGlobalId(e.target.value)}
          />

          <input
            placeholder="Search keywords / notes"
            value={query}
            onChange={e => setQuery(e.target.value)}
          />

          <button type="submit" className="btn btn-primary" style={{ justifyContent: 'center' }}>
            <Search size={16} /> Search
          </button>
        </form>
      </div>

      {/* Events Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Camera</th>
              <th>Type</th>
              <th>Track #</th>
              <th>Global ID</th>
              <th>Confidence</th>
              <th>BBox (x1, y1, x2, y2)</th>
              <th>Metadata</th>
            </tr>
          </thead>
          <tbody>
            {events.length === 0 && (
              <tr>
                <td colSpan="8" style={{ textAlign: 'center', padding: 30, color: '#64748b' }}>
                  No event records found matching search criteria.
                </td>
              </tr>
            )}
            {events.map(ev => (
              <tr key={ev.id}>
                <td style={{ whiteSpace: 'nowrap' }}>{timeAgo(ev.timestamp)}</td>
                <td><span className="badge badge-info">{ev.camera_id}</span></td>
                <td><span className="badge badge-low">{ev.event_type}</span></td>
                <td>#{ev.track_id}</td>
                <td>
                  {ev.global_id ? (
                    <span style={{ color: '#38bdf8', fontWeight: 600 }}>GID-{ev.global_id}</span>
                  ) : (
                    <span style={{ color: '#64748b' }}>—</span>
                  )}
                </td>
                <td>{(ev.confidence * 100).toFixed(1)}%</td>
                <td style={{ fontFamily: 'monospace', fontSize: 11 }}>
                  {ev.bbox && ev.bbox.length === 4
                    ? `[${ev.bbox.map(v => Math.round(v)).join(', ')}]`
                    : '—'}
                </td>
                <td style={{ fontSize: 11, color: '#94a3b8' }}>
                  {Object.keys(ev.metadata || {}).length > 0
                    ? JSON.stringify(ev.metadata)
                    : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
