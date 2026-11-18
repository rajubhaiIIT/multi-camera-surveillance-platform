/* API client — talks to the Phase 7 FastAPI backend. */
const API_BASE = '/api';

let _token = localStorage.getItem('token');
let _user = JSON.parse(localStorage.getItem('user') || 'null');

function headers() {
  const h = { 'Content-Type': 'application/json' };
  if (_token) h['Authorization'] = `Bearer ${_token}`;
  return h;
}

async function request(method, path, body) {
  const opts = { method, headers: headers() };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(`${API_BASE}${path}`, opts);
  if (res.status === 401) { logout(); throw new Error('Unauthorized'); }
  if (res.status === 204) return null;
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || res.statusText); }
  return res.json();
}

/* ─── Auth ─── */
export async function login(username, password) {
  const form = new URLSearchParams();
  form.append('username', username);
  form.append('password', password);
  const res = await fetch(`${API_BASE}/auth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: form,
  });
  if (!res.ok) throw new Error('Invalid credentials');
  const data = await res.json();
  _token = data.access_token;
  _user = { username: data.username, role: data.role };
  localStorage.setItem('token', _token);
  localStorage.setItem('user', JSON.stringify(_user));
  return data;
}

export function logout() {
  _token = null;
  _user = null;
  localStorage.removeItem('token');
  localStorage.removeItem('user');
  window.location.href = '/';
}

export function getUser() { return _user; }
export function isLoggedIn() { return !!_token; }

/* ─── Cameras ─── */
export const getCameras = () => request('GET', '/cameras');
export const getCamera = (id) => request('GET', `/cameras/${id}`);
export const createCamera = (data) => request('POST', '/cameras', data);
export const updateCamera = (id, data) => request('PUT', `/cameras/${id}`, data);
export const deleteCamera = (id) => request('DELETE', `/cameras/${id}`);
export const getCameraStreams = (id) => request('GET', `/cameras/${id}/streams`);
export const getAvailableVideos = () => request('GET', '/cameras/available_videos');
export const attachVideoToCamera = (id, filename) => request('POST', `/cameras/${id}/attach_video`, { filename });
export const triggerAIAnalysis = (id, maxFrames) => request('POST', `/cameras/${id}/analyze${maxFrames ? `?max_frames=${maxFrames}` : ''}`);
export const getAIAnalysisStatus = (id) => request('GET', `/cameras/${id}/analysis_status`);
export const stopAIAnalysis = (id) => request('POST', `/cameras/${id}/stop_analysis`);

export async function uploadCameraVideo(formData) {
  const h = {};
  if (_token) h['Authorization'] = `Bearer ${_token}`;
  const res = await fetch(`${API_BASE}/cameras/upload`, {
    method: 'POST',
    headers: h,
    body: formData,
  });
  if (res.status === 401) { logout(); throw new Error('Unauthorized'); }
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    throw new Error(e.detail || res.statusText);
  }
  return res.json();
}

/* ─── Alerts ─── */
export const getAlerts = (params) => {
  const qs = new URLSearchParams();
  if (params) Object.entries(params).forEach(([k, v]) => { if (v != null) qs.set(k, v); });
  return request('GET', `/alerts?${qs}`);
};
export const getAlert = (id) => request('GET', `/alerts/${id}`);
export const createAlert = (data) => request('POST', '/alerts', data);
export const acknowledgeAlert = (id, data) => request('POST', `/alerts/${id}/ack`, data || {});
export const deleteAlert = (id) => request('DELETE', `/alerts/${id}`);

/* ─── Events ─── */
export const getEvents = (params) => {
  const qs = new URLSearchParams();
  if (params) Object.entries(params).forEach(([k, v]) => { if (v != null) qs.set(k, v); });
  return request('GET', `/events?${qs}`);
};
export const createEvent = (data) => request('POST', '/events', data);

/* ─── Vehicles / Watchlist ─── */
export const getWatchlist = () => request('GET', '/vehicles/watchlist');
export const addWatchlistPlate = (data) => request('POST', '/vehicles/watchlist', data);
export const removeWatchlistPlate = (plate) => request('DELETE', `/vehicles/watchlist/${plate}`);
export const getVehicleSightings = (params) => {
  const qs = new URLSearchParams();
  if (params) Object.entries(params).forEach(([k, v]) => { if (v != null) qs.set(k, v); });
  return request('GET', `/vehicles/sightings?${qs}`);
};

/* ─── Stats & Health ─── */
export const getStats = () => request('GET', '/stats');
export const getHealth = () => request('GET', '/health');

/* ─── Users ─── */
export const getUsers = () => request('GET', '/auth/users');
export const createUser = (data) => request('POST', '/auth/users', data);
