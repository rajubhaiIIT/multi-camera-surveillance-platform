import { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { isLoggedIn } from './api.js';
import Layout from './Layout.jsx';
import LoginPage from './pages/LoginPage.jsx';
import DashboardPage from './pages/DashboardPage.jsx';
import CamerasPage from './pages/CamerasPage.jsx';
import AlertsPage from './pages/AlertsPage.jsx';
import EventsPage from './pages/EventsPage.jsx';
import WatchlistPage from './pages/WatchlistPage.jsx';
import TrafficANPRPage from './pages/TrafficANPRPage.jsx';

function ProtectedRoute({ children }) {
  return isLoggedIn() ? children : <Navigate to="/login" replace />;
}

export default function App() {
  const [authTick, setAuthTick] = useState(0);
  const onLogin = () => setAuthTick(t => t + 1);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage onLogin={onLogin} />} />
        <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
          <Route index element={<DashboardPage />} />
          <Route path="cameras" element={<CamerasPage />} />
          <Route path="alerts" element={<AlertsPage />} />
          <Route path="events" element={<EventsPage />} />
          <Route path="watchlist" element={<WatchlistPage />} />
          <Route path="traffic" element={<TrafficANPRPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
