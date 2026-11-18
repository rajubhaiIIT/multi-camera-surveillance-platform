import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login } from '../api.js';
import { Shield } from 'lucide-react';

export default function LoginPage({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(username, password);
      onLogin();
      navigate('/', { replace: true });
    } catch (err) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  }

  function quickLogin(user, pass) {
    setUsername(user);
    setPassword(pass);
  }

  return (
    <div className="login-page">
      <div className="login-box">
        <div style={{ textAlign: 'center', marginBottom: 16 }}>
          <Shield size={48} color="#3b82f6" />
        </div>
        <h2>Surveillance Platform</h2>
        <p className="login-subtitle">Intelligent Multi-Camera Command Center</p>

        {error && <div className="login-error">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Username</label>
            <input type="text" value={username} onChange={e => setUsername(e.target.value)}
              placeholder="Enter username" autoFocus required />
          </div>
          <div className="form-group">
            <label>Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)}
              placeholder="Enter password" required />
          </div>
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? 'Authenticating...' : 'Sign In'}
          </button>
        </form>

        <div className="demo-accounts">
          <h4>Quick Login (Demo Accounts)</h4>
          <div className="demo-row" style={{ cursor: 'pointer' }} onClick={() => quickLogin('admin', 'admin123')}>
            <span>Administrator</span>
            <code>admin / admin123</code>
          </div>
          <div className="demo-row" style={{ cursor: 'pointer' }} onClick={() => quickLogin('operator', 'operator123')}>
            <span>Operator (SOC)</span>
            <code>operator / operator123</code>
          </div>
          <div className="demo-row" style={{ cursor: 'pointer' }} onClick={() => quickLogin('viewer', 'viewer123')}>
            <span>Viewer (Audit)</span>
            <code>viewer / viewer123</code>
          </div>
        </div>
      </div>
    </div>
  );
}
