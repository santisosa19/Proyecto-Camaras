import { Link, Navigate, Route, Routes, useNavigate, useParams } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const DEMO_USERS = [
  { username: "admin", password: "admin123", role: "superadmin" },
  { username: "gerente", password: "gerente123", role: "manager" },
  {
    username: "sucursal",
    password: "sucursal123",
    role: "branch_manager",
    branch_id: "sucursal_1"
  }
];

function useAuth() {
  const [user, setUser] = useState(() => {
    const raw = localStorage.getItem("traffic_user");
    return raw ? JSON.parse(raw) : null;
  });

  const login = (username, password) => {
    const found = DEMO_USERS.find((u) => u.username === username && u.password === password);
    if (!found) return false;
    const safeUser = {
      username: found.username,
      role: found.role,
      branch_id: found.branch_id || null
    };
    localStorage.setItem("traffic_user", JSON.stringify(safeUser));
    setUser(safeUser);
    return true;
  };

  const logout = () => {
    localStorage.removeItem("traffic_user");
    setUser(null);
  };

  return { user, login, logout };
}

async function fetchJson(path) {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  return res.json();
}

function LoginPage({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const submit = (e) => {
    e.preventDefault();
    const ok = onLogin(username, password);
    if (!ok) {
      setError("Credenciales invalidas");
      return;
    }
    navigate("/");
  };

  return (
    <div className="login-wrap">
      <form className="card login-card" onSubmit={submit}>
        <h1>Traffic Dashboard</h1>
        <p>Acceso demo</p>
        <input
          placeholder="Usuario"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          placeholder="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <div className="error">{error}</div>}
        <button type="submit">Ingresar</button>
        <small>admin/admin123, gerente/gerente123, sucursal/sucursal123</small>
      </form>
    </div>
  );
}

function AppLayout({ user, onLogout, children }) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <strong>Traffic Dashboard</strong>
        </div>
        <nav>
          <Link to="/">Overview</Link>
          <Link to="/branches">Sucursales</Link>
        </nav>
        <div className="right-group">
          <span>{user.username}</span>
          <button className="ghost" onClick={onLogout}>
            Salir
          </button>
        </div>
      </header>
      <main className="content">{children}</main>
    </div>
  );
}

function Kpi({ title, value }) {
  return (
    <article className="card kpi">
      <h3>{title}</h3>
      <p>{value}</p>
    </article>
  );
}

function OverviewPage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchJson("/api/v1/dashboard/overview")
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="card error">Error cargando overview: {error}</div>;
  if (!data) return <div className="card">Cargando...</div>;

  return (
    <section className="grid">
      <Kpi title="Sucursales" value={data.total_branches} />
      <Kpi title="Camaras" value={data.total_cameras} />
      <Kpi title="Camaras online" value={data.online_cameras} />
      <Kpi title="Ocupacion actual" value={data.current_occupancy} />
      <Kpi title="Entradas hoy" value={data.entries_today} />
      <Kpi title="Salidas hoy" value={data.exits_today} />
    </section>
  );
}

function BranchesPage({ user }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchJson("/api/v1/dashboard/branches")
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  const branches = useMemo(() => {
    if (!data?.branches) return [];
    if (user.role === "branch_manager") {
      return data.branches.filter((b) => b.branch_id === user.branch_id);
    }
    return data.branches;
  }, [data, user]);

  if (error) return <div className="card error">Error cargando sucursales: {error}</div>;
  if (!data) return <div className="card">Cargando...</div>;

  return (
    <section className="list">
      {branches.map((b) => (
        <article className="card branch-card" key={b.branch_id}>
          <div>
            <h3>{b.branch_name}</h3>
            <small>{b.branch_id}</small>
          </div>
          <div className="branch-metrics">
            <span>Ocupacion: {b.current_occupancy}</span>
            <span>Camaras: {b.online_cameras}/{b.total_cameras}</span>
            <span>Entradas: {b.entries_today}</span>
            <span>Salidas: {b.exits_today}</span>
          </div>
          <Link to={`/branches/${b.branch_id}`}>Ver detalle</Link>
        </article>
      ))}
    </section>
  );
}

function BranchDetailPage({ user }) {
  const { branchId } = useParams();
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchJson(`/api/v1/dashboard/branches/${branchId}`)
      .then(setData)
      .catch((e) => setError(e.message));
  }, [branchId]);

  if (user.role === "branch_manager" && user.branch_id !== branchId) {
    return <Navigate to="/branches" replace />;
  }

  if (error) return <div className="card error">Error cargando detalle: {error}</div>;
  if (!data) return <div className="card">Cargando...</div>;

  return (
    <section className="detail-wrap">
      <div className="grid">
        <Kpi title="Sucursal" value={data.branch_name} />
        <Kpi title="Ocupacion actual" value={data.current_occupancy} />
        <Kpi title="Entradas hoy" value={data.entries_today} />
        <Kpi title="Salidas hoy" value={data.exits_today} />
      </div>

      <article className="card">
        <h3>Camaras</h3>
        <table>
          <thead>
            <tr>
              <th>Camara</th>
              <th>Estado</th>
              <th>FPS</th>
              <th>Errores</th>
              <th>Conteo actual</th>
            </tr>
          </thead>
          <tbody>
            {data.cameras.map((c) => (
              <tr key={c.camera_id}>
                <td>{c.camera_name || c.camera_id}</td>
                <td>{c.is_connected ? "Online" : "Offline"}</td>
                <td>{Number(c.fps || 0).toFixed(1)}</td>
                <td>{c.error_count || 0}</td>
                <td>{c.current_count || 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </article>

      <article className="card">
        <h3>Flujo por hora (ultimas 24h)</h3>
        <table>
          <thead>
            <tr>
              <th>Hora</th>
              <th>Entradas</th>
              <th>Salidas</th>
            </tr>
          </thead>
          <tbody>
            {data.hourly_flow.map((h) => (
              <tr key={h.hour}>
                <td>{h.hour}</td>
                <td>{h.entry}</td>
                <td>{h.exit}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </article>
    </section>
  );
}

function PrivateApp({ user, logout }) {
  return (
    <AppLayout user={user} onLogout={logout}>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/branches" element={<BranchesPage user={user} />} />
        <Route path="/branches/:branchId" element={<BranchDetailPage user={user} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppLayout>
  );
}

export default function App() {
  const { user, login, logout } = useAuth();

  if (!user) {
    return <LoginPage onLogin={login} />;
  }

  return <PrivateApp user={user} logout={logout} />;
}
