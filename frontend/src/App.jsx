import { Link, Navigate, Route, Routes, useNavigate, useParams } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

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

const ROLE_LABELS = {
  superadmin: "Superadmin",
  manager: "Manager",
  branch_manager: "Sucursal"
};

const CHART_COLORS = {
  entry: "#17a672",
  exit: "#df7c2a",
  occupancy: "#1459c7",
  online: "#1b8f5a",
  offline: "#cc3d2f",
  high_error: "#da8f19"
};

function formatNumber(value) {
  return new Intl.NumberFormat("es-AR").format(Number(value || 0));
}

function formatDateTime(value) {
  if (!value) return "-";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return "-";
  return dt.toLocaleString("es-AR", { hour12: false });
}

function formatHour(value) {
  if (!value) return "-";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return value;
  return dt.toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit", hour12: false });
}

function toQuery(params = {}) {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, val]) => {
    if (val !== undefined && val !== null && `${val}`.trim() !== "") {
      searchParams.set(key, String(val));
    }
  });
  return searchParams.toString();
}

function severityLabel(severity) {
  if (severity === "high") return "Alta";
  if (severity === "medium") return "Media";
  return "Baja";
}

function branchShortName(name, max = 16) {
  if (!name) return "-";
  return name.length > max ? `${name.slice(0, max)}…` : name;
}

function chartTickLabel(isoDate, windowHours) {
  const dt = new Date(isoDate);
  if (Number.isNaN(dt.getTime())) return "-";

  if (windowHours <= 24) {
    return dt.toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit", hour12: false });
  }

  if (windowHours <= 72) {
    const day = dt.toLocaleDateString("es-AR", { day: "2-digit", month: "2-digit" });
    const hour = dt.toLocaleTimeString("es-AR", { hour: "2-digit", hour12: false });
    return `${day} ${hour}h`;
  }

  return dt.toLocaleDateString("es-AR", { day: "2-digit", month: "2-digit" });
}

function useAuth() {
  const [user, setUser] = useState(() => {
    const raw = localStorage.getItem("traffic_user");
    return raw ? JSON.parse(raw) : null;
  });

  const login = (username, password) => {
    const found = DEMO_USERS.find((candidate) => candidate.username === username && candidate.password === password);
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

function StatusPill({ online }) {
  return <span className={`status-pill ${online ? "online" : "offline"}`}>{online ? "Online" : "Offline"}</span>;
}

function SeverityPill({ severity }) {
  return <span className={`severity-pill ${severity || "low"}`}>{severityLabel(severity)}</span>;
}

function KpiCard({ title, value, helper }) {
  return (
    <article className="kpi-card">
      <p className="kpi-title">{title}</p>
      <p className="kpi-value">{value}</p>
      {helper ? <p className="kpi-helper">{helper}</p> : null}
    </article>
  );
}

function ChartTooltip({ active, label, payload }) {
  if (!active || !payload?.length) return null;

  return (
    <div className="chart-tooltip">
      <p>{label}</p>
      {payload.map((item) => (
        <div key={item.dataKey} className="tooltip-row">
          <span style={{ color: item.color }}>{item.name}:</span>
          <strong>{formatNumber(item.value)}</strong>
        </div>
      ))}
    </div>
  );
}

function TrafficChart({ series, windowHours }) {
  const data = useMemo(
    () =>
      (series || []).map((item) => ({
        ...item,
        chartLabel: chartTickLabel(item.hour, windowHours),
        tooltipLabel: formatDateTime(item.hour)
      })),
    [series, windowHours]
  );

  if (!data.length) {
    return <div className="empty-state">Sin datos de flujo para el rango seleccionado.</div>;
  }

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 6, right: 14, bottom: 6, left: 4 }}>
          <CartesianGrid stroke="#e6edf2" strokeDasharray="3 3" />
          <XAxis dataKey="chartLabel" tick={{ fontSize: 11 }} minTickGap={24} />
          <YAxis yAxisId="left" tick={{ fontSize: 11 }} allowDecimals={false} />
          <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} allowDecimals={false} />
          <Tooltip
            content={<ChartTooltip />}
            formatter={(value, name) => [formatNumber(value), name]}
            labelFormatter={(_, payload) => payload?.[0]?.payload?.tooltipLabel || "-"}
          />
          <Legend />
          <Bar yAxisId="left" dataKey="entry" name="Entradas" fill={CHART_COLORS.entry} radius={[6, 6, 0, 0]} />
          <Bar yAxisId="left" dataKey="exit" name="Salidas" fill={CHART_COLORS.exit} radius={[6, 6, 0, 0]} />
          <Line yAxisId="right" type="monotone" dataKey="occupancy_end" name="Ocupación" stroke={CHART_COLORS.occupancy} strokeWidth={2.2} dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function CameraHealthDonut({ health }) {
  const total = Number(health?.total || 0);
  const online = Number(health?.online || 0);
  const offline = Number(health?.offline || 0);
  const highError = Number(health?.high_error || 0);

  const data = [
    { name: "Online", value: online, color: CHART_COLORS.online },
    { name: "Offline", value: offline, color: CHART_COLORS.offline },
    { name: "Error alto", value: highError, color: CHART_COLORS.high_error }
  ].filter((item) => item.value > 0);

  if (!total) {
    return <div className="empty-state">Sin cámaras registradas.</div>;
  }

  return (
    <div className="donut-wrap">
      <ResponsiveContainer width="100%" height={245}>
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={62} outerRadius={92} paddingAngle={2}>
            {data.map((item) => (
              <Cell key={item.name} fill={item.color} />
            ))}
          </Pie>
          <Tooltip formatter={(value) => formatNumber(value)} />
          <Legend verticalAlign="bottom" height={26} />
        </PieChart>
      </ResponsiveContainer>
      <p className="donut-note">Disponibilidad: {Math.round((online / Math.max(1, total)) * 100)}% online</p>
    </div>
  );
}

function TopBranchesChart({ branches }) {
  const data = useMemo(
    () =>
      (branches || []).map((item) => ({
        ...item,
        shortName: branchShortName(item.branch_name)
      })),
    [branches]
  );

  if (!data.length) {
    return <div className="empty-state">Sin datos de sucursales.</div>;
  }

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 10, left: 8, bottom: 4 }}>
          <CartesianGrid stroke="#e6edf2" strokeDasharray="3 3" />
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="shortName" width={110} tick={{ fontSize: 11 }} />
          <Tooltip formatter={(value, name) => [formatNumber(value), name]} />
          <Legend />
          <Bar dataKey="entries_today" name="Entradas" fill={CHART_COLORS.entry} radius={[0, 6, 6, 0]} />
          <Bar dataKey="current_occupancy" name="Ocupación" fill={CHART_COLORS.occupancy} radius={[0, 6, 6, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function BranchOverviewChart({ branches }) {
  const data = useMemo(
    () =>
      (branches || []).slice(0, 10).map((item) => ({
        ...item,
        shortName: branchShortName(item.branch_name, 14)
      })),
    [branches]
  );

  if (!data.length) {
    return <div className="empty-state">Sin datos para comparar sucursales.</div>;
  }

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid stroke="#e6edf2" strokeDasharray="3 3" />
          <XAxis dataKey="shortName" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
          <Tooltip formatter={(value, name) => [formatNumber(value), name]} />
          <Legend />
          <Bar dataKey="entries_today" name="Entradas" fill={CHART_COLORS.entry} radius={[6, 6, 0, 0]} />
          <Bar dataKey="exits_today" name="Salidas" fill={CHART_COLORS.exit} radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function CameraLoadChart({ cameras }) {
  const data = useMemo(
    () =>
      (cameras || []).map((camera) => ({
        ...camera,
        shortName: branchShortName(camera.camera_name || camera.camera_id, 14)
      })),
    [cameras]
  );

  if (!data.length) {
    return <div className="empty-state">Sin cámaras para graficar.</div>;
  }

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 8, right: 10, left: 0, bottom: 10 }}>
          <CartesianGrid stroke="#e6edf2" strokeDasharray="3 3" />
          <XAxis dataKey="shortName" tick={{ fontSize: 11 }} />
          <YAxis yAxisId="left" allowDecimals={false} tick={{ fontSize: 11 }} />
          <YAxis yAxisId="right" orientation="right" allowDecimals={false} tick={{ fontSize: 11 }} />
          <Tooltip formatter={(value, name) => [formatNumber(value), name]} />
          <Legend />
          <Bar yAxisId="left" dataKey="current_count" name="Conteo actual" fill={CHART_COLORS.occupancy} radius={[6, 6, 0, 0]} />
          <Line yAxisId="right" type="monotone" dataKey="error_count" name="Errores" stroke={CHART_COLORS.offline} strokeWidth={2} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function LoginPage({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const submit = (event) => {
    event.preventDefault();
    const ok = onLogin(username, password);
    if (!ok) {
      setError("Credenciales inválidas");
      return;
    }
    navigate("/");
  };

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <h1>Traffic Control Center</h1>
        <p>Monitoreo operativo de sucursales y cámaras</p>
        <label>
          Usuario
          <input placeholder="usuario" value={username} onChange={(event) => setUsername(event.target.value)} />
        </label>
        <label>
          Password
          <input
            placeholder="password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        {error ? <div className="error-box">{error}</div> : null}
        <button type="submit">Ingresar</button>
        <small>Demo: admin/admin123, gerente/gerente123, sucursal/sucursal123</small>
      </form>
    </div>
  );
}

function AppLayout({ user, onLogout, children }) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-box">
          <p className="brand-kicker">Traffic Analysis</p>
          <h2>Operations Dashboard</h2>
        </div>
        <nav className="nav-menu">
          <Link to="/">Visión general</Link>
          <Link to="/branches">Sucursales</Link>
        </nav>
        <div className="sidebar-footer">
          <p>{user.username}</p>
          <span>{ROLE_LABELS[user.role] || user.role}</span>
          <button className="ghost" onClick={onLogout}>
            Cerrar sesión
          </button>
        </div>
      </aside>
      <div className="main-area">
        <header className="main-header">
          <div>
            <h1>Panel de Tráfico</h1>
            <p>Datos en tiempo real para operación de tiendas</p>
          </div>
          <div className="clock-chip">{new Date().toLocaleString("es-AR", { hour12: false })}</div>
        </header>
        <main className="content">{children}</main>
      </div>
    </div>
  );
}

function OverviewPage() {
  const [hours, setHours] = useState(24);
  const [refreshSeconds, setRefreshSeconds] = useState(30);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const query = toQuery({ hours });
        const payload = await fetchJson(`/api/v1/dashboard/overview?${query}`);
        if (!cancelled) {
          setData(payload);
          setError("");
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      }
    };

    load();
    const intervalId = refreshSeconds > 0 ? setInterval(load, refreshSeconds * 1000) : null;

    return () => {
      cancelled = true;
      if (intervalId) clearInterval(intervalId);
    };
  }, [hours, refreshSeconds]);

  if (loading) return <div className="panel">Cargando overview...</div>;
  if (error) return <div className="panel error-box">Error cargando overview: {error}</div>;

  const onlineRatio = data.total_cameras ? Math.round((100 * data.online_cameras) / data.total_cameras) : 0;

  return (
    <section className="dashboard-grid">
      <article className="panel controls-row">
        <div className="control-group">
          <label>Ventana</label>
          <select value={hours} onChange={(event) => setHours(Number(event.target.value))}>
            <option value={24}>Últimas 24h</option>
            <option value={48}>Últimas 48h</option>
            <option value={72}>Últimas 72h</option>
            <option value={168}>Últimos 7 días</option>
          </select>
        </div>
        <div className="control-group">
          <label>Auto-refresh</label>
          <select value={refreshSeconds} onChange={(event) => setRefreshSeconds(Number(event.target.value))}>
            <option value={15}>15s</option>
            <option value={30}>30s</option>
            <option value={60}>60s</option>
            <option value={0}>Manual</option>
          </select>
        </div>
        <div className="updated-at">Actualizado: {formatDateTime(data.timestamp)}</div>
      </article>

      <div className="kpi-grid">
        <KpiCard title="Ocupación actual" value={formatNumber(data.current_occupancy)} helper="Total en todas las sucursales" />
        <KpiCard title="Entradas hoy" value={formatNumber(data.entries_today)} helper={`Neto: ${formatNumber(data.net_today)}`} />
        <KpiCard title="Salidas hoy" value={formatNumber(data.exits_today)} helper="Hasta el momento" />
        <KpiCard title="Sucursales" value={formatNumber(data.total_branches)} helper="Con actividad reportada" />
        <KpiCard
          title="Estado de cámaras"
          value={`${formatNumber(data.online_cameras)}/${formatNumber(data.total_cameras)}`}
          helper={`${onlineRatio}% online`}
        />
        <KpiCard title="Alertas" value={formatNumber((data.alerts || []).length)} helper="Prioridad operativa" />
      </div>

      <article className="panel flow-panel">
        <div className="panel-header">
          <h3>Tendencia de tráfico y ocupación</h3>
          <p>Entradas, salidas y ocupación acumulada por hora</p>
        </div>
        <TrafficChart series={data.flow_series || []} windowHours={hours} />
      </article>

      <div className="charts-grid two-cols">
        <article className="panel">
          <div className="panel-header">
            <h3>Salud de cámaras</h3>
            <p>Distribución de estado operativo</p>
          </div>
          <CameraHealthDonut health={data.camera_health} />
        </article>

        <article className="panel">
          <div className="panel-header">
            <h3>Top sucursales</h3>
            <p>Comparativo de entradas y ocupación</p>
          </div>
          <TopBranchesChart branches={data.top_branches || []} />
        </article>
      </div>

      <article className="panel alerts-panel">
        <div className="panel-header">
          <h3>Alertas operativas</h3>
          <p>Prioridad para intervención</p>
        </div>
        <div className="alerts-list">
          {(data.alerts || []).length === 0 ? <p className="empty-state">Sin alertas activas.</p> : null}
          {(data.alerts || []).map((alert, idx) => (
            <div key={`${alert.camera_id || idx}-${alert.title}`} className="alert-item">
              <SeverityPill severity={alert.severity} />
              <div>
                <p>{alert.title}</p>
                <small>{alert.description}</small>
              </div>
            </div>
          ))}
        </div>
      </article>
    </section>
  );
}

function BranchesPage({ user }) {
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("entries");
  const [order, setOrder] = useState("desc");
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const query = toQuery({ q: search || undefined, sort_by: sortBy, order });
        const payload = await fetchJson(`/api/v1/dashboard/branches?${query}`);
        if (!cancelled) {
          setData(payload);
          setError("");
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    };

    const timeoutId = setTimeout(load, 250);
    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
    };
  }, [search, sortBy, order]);

  const branches = useMemo(() => {
    if (!data?.branches) return [];
    if (user.role === "branch_manager") {
      return data.branches.filter((branch) => branch.branch_id === user.branch_id);
    }
    return data.branches;
  }, [data, user]);

  if (error) return <div className="panel error-box">Error cargando sucursales: {error}</div>;
  if (!data) return <div className="panel">Cargando sucursales...</div>;

  return (
    <section className="branch-page">
      <article className="panel controls-row">
        <div className="control-group search-field">
          <label>Buscar</label>
          <input
            placeholder="Nombre o ID de sucursal"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
        <div className="control-group">
          <label>Ordenar por</label>
          <select value={sortBy} onChange={(event) => setSortBy(event.target.value)}>
            <option value="entries">Entradas</option>
            <option value="occupancy">Ocupación</option>
            <option value="online_ratio">Ratio online</option>
            <option value="name">Nombre</option>
          </select>
        </div>
        <div className="control-group">
          <label>Orden</label>
          <select value={order} onChange={(event) => setOrder(event.target.value)}>
            <option value="desc">Desc</option>
            <option value="asc">Asc</option>
          </select>
        </div>
      </article>

      <div className="branch-summary-strip">
        <span>Sucursales: {formatNumber(data.summary?.branch_count)}</span>
        <span>Entradas hoy: {formatNumber(data.summary?.entries_today)}</span>
        <span>Salidas hoy: {formatNumber(data.summary?.exits_today)}</span>
        <span>Neto: {formatNumber(data.summary?.net_today)}</span>
      </div>

      <article className="panel">
        <div className="panel-header">
          <h3>Comparativo de sucursales</h3>
          <p>Top 10 por entradas/salidas para priorizar operación</p>
        </div>
        <BranchOverviewChart branches={branches} />
      </article>

      <div className="branch-cards-grid">
        {branches.map((branch) => (
          <article className="panel branch-card" key={branch.branch_id}>
            <div className="branch-card-header">
              <div>
                <h3>{branch.branch_name}</h3>
                <small>{branch.branch_id}</small>
              </div>
              <StatusPill online={branch.online_ratio >= 0.7} />
            </div>
            <div className="branch-card-kpis">
              <div>
                <span>Ocupación</span>
                <strong>{formatNumber(branch.current_occupancy)}</strong>
              </div>
              <div>
                <span>Entradas</span>
                <strong>{formatNumber(branch.entries_today)}</strong>
              </div>
              <div>
                <span>Salidas</span>
                <strong>{formatNumber(branch.exits_today)}</strong>
              </div>
            </div>
            <div className="ratio-bar-wrap">
              <div className="ratio-bar" style={{ width: `${Math.round((branch.online_ratio || 0) * 100)}%` }} />
            </div>
            <p className="branch-footnote">
              Cámaras online: {branch.online_cameras}/{branch.total_cameras}
            </p>
            <Link className="branch-link" to={`/branches/${branch.branch_id}`}>
              Ver detalle operativo
            </Link>
          </article>
        ))}
      </div>
    </section>
  );
}

function BranchDetailPage({ user }) {
  const { branchId } = useParams();
  const [hours, setHours] = useState(24);
  const [refreshSeconds, setRefreshSeconds] = useState(30);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const query = toQuery({ hours });
        const payload = await fetchJson(`/api/v1/dashboard/branches/${branchId}?${query}`);
        if (!cancelled) {
          setData(payload);
          setError("");
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    };

    load();
    const intervalId = refreshSeconds > 0 ? setInterval(load, refreshSeconds * 1000) : null;

    return () => {
      cancelled = true;
      if (intervalId) clearInterval(intervalId);
    };
  }, [branchId, hours, refreshSeconds]);

  if (user.role === "branch_manager" && user.branch_id !== branchId) {
    return <Navigate to="/branches" replace />;
  }

  if (error) return <div className="panel error-box">Error cargando detalle: {error}</div>;
  if (!data) return <div className="panel">Cargando detalle...</div>;

  return (
    <section className="branch-detail-page">
      <article className="panel controls-row">
        <Link to="/branches">← Volver</Link>
        <div className="control-group">
          <label>Ventana</label>
          <select value={hours} onChange={(event) => setHours(Number(event.target.value))}>
            <option value={24}>24h</option>
            <option value={48}>48h</option>
            <option value={72}>72h</option>
            <option value={168}>7 días</option>
          </select>
        </div>
        <div className="control-group">
          <label>Auto-refresh</label>
          <select value={refreshSeconds} onChange={(event) => setRefreshSeconds(Number(event.target.value))}>
            <option value={15}>15s</option>
            <option value={30}>30s</option>
            <option value={60}>60s</option>
            <option value={0}>Manual</option>
          </select>
        </div>
        <div className="updated-at">Actualizado: {formatDateTime(data.timestamp)}</div>
      </article>

      <div className="kpi-grid">
        <KpiCard title="Sucursal" value={data.branch_name} helper={data.branch_id} />
        <KpiCard title="Ocupación actual" value={formatNumber(data.current_occupancy)} helper={`Pico: ${formatNumber(data.occupancy_peak)}`} />
        <KpiCard title="Entradas hoy" value={formatNumber(data.entries_today)} helper={`Neto: ${formatNumber(data.net_today)}`} />
        <KpiCard title="Salidas hoy" value={formatNumber(data.exits_today)} helper="Total del día" />
        <KpiCard title="Estado cámaras" value={`${data.online_cameras}/${data.total_cameras}`} helper={`${Math.round((data.online_ratio || 0) * 100)}% online`} />
      </div>

      <article className="panel flow-panel">
        <div className="panel-header">
          <h3>Tendencia de tráfico de la sucursal</h3>
          <p>Flujo horario y ocupación acumulada</p>
        </div>
        <TrafficChart series={data.hourly_flow || []} windowHours={hours} />
      </article>

      <article className="panel">
        <div className="panel-header">
          <h3>Performance por cámara</h3>
          <p>Conteo actual y errores reportados</p>
        </div>
        <CameraLoadChart cameras={data.cameras || []} />
      </article>

      <article className="panel">
        <div className="panel-header">
          <h3>Cámaras</h3>
          <p>Estado y rendimiento operativo</p>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Cámara</th>
                <th>Estado</th>
                <th>FPS</th>
                <th>Errores</th>
                <th>Conteo actual</th>
                <th>Entradas</th>
                <th>Salidas</th>
                <th>Último frame</th>
              </tr>
            </thead>
            <tbody>
              {(data.cameras || []).map((camera) => (
                <tr key={camera.camera_id}>
                  <td>{camera.camera_name || camera.camera_id}</td>
                  <td>
                    <StatusPill online={camera.is_connected} />
                  </td>
                  <td>{Number(camera.fps || 0).toFixed(1)}</td>
                  <td>{formatNumber(camera.error_count || 0)}</td>
                  <td>{formatNumber(camera.current_count || 0)}</td>
                  <td>{formatNumber(camera.entry_today || 0)}</td>
                  <td>{formatNumber(camera.exit_today || 0)}</td>
                  <td>{formatDateTime(camera.last_frame_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>

      <article className="panel alerts-panel">
        <div className="panel-header">
          <h3>Alertas de la sucursal</h3>
          <p>Eventos que requieren acción</p>
        </div>
        <div className="alerts-list">
          {(data.alerts || []).length === 0 ? <p className="empty-state">Sin alertas activas en esta sucursal.</p> : null}
          {(data.alerts || []).map((alert, idx) => (
            <div key={`${idx}-${alert.title}`} className="alert-item">
              <SeverityPill severity={alert.severity} />
              <div>
                <p>{alert.title}</p>
                <small>{alert.description}</small>
              </div>
            </div>
          ))}
        </div>
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
