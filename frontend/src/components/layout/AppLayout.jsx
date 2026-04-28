import { Link } from "react-router-dom";
import { ROLE_LABELS } from "../../constants/app";

export function AppLayout({ user, onLogout, children }) {
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
