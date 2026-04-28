import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { fetchJson } from "../services/api";
import { toQuery } from "../utils/query";
import { formatNumber } from "../utils/formatters";
import { StatusPill } from "../components/common/Pills";
import { BranchOverviewChart } from "../components/charts/DashboardCharts";

export function BranchesPage({ user }) {
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
            <option value="occupancy">Ocupacion</option>
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
          <p>Top 10 por entradas/salidas para priorizar operacion</p>
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
                <span>Ocupacion</span>
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
              Camaras online: {branch.online_cameras}/{branch.total_cameras}
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
