import { useEffect, useState } from "react";
import { fetchJson } from "../services/api";
import { defaultLastDays, normalizeDateRange, rangeWindowHours } from "../utils/dateRange";
import { toQuery } from "../utils/query";
import { formatDateTime, formatNumber } from "../utils/formatters";
import { KpiCard } from "../components/common/KpiCard";
import { SeverityPill } from "../components/common/Pills";
import { CameraHealthDonut, TopBranchesChart, TrafficChart } from "../components/charts/DashboardCharts";
import { DateRangeFilter } from "../components/filters/DateRangeFilter";

export function OverviewPage() {
  const [dateRange, setDateRange] = useState(() => defaultLastDays(7));
  const [appliedRange, setAppliedRange] = useState(() => defaultLastDays(7));
  const [refreshSeconds, setRefreshSeconds] = useState(30);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const normalizedRange = normalizeDateRange(dateRange.startDate, dateRange.endDate);
        if (!normalizedRange.startDate || !normalizedRange.endDate) {
          return;
        }
        const query = toQuery({
          start_date: normalizedRange.startDate,
          end_date: normalizedRange.endDate
        });
        console.debug("[overview] fetching range", normalizedRange.startDate, normalizedRange.endDate);
        const payload = await fetchJson(`/api/v1/dashboard/overview?${query}`);
        if (!cancelled) {
          setData(payload);
          setAppliedRange({
            startDate: payload?.start_date || normalizedRange.startDate,
            endDate: payload?.end_date || normalizedRange.endDate
          });
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
  }, [dateRange.startDate, dateRange.endDate, refreshSeconds]);

  if (loading) return <div className="panel">Cargando overview...</div>;
  if (error) return <div className="panel error-box">Error cargando overview: {error}</div>;

  const onlineRatio = data.total_cameras ? Math.round((100 * data.online_cameras) / data.total_cameras) : 0;
  const normalizedRange = normalizeDateRange(dateRange.startDate, dateRange.endDate);
  const chartWindowHours = rangeWindowHours(normalizedRange.startDate, normalizedRange.endDate);

  return (
    <section className="dashboard-grid">
      <article className="panel controls-row">
        <div className="control-group">
          <label>Rango de fechas</label>
          <DateRangeFilter
            startDate={dateRange.startDate}
            endDate={dateRange.endDate}
            onChange={(nextRange) => setDateRange(nextRange)}
          />
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
        <div className="updated-at">Rango aplicado: {appliedRange.startDate} a {appliedRange.endDate}</div>
      </article>

      <div className="kpi-grid">
        <KpiCard title="Ocupacion actual" value={formatNumber(data.current_occupancy)} helper="Total en todas las sucursales" />
        <KpiCard title="Entradas periodo" value={formatNumber(data.entries_today)} helper={`Neto: ${formatNumber(data.net_today)}`} />
        <KpiCard title="Salidas periodo" value={formatNumber(data.exits_today)} helper="En rango seleccionado" />
        <KpiCard title="Sucursales" value={formatNumber(data.total_branches)} helper="Con actividad reportada" />
        <KpiCard
          title="Estado de camaras"
          value={`${formatNumber(data.online_cameras)}/${formatNumber(data.total_cameras)}`}
          helper={`${onlineRatio}% online`}
        />
        <KpiCard title="Alertas" value={formatNumber((data.alerts || []).length)} helper="Prioridad operativa" />
      </div>

      <article className="panel flow-panel">
        <div className="panel-header">
          <h3>Tendencia de trafico y ocupacion</h3>
          <p>
            Entradas, salidas y ocupacion acumulada por hora ({normalizedRange.startDate} a {normalizedRange.endDate})
          </p>
        </div>
        <TrafficChart series={data.flow_series || []} windowHours={chartWindowHours} />
      </article>

      <div className="charts-grid two-cols">
        <article className="panel">
          <div className="panel-header">
            <h3>Salud de camaras</h3>
            <p>Distribucion de estado operativo</p>
          </div>
          <CameraHealthDonut health={data.camera_health} />
        </article>

        <article className="panel">
          <div className="panel-header">
            <h3>Top sucursales</h3>
            <p>Comparativo de entradas y ocupacion</p>
          </div>
          <TopBranchesChart branches={data.top_branches || []} />
        </article>
      </div>

      <article className="panel alerts-panel">
        <div className="panel-header">
          <h3>Alertas operativas</h3>
          <p>Prioridad para intervencion</p>
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
