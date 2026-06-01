import { useEffect, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import { CameraLoadChart, TrafficChart } from "../components/charts/DashboardCharts";
import { KpiCard } from "../components/common/KpiCard";
import { SeverityPill, StatusPill } from "../components/common/Pills";
import { HeatmapCameraGrid } from "../components/heatmaps/HeatmapCameraGrid";
import { DateRangeFilter } from "../components/filters/DateRangeFilter";
import { fetchJson } from "../services/api";
import { defaultLastDays, normalizeDateRange, rangeWindowHours } from "../utils/dateRange";
import { formatDateTime, formatNumber, formatSlotLabel } from "../utils/formatters";
import { toQuery } from "../utils/query";

export function BranchDetailPage({ user }) {
  const { branchId } = useParams();
  const [dateRange, setDateRange] = useState(() => defaultLastDays(1));
  const [appliedRange, setAppliedRange] = useState(() => defaultLastDays(1));
  const [refreshSeconds, setRefreshSeconds] = useState(30);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [heatmapData, setHeatmapData] = useState(null);
  const [heatmapError, setHeatmapError] = useState("");
  const [selectedHeatmapSlot, setSelectedHeatmapSlot] = useState("");
  const [overlayOpacity, setOverlayOpacity] = useState(0.58);

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
        console.debug("[branch-detail] fetching range", normalizedRange.startDate, normalizedRange.endDate);
        const payload = await fetchJson(`/api/v1/dashboard/branches/${branchId}?${query}`);
        if (!cancelled) {
          setData(payload);
          setAppliedRange({
            startDate: payload?.start_date || normalizedRange.startDate,
            endDate: payload?.end_date || normalizedRange.endDate
          });
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
  }, [branchId, dateRange.startDate, dateRange.endDate, refreshSeconds]);

  useEffect(() => {
    let cancelled = false;

    const loadHeatmaps = async () => {
      try {
        let targetDate;
        let targetHour;
        if (selectedHeatmapSlot) {
          const [slotDate, slotHour] = selectedHeatmapSlot.split("|");
          targetDate = slotDate;
          const parsedHour = Number(slotHour);
          if (!Number.isNaN(parsedHour)) targetHour = parsedHour;
        }

        const query = toQuery({ target_date: targetDate, hour: targetHour });
        const payload = await fetchJson(`/api/v1/dashboard/branches/${branchId}/heatmaps?${query}`);
        if (!cancelled) {
          setHeatmapData(payload);
          setHeatmapError("");
          const selected = payload?.selected_slot;
          if (selected?.date !== undefined && selected?.hour !== undefined) {
            const canonical = `${selected.date}|${selected.hour}`;
            if (canonical !== selectedHeatmapSlot) {
              setSelectedHeatmapSlot(canonical);
            }
          }
        }
      } catch (err) {
        if (!cancelled) setHeatmapError(err.message);
      }
    };

    loadHeatmaps();
    const intervalId = refreshSeconds > 0 ? setInterval(loadHeatmaps, refreshSeconds * 1000) : null;

    return () => {
      cancelled = true;
      if (intervalId) clearInterval(intervalId);
    };
  }, [branchId, selectedHeatmapSlot, refreshSeconds]);

  if (user.role === "branch_manager" && user.branch_id !== branchId) {
    return <Navigate to="/branches" replace />;
  }

  if (error) return <div className="panel error-box">Error cargando detalle: {error}</div>;
  if (!data) return <div className="panel">Cargando detalle...</div>;
  const normalizedRange = normalizeDateRange(dateRange.startDate, dateRange.endDate);
  const chartWindowHours = rangeWindowHours(normalizedRange.startDate, normalizedRange.endDate);
  const chartGranularityLabel = chartWindowHours >= 168 ? "por dia" : "por hora";

  return (
    <section className="branch-detail-page">
      <article className="panel controls-row">
        <Link to="/branches">Volver</Link>
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
        <div className="updated-at" aria-live="polite">Rango aplicado: {appliedRange.startDate} a {appliedRange.endDate}</div>
      </article>

      <div className="kpi-grid">
        <KpiCard title="Sucursal" value={data.branch_name} helper={data.branch_id} />
        <KpiCard title="Ocupacion actual" value={formatNumber(data.current_occupancy)} helper={`Pico: ${formatNumber(data.occupancy_peak)}`} />
        <KpiCard title="Entradas periodo" value={formatNumber(data.entries_today)} helper={`Neto: ${formatNumber(data.net_today)}`} />
        <KpiCard title="Salidas periodo" value={formatNumber(data.exits_today)} helper="En rango seleccionado" />
        <KpiCard title="Estado camaras" value={`${data.online_cameras}/${data.total_cameras}`} helper={`${Math.round((data.online_ratio || 0) * 100)}% online`} />
      </div>

      <article className="panel flow-panel">
        <div className="panel-header">
          <h3>Tendencia de trafico de la sucursal</h3>
          <p>
            Flujo y ocupacion acumulada {chartGranularityLabel} ({normalizedRange.startDate} a {normalizedRange.endDate})
          </p>
        </div>
        <TrafficChart series={data.hourly_flow || []} windowHours={chartWindowHours} />
      </article>

      <article className="panel">
        <div className="panel-header">
          <h3>Performance por camara</h3>
          <p>Conteo actual por camara</p>
        </div>
        <CameraLoadChart cameras={data.cameras || []} />
      </article>

      <article className="panel">
        <div className="panel-header">
          <h3>Mapa de calor por hora</h3>
          <p>Overlay por camara para la hora seleccionada</p>
        </div>

        <div className="controls-row heatmap-controls">
          <div className="control-group">
            <label>Hora</label>
            <select value={selectedHeatmapSlot} onChange={(event) => setSelectedHeatmapSlot(event.target.value)}>
              {(heatmapData?.available_slots || []).length === 0 ? (
                <option value="">Sin slots disponibles</option>
              ) : null}
              {(heatmapData?.available_slots || []).map((slot) => {
                const value = `${slot.date}|${slot.hour}`;
                return (
                  <option key={value} value={value}>
                    {formatSlotLabel(slot)} ({slot.camera_count} cams)
                  </option>
                );
              })}
            </select>
          </div>

          <div className="control-group">
            <label>Opacidad overlay ({Math.round(overlayOpacity * 100)}%)</label>
            <input
              type="range"
              min="0.15"
              max="0.95"
              step="0.05"
              value={overlayOpacity}
              onChange={(event) => setOverlayOpacity(Number(event.target.value))}
            />
          </div>

          <div className="updated-at">
            Slot actual: {heatmapData?.selected_slot ? formatSlotLabel(heatmapData.selected_slot) : "Sin datos"}
          </div>
        </div>

        {heatmapError ? <div className="error-box">Error cargando heatmaps: {heatmapError}</div> : null}

        <HeatmapCameraGrid cameras={heatmapData?.cameras || []} overlayOpacity={overlayOpacity} />
      </article>

      <article className="panel">
        <div className="panel-header">
          <h3>Camaras</h3>
          <p>Estado y rendimiento operativo</p>
        </div>
        <div className="table-wrap">
          <table>
            <caption className="sr-only">Tabla de camaras con estado, FPS, conteo, entradas y salidas</caption>
            <thead>
              <tr>
                <th scope="col">Camara</th>
                <th scope="col">Estado</th>
                <th scope="col">FPS</th>
                <th scope="col">Conteo actual</th>
                <th scope="col">Entradas</th>
                <th scope="col">Salidas</th>
                <th scope="col">Ultimo frame</th>
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
          <p>Eventos que requieren accion</p>
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
