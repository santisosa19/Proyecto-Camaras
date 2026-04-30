import { useMemo } from "react";
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
import { CHART_COLORS } from "../../constants/app";
import { branchShortName, chartTickLabel, formatDateTime, formatNumber } from "../../utils/formatters";

function TrafficTooltip({ active, label, payload }) {
  if (!active || !payload?.length) return null;

  const row = payload[0]?.payload;
  if (!row) return null;

  const entry = Number(row.entry || 0);
  const exit = Number(row.exit || 0);
  const occupancy = Number(row.occupancy_end || 0);
  const title = row.tooltipLabel || label || "-";

  return (
    <div className="chart-tooltip">
      <p>{title}</p>
      <div className="tooltip-row">
        <span style={{ color: CHART_COLORS.entry }}>Entradas:</span>
        <strong>{formatNumber(entry)}</strong>
      </div>
      <div className="tooltip-row">
        <span style={{ color: CHART_COLORS.exit }}>Salidas:</span>
        <strong>{formatNumber(exit)}</strong>
      </div>
      <div className="tooltip-row">
        <span style={{ color: CHART_COLORS.occupancy }}>Ocupacion:</span>
        <strong>{formatNumber(occupancy)}</strong>
      </div>
    </div>
  );
}

export function TrafficChart({ series, windowHours }) {
  const data = useMemo(
    () => {
      const rows = series || [];
      if (windowHours >= 168) {
        const byDay = new Map();

        rows.forEach((item) => {
          const dt = new Date(item.hour);
          if (Number.isNaN(dt.getTime())) return;
          const year = dt.getFullYear();
          const month = String(dt.getMonth() + 1).padStart(2, "0");
          const day = String(dt.getDate()).padStart(2, "0");
          const dayKey = `${year}-${month}-${day}`;
          const bucket = byDay.get(dayKey) || {
            dayKey,
            entry: 0,
            exit: 0,
            occupancy_end: 0,
            latestHour: item.hour
          };

          bucket.entry += Number(item.entry || 0);
          bucket.exit += Number(item.exit || 0);

          if (item.hour >= bucket.latestHour) {
            bucket.latestHour = item.hour;
            bucket.occupancy_end = Number(item.occupancy_end || 0);
          }

          byDay.set(dayKey, bucket);
        });

        return Array.from(byDay.values())
          .sort((a, b) => a.dayKey.localeCompare(b.dayKey))
          .map((item) => ({
            ...item,
            chartLabel: new Date(`${item.dayKey}T00:00:00`).toLocaleDateString("es-AR", {
              day: "2-digit",
              month: "2-digit"
            }),
            tooltipLabel: item.dayKey
          }));
      }

      return rows.map((item) => ({
        ...item,
        chartLabel: chartTickLabel(item.hour, windowHours),
        tooltipLabel: formatDateTime(item.hour)
      }));
    },
    [series, windowHours]
  );

  if (!data.length) {
    return <div className="empty-state">Sin datos de flujo para el rango seleccionado.</div>;
  }

  return (
    <div className="chart-wrap" role="img" aria-label="Grafico de entradas, salidas y ocupacion">
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 6, right: 14, bottom: 6, left: 4 }}>
          <CartesianGrid stroke="#e6edf2" strokeDasharray="3 3" />
          <XAxis dataKey="chartLabel" tick={{ fontSize: 11 }} minTickGap={24} />
          <YAxis yAxisId="left" tick={{ fontSize: 11 }} allowDecimals={false} />
          <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} allowDecimals={false} />
          <Tooltip
            shared
            content={<TrafficTooltip />}
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

export function CameraHealthDonut({ health }) {
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
    <div className="donut-wrap" role="img" aria-label="Distribucion del estado de camaras">
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

export function TopBranchesChart({ branches }) {
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
    <div className="chart-wrap" role="img" aria-label="Comparativo de sucursales por entradas y ocupacion">
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

export function BranchOverviewChart({ branches }) {
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
    <div className="chart-wrap" role="img" aria-label="Comparativo de sucursales por entradas y salidas">
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

export function CameraLoadChart({ cameras }) {
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
    <div className="chart-wrap" role="img" aria-label="Performance por camara: conteo actual y errores">
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
