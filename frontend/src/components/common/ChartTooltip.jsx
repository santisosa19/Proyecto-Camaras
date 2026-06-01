import { formatNumber } from "../../utils/formatters";

export function ChartTooltip({ active, label, payload }) {
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
