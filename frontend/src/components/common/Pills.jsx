import { severityLabel } from "../../utils/formatters";

export function StatusPill({ online }) {
  return <span className={`status-pill ${online ? "online" : "offline"}`}>{online ? "Online" : "Offline"}</span>;
}

export function SeverityPill({ severity }) {
  return <span className={`severity-pill ${severity || "low"}`}>{severityLabel(severity)}</span>;
}
