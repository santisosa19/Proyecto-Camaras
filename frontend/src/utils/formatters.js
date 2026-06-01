export function formatNumber(value) {
  return new Intl.NumberFormat("es-AR").format(Number(value || 0));
}

export function formatDateTime(value) {
  if (!value) return "-";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return "-";
  return dt.toLocaleString("es-AR", { hour12: false });
}

export function branchShortName(name, max = 16) {
  if (!name) return "-";
  return name.length > max ? `${name.slice(0, max)}…` : name;
}

export function chartTickLabel(isoDate, windowHours) {
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

export function formatSlotLabel(slot) {
  if (!slot?.date || slot?.hour === undefined || slot?.hour === null) return "-";
  const hour = String(slot.hour).padStart(2, "0");
  return `${slot.date} ${hour}:00`;
}

export function genderTitle(kind, gender) {
  const prefix = kind === "entry" ? "Entradas" : "Salidas";
  if (gender === "male") return `${prefix} - Masculino`;
  if (gender === "female") return `${prefix} - Femenino`;
  return `${prefix} - Sin clasificar`;
}

export function severityLabel(severity) {
  if (severity === "high") return "Alta";
  if (severity === "medium") return "Media";
  return "Baja";
}
