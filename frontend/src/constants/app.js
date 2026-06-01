export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const AUTH_STORAGE_KEY = "traffic_user";

export const ROLE_LABELS = {
  superadmin: "Superadmin",
  manager: "Manager",
  branch_manager: "Sucursal"
};

export const CHART_COLORS = {
  entry: "#17a672",
  exit: "#df7c2a",
  occupancy: "#1459c7",
  online: "#1b8f5a",
  offline: "#cc3d2f",
  high_error: "#da8f19"
};
