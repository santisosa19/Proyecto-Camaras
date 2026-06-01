import { API_BASE_URL } from "../constants/app";
import { parseStoredUser } from "./authStorage";

export async function loginRequest(username, password) {
  const res = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ username, password })
  });

  let payload = null;
  try {
    payload = await res.json();
  } catch (_err) {
    payload = null;
  }

  if (!res.ok) {
    const message = payload?.detail || `HTTP ${res.status}`;
    throw new Error(message);
  }

  return payload;
}

export async function fetchJson(path) {
  const storedUser = parseStoredUser();
  const headers = {};
  if (storedUser?.token) {
    headers.Authorization = `Bearer ${storedUser.token}`;
  }

  const res = await fetch(`${API_BASE_URL}${path}`, { headers });
  if (!res.ok) {
    let detail = "";
    try {
      const payload = await res.json();
      detail = payload?.detail || "";
    } catch (_err) {
      detail = "";
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }

  return res.json();
}
