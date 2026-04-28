import { useState } from "react";
import { clearStoredUser, parseStoredUser, saveStoredUser } from "../services/authStorage";
import { loginRequest } from "../services/api";

export function useAuth() {
  const [user, setUser] = useState(() => parseStoredUser());

  const login = async (username, password) => {
    const payload = await loginRequest(username, password);
    const safeUser = {
      username: payload?.user?.username || username,
      role: payload?.user?.role || "branch_manager",
      branch_id: payload?.user?.branch_id || null,
      token: payload?.access_token || ""
    };
    saveStoredUser(safeUser);
    setUser(safeUser);
    return true;
  };

  const logout = () => {
    clearStoredUser();
    setUser(null);
  };

  return { user, login, logout };
}
