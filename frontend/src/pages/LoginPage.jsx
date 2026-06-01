import { useState } from "react";
import { useNavigate } from "react-router-dom";

export function LoginPage({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const navigate = useNavigate();

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setIsSubmitting(true);
    try {
      await onLogin(username, password);
      navigate("/");
    } catch (err) {
      setError(err?.message || "Credenciales inválidas");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <h1>Traffic Control Center</h1>
        <p>Monitoreo operativo de sucursales y cámaras</p>
        <label>
          Usuario
          <input placeholder="usuario" value={username} onChange={(event) => setUsername(event.target.value)} />
        </label>
        <label>
          Password
          <input
            placeholder="password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        {error ? <div className="error-box">{error}</div> : null}
        <button type="submit" disabled={isSubmitting}>{isSubmitting ? "Validando..." : "Ingresar"}</button>
        <small>Usá credenciales configuradas en el backend.</small>
      </form>
    </div>
  );
}
