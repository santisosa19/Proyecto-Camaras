import React from "react";
import { clearStoredUser } from "../services/authStorage";

export class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Error de render en dashboard:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="login-wrap">
          <div className="login-card">
            <h1>Error de interfaz</h1>
            <p>La página falló al renderizar. Probá refrescar y limpiar sesión local.</p>
            <button
              type="button"
              onClick={() => {
                clearStoredUser();
                window.location.reload();
              }}
            >
              Limpiar sesión y recargar
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
