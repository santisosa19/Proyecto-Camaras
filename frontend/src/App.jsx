import React from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppErrorBoundary } from "./components/AppErrorBoundary";
import { AppLayout } from "./components/layout/AppLayout";
import { useAuth } from "./hooks/useAuth";
import { BranchDetailPage } from "./pages/BranchDetailPage";
import { BranchesPage } from "./pages/BranchesPage";
import { LoginPage } from "./pages/LoginPage";
import { OverviewPage } from "./pages/OverviewPage";

function PrivateApp({ user, logout }) {
  return (
    <AppLayout user={user} onLogout={logout}>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/branches" element={<BranchesPage user={user} />} />
        <Route path="/branches/:branchId" element={<BranchDetailPage user={user} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppLayout>
  );
}

function AppContent() {
  const { user, login, logout } = useAuth();

  if (!user?.token) {
    return <LoginPage onLogin={login} />;
  }

  return <PrivateApp user={user} logout={logout} />;
}

export default function App() {
  return (
    <AppErrorBoundary>
      <AppContent />
    </AppErrorBoundary>
  );
}
