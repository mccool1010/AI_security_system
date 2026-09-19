import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import AuthGate from "./AuthGate.jsx";
import DashboardLayout from "./layout/DashboardLayout";
import { DEMO_MODE } from "./config.js";
import { lazy, Suspense } from "react";
import DashboardPage from "./pages/DashboardPage";
import EventsPage from "./pages/EventsPage";
import SettingsPage from "./pages/SettingsPage";
import CamerasPage from "./pages/CamerasPage";

// Heavy pages load on demand: face-api.js (enrollment) and chart.js (analytics).
const MetricsPage = lazy(() => import("./pages/MetricsPage"));
const RegisterPage = lazy(() => import("./pages/RegisterPage"));
const LandingPage = lazy(() => import("./pages/LandingPage"));

export default function App() {
  return (
    <AuthGate>
    <BrowserRouter>
      <Suspense fallback={<div style={{ padding: 24, color: "var(--text-muted)" }}>Loading…</div>}>
      <Routes>
        {/* The portfolio landing page only exists in the demo build; real installs open the dashboard. */}
        <Route
          path="/"
          element={DEMO_MODE ? <LandingPage /> : <Navigate to="/dashboard" replace />}
        />

        {/* Dashboard routes — wrapped in sidebar layout */}
        <Route
          path="/dashboard"
          element={
            <DashboardLayout>
              <DashboardPage />
            </DashboardLayout>
          }
        />
        <Route
          path="/events"
          element={
            <DashboardLayout>
              <EventsPage />
            </DashboardLayout>
          }
        />
        <Route
          path="/metrics"
          element={
            <DashboardLayout>
              <MetricsPage />
            </DashboardLayout>
          }
        />
        <Route
          path="/settings"
          element={
            <DashboardLayout>
              <SettingsPage />
            </DashboardLayout>
          }
        />
        <Route
          path="/register"
          element={
            <DashboardLayout>
              <RegisterPage />
            </DashboardLayout>
          }
        />
        <Route
          path="/cameras"
          element={
            <DashboardLayout>
              <CamerasPage />
            </DashboardLayout>
          }
        />
      </Routes>
      </Suspense>
    </BrowserRouter>
    </AuthGate>
  );
}
