import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import DashboardLayout from "./layout/DashboardLayout";
import DashboardPage from "./pages/DashboardPage";
import EventsPage from "./pages/EventsPage";
import MetricsPage from "./pages/MetricsPage";
import SettingsPage from "./pages/SettingsPage";
import RegisterPage from "./pages/RegisterPage";
import CamerasPage from "./pages/CamerasPage";
import LandingPage from "./pages/LandingPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* If running locally via run.bat (dev mode), bypass landing page. Otherwise show portfolio. */}
        <Route 
          path="/" 
          element={import.meta.env.DEV ? <Navigate to="/dashboard" replace /> : <LandingPage />} 
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
    </BrowserRouter>
  );
}
