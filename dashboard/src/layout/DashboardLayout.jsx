import { Link, useLocation } from "react-router-dom";
import { DEMO_MODE } from "../config.js";
import { useBackendStatus } from "../hooks/useLiveData.js";

export default function DashboardLayout({ children }) {
  const location = useLocation();
  const { online, mongo, perf } = useBackendStatus();
  const statusText = DEMO_MODE ? "Demo data"
    : online === null ? "Connecting…"
      : !online ? "Backend offline"
        : mongo === false ? "Database offline"
          : "System online";
  const dotColor = DEMO_MODE ? "var(--warning)"
    : online && mongo !== false ? "var(--success)"
      : online === null ? "var(--text-muted)" : "var(--danger)";
  const navItems = [
    { to: "/dashboard", label: "Dashboard", icon: "⬡" },
    { to: "/cameras", label: "Cameras", icon: "◉" },
    { to: "/events", label: "Events", icon: "⚡" },
    { to: "/metrics", label: "Analytics", icon: "◈" },
    { to: "/settings", label: "Settings", icon: "⚙" },
    { to: "/register", label: "Enroll Face", icon: "◎" },
  ];

  return (
    <div className="flex h-screen" style={{ background: "var(--bg-primary)" }}>
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>
            <span className="brand-dot" />
            <span>SecureVision</span>
          </h1>
          <p style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4, paddingLeft: 18 }}>
            AI Surveillance Platform
          </p>
        </div>

        <nav style={{ padding: "12px 0", flex: 1 }}>
          {navItems.map((item) => {
            const active = location.pathname === item.to;
            return (
              <Link
                key={item.to}
                to={item.to}
                className={`nav-item ${active ? "active" : ""}`}
              >
                <span className="nav-icon">{item.icon}</span>
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* System status footer — reflects the real backend */}
        <div style={{
          padding: "14px 16px",
          borderTop: "1px solid var(--border-subtle)",
          fontSize: 10,
          color: "var(--text-muted)",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%",
              background: dotColor,
              boxShadow: `0 0 6px ${dotColor}`,
            }} />
            <span style={{ color: "var(--text-secondary)", fontWeight: 500 }}>{statusText}</span>
          </div>
          {perf && (
            <div>
              {perf.device === "cuda" ? `GPU · ${perf.gpu}` : "CPU"}
              {perf.pipeline_fps ? ` · ${perf.pipeline_fps} fps` : ""}
            </div>
          )}
          {perf?.face && !perf.face.available && (
            <div style={{ color: "var(--warning)" }}>Face recognition off</div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main style={{
        flex: 1,
        overflow: "auto",
        padding: "20px 24px",
        background: "var(--bg-primary)",
      }}>
        {DEMO_MODE && (
          <div className="demo-banner">
            Demo mode — simulated data, no backend connected
          </div>
        )}
        {children}
      </main>
    </div>
  );
}
