import { Link, useLocation } from "react-router-dom";

export default function DashboardLayout({ children }) {
  const location = useLocation();
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

        {/* System status footer */}
        <div style={{
          padding: "14px 16px",
          borderTop: "1px solid var(--border-subtle)",
          fontSize: 10,
          color: "var(--text-muted)",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%",
              background: "var(--success)",
              boxShadow: "0 0 6px rgba(16,185,129,0.5)",
            }} />
            <span style={{ color: "var(--text-secondary)", fontWeight: 500 }}>System Online</span>
          </div>
          <div>YOLOv8-Pose + SlowFast R50</div>
        </div>
      </aside>

      {/* Main Content */}
      <main style={{
        flex: 1,
        overflow: "auto",
        padding: "20px 24px",
        background: "var(--bg-primary)",
      }}>
        {children}
      </main>
    </div>
  );
}
