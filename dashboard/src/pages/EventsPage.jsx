import { useEffect, useState } from "react";

export default function EventsPage() {
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState(null);

  useEffect(() => {
    const fetchEvents = async () => {
      try {
        const res = await fetch("/events");
        const data = await res.json();
        setEvents(data);
      } catch (err) {
        console.error("Failed to fetch /events", err);
      }
    };
    fetchEvents();
    const id = setInterval(fetchEvents, 3000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch("/status");
        const data = await res.json();
        setStatus(data);
      } catch (err) {
        console.error("Failed to fetch /status", err);
      }
    };
    fetchStatus();
    const id = setInterval(fetchStatus, 5000);
    return () => clearInterval(id);
  }, []);

  const fmt = (ts) => {
    if (!ts) return "—";
    try { return new Date(ts).toLocaleString(); }
    catch { return String(ts); }
  };

  const riskClass = (level) => {
    if (level === "HIGH") return "badge-danger";
    if (level === "MEDIUM") return "badge-warning";
    return "badge-success";
  };

  const buildDescription = (ev) => {
    const parts = [];
    const face = ev?.faces?.[0];
    const det = ev?.detections?.[0];
    const personName = face?.name && face.name !== "unknown" ? face.name : null;

    if (personName) {
      parts.push(`Recognized: ${personName} (${(face.score * 100).toFixed(1)}%)`);
    } else if (det) {
      parts.push(`Unknown person (${(det.confidence * 100).toFixed(1)}% conf)`);
    }

    if (ev.action) parts.push(ev.action);
    if (ev.height_m) parts.push(`Height: ${ev.height_m}m`);
    return parts;
  };

  return (
    <div className="page-enter" style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: 16 }}>
      {/* Main event list */}
      <div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
          <h2 style={{ fontSize: 18, fontWeight: 700, letterSpacing: -0.3 }}>Event Log</h2>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {events.length} event{events.length !== 1 ? "s" : ""}
          </span>
        </div>

        {events.length === 0 ? (
          <div className="card" style={{ padding: 40, textAlign: "center" }}>
            <div style={{ fontSize: 24, marginBottom: 8, opacity: 0.4 }}>⚡</div>
            <p style={{ fontSize: 12, color: "var(--text-muted)" }}>No events recorded yet</p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {events.map((ev, i) => {
              const face = ev?.faces?.[0];
              const det = ev?.detections?.[0];
              const personName = face?.name && face.name !== "unknown" ? face.name : null;
              const label = personName || det?.class || "Unknown";
              const conf = personName && face?.score != null
                ? (face.score * 100).toFixed(1) + "%"
                : det?.confidence != null
                  ? (det.confidence * 100).toFixed(1) + "%"
                  : "";
              const descLines = buildDescription(ev);

              return (
                <div key={i} className="card" style={{ display: "flex", gap: 12, padding: 14 }}>
                  {/* Screenshot */}
                  {ev.screenshot_path ? (
                    <img
                      src={`/api/screenshots/${ev.screenshot_path}`}
                      alt="event"
                      style={{
                        width: 80, height: 56, objectFit: "cover",
                        borderRadius: 8, flexShrink: 0,
                        border: "1px solid var(--border-subtle)",
                      }}
                    />
                  ) : (
                    <div style={{
                      width: 80, height: 56, borderRadius: 8,
                      background: "var(--bg-primary)", display: "flex",
                      alignItems: "center", justifyContent: "center",
                      fontSize: 10, color: "var(--text-muted)", flexShrink: 0,
                      border: "1px solid var(--border-subtle)",
                    }}>
                      No image
                    </div>
                  )}

                  {/* Content */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>
                        {label}
                      </span>
                      {conf && (
                        <span style={{ fontSize: 11, color: "var(--accent)" }}>{conf}</span>
                      )}
                      {ev.risk_level && (
                        <span className={`badge ${riskClass(ev.risk_level)}`}>
                          {ev.risk_level}
                        </span>
                      )}
                    </div>

                    {descLines.length > 0 && (
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 12px", marginBottom: 4 }}>
                        {descLines.map((line, j) => (
                          <span key={j} style={{ fontSize: 11, color: "var(--text-muted)" }}>
                            {line}
                          </span>
                        ))}
                      </div>
                    )}

                    <div style={{
                      display: "flex", gap: 12, fontSize: 10,
                      color: "var(--text-muted)", marginTop: 2,
                    }}>
                      <span>{fmt(ev.timestamp)}</span>
                      {ev.camera_name && <span>📷 {ev.camera_name}</span>}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Sidebar */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12, position: "sticky", top: 20 }}>
        {/* Backend Status */}
        <div className="card">
          <div className="card-header">
            <h3>Backend Status</h3>
          </div>
          <div className="card-body" style={{ padding: "10px 16px" }}>
            {!status ? (
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Loading…</div>
            ) : (
              <>
                <div className="status-row">
                  <span className="status-label">Service</span>
                  <span className="status-value" style={{ color: status.service === "ok" ? "var(--success)" : "var(--danger)" }}>
                    {status.service === "ok" ? "Online" : "Down"}
                  </span>
                </div>
                <div className="status-row">
                  <span className="status-label">MongoDB</span>
                  <span className="status-value" style={{ color: status.mongo?.ok ? "var(--success)" : "var(--danger)" }}>
                    {status.mongo?.ok ? "Connected" : "Down"}
                  </span>
                </div>
                <div className="status-row">
                  <span className="status-label">Camera</span>
                  <span className="status-value" style={{ color: status.camera_open ? "var(--success)" : "var(--danger)" }}>
                    {status.camera_open ? "Active" : "Closed"}
                  </span>
                </div>
              </>
            )}
          </div>
        </div>

        {/* ML Pipeline */}
        <div className="card">
          <div className="card-header">
            <h3>ML Pipeline</h3>
          </div>
          <div className="card-body" style={{ padding: "10px 16px" }}>
            {[
              { name: "Object Detection", value: "YOLOv8-Pose" },
              { name: "Activity AI", value: "SlowFast R50" },
              { name: "Face Recognition", value: "DeepFace" },
              { name: "Tracking", value: "IoU Tracker" },
              { name: "Risk Scoring", value: "Hybrid Engine" },
              { name: "Height", value: "Auto-calibrated" },
            ].map((item, i) => (
              <div key={i} className="status-row">
                <span className="status-label">{item.name}</span>
                <span className="status-value" style={{ color: "var(--success)", fontSize: 11 }}>
                  {item.value}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Quick link */}
        <a
          href="/settings"
          className="btn-secondary"
          style={{ justifyContent: "center", textDecoration: "none" }}
        >
          ⚙ Settings
        </a>
      </div>
    </div>
  );
}
