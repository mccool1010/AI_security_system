import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useBackendStatus } from "../hooks/useLiveData.js";

// Events written before pipeline 2.0 stored {name: "unknown", score: 0} when no face was visible.
const realFaces = (ev) => (ev?.faces || []).filter((f) => !(f.name === "unknown" && !f.score));

export default function EventsPage() {
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState(null);
  const { perf } = useBackendStatus(10000);

  useEffect(() => {
    const fetchEvents = async () => {
      try {
        const res = await fetch("/api/events?limit=100");
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
        const res = await fetch("/api/status");
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
    const faces = realFaces(ev);
    for (const f of faces) {
      parts.push(f.name !== "unknown"
        ? `Recognized: ${f.name} (${(f.score * 100).toFixed(0)}% similarity)`
        : "Unrecognised face");
    }
    if (!faces.length && ev.face_status === "no_face_visible") parts.push("Face not visible");
    if (ev.trigger && ev.trigger !== "activity") parts.push(`Trigger: ${ev.trigger.replace(/_/g, " ")}`);

    if (ev.action) parts.push(ev.action);
    if (ev.height_m) {
      const spread = ev.height_spread_m != null ? ` ±${Math.round(ev.height_spread_m * 100)} cm` : "";
      parts.push(`Height: ${Number(ev.height_m).toFixed(2)} m${spread}`);
    }
    for (const r of ev.risk_reasons || []) parts.push(r);
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
              const known = realFaces(ev).filter((f) => f.name !== "unknown").map((f) => f.name);
              const count = ev?.persons?.length ?? ev?.detections?.length ?? 0;
              const label = known.length ? known.join(", ") : count > 1 ? `${count} people` : "Person";
              const det = ev?.detections?.[0];
              const conf = det?.confidence != null ? `detection ${(det.confidence * 100).toFixed(0)}%` : "";
              const descLines = buildDescription(ev);

              return (
                <div key={`${ev.source}-${ev.unix_ts ?? i}`} className="card" style={{ display: "flex", gap: 12, padding: 14 }}>
                  {/* Screenshot */}
                  {ev.clip_status === "ready" && ev.clip_path ? (
                    <video
                      className="event-clip"
                      src={`/api/clips/${ev.source}/${ev.clip_path}`}
                      poster={ev.screenshot_path ? `/api/screenshots/${ev.screenshot_path}` : undefined}
                      controls
                      preload="none"
                      muted
                      playsInline
                    />
                  ) : ev.screenshot_path ? (
                    <img
                      src={`/api/screenshots/${ev.screenshot_path}`}
                      alt="event"
                      title={ev.clip_status === "recording" ? "clip still recording" : undefined}
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
              { name: "Compute", value: perf ? (perf.device === "cuda" ? "GPU" : "CPU") : "—", ok: !!perf },
              { name: "Pose", value: "YOLOv8n-Pose", ok: !!perf },
              { name: "Activity AI", value: perf?.activity?.ready ? "SlowFast R50" : perf?.activity?.enabled === false ? "Disabled" : "Loading…", ok: perf?.activity?.ready },
              { name: "Faces", value: perf?.face?.available ? perf.face.backend : "Disabled", ok: perf?.face?.available },
              { name: "Depth", value: perf?.depth?.ready ? "MiDaS (relative)" : "Off", ok: perf?.depth?.ready },
              { name: "Tracking", value: "IoU + distance", ok: !!perf },
            ].map((item) => (
              <div key={item.name} className="status-row">
                <span className="status-label">{item.name}</span>
                <span className="status-value" style={{ color: item.ok ? "var(--success)" : "var(--text-muted)", fontSize: 11 }}>
                  {item.value}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Quick link */}
        <Link
          to="/settings"
          className="btn-secondary"
          style={{ justifyContent: "center", textDecoration: "none" }}
        >
          ⚙ Settings
        </Link>
      </div>
    </div>
  );
}
