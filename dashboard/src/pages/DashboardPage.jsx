import { useEffect, useState, useCallback, useRef } from "react";
import { DEMO_MODE } from "../config.js";
import { EMPTY_SNAPSHOT, titleCase, useBackendStatus, useLiveData } from "../hooks/useLiveData.js";

// ── Layout modes ──────────────────────────────────────────────────
const LAYOUT = { SPOTLIGHT: "spotlight", GRID: "grid", LIST: "list" };

// ── Detection Info Panel (OUTSIDE component to avoid remount flicker) ──
function DetectionPanel({ detections, riskClass, transport }) {
  const { persons = [], activities = [], risk_level, risk_reasons = [], person_count = 0, model_ready } = detections;
  const topActivity = activities.length > 0 ? activities[0] : null;
  const person = persons.length > 0 ? persons[0] : null;
  const identity = person?.identity;

  return (
    <div className="detection-panel">
      <div className="detection-panel-header">
        <h3>Live Detection</h3>
        <span
          title={transport === "stream" ? "Live push (SSE)" : "Polling"}
          className={`detection-status-dot ${person_count > 0 ? "active" : ""}`}
        />
      </div>

      <div className="detection-panel-body">
        <div className="det-row">
          <span className="det-icon">👤</span>
          <span className="det-label">Persons</span>
          <span className="det-value">{person_count}</span>
        </div>

        {/* Activity (SlowFast) — only what the model actually reported */}
        <div className="det-row">
          <span className="det-icon">🎯</span>
          <span className="det-label">Activity</span>
          <span className="det-value det-activity">
            {topActivity ? topActivity.action : model_ready ? "None detected" : "Model loading…"}
          </span>
        </div>
        {topActivity && (
          <div className="det-bar-container">
            <div className="det-bar" style={{ width: `${Math.min(100, topActivity.confidence * 100)}%` }} />
            <span className="det-bar-label">
              {(topActivity.confidence * 100).toFixed(0)}% · {topActivity.severity}
            </span>
          </div>
        )}
        {activities.length > 1 && (
          <div className="det-secondary-activities">
            {activities.slice(1).map((a) => (
              <div key={a.category} className="det-secondary-row">
                <span className="det-secondary-name">{a.action}</span>
                <span className="det-secondary-conf">{(a.confidence * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        )}

        <div className="det-divider" />

        <div className="det-row">
          <span className="det-icon">🧍</span>
          <span className="det-label">Posture</span>
          <span className="det-value">{person ? titleCase(person.pose) : "—"}</span>
        </div>

        <div className="det-row">
          <span className="det-icon">⏱</span>
          <span className="det-label">In view</span>
          <span className="det-value">
            {person ? `${Math.round(person.dwell_s)}s · ${person.dwell}` : "—"}
          </span>
        </div>

        <div className="det-row">
          <span className="det-icon">🪪</span>
          <span className="det-label">Identity</span>
          <span className="det-value">
            {!person ? "—" : !identity ? "No face yet"
              : identity.name === "unknown" ? "Unrecognised"
                : `${identity.name} (${(identity.score * 100).toFixed(0)}%)`}
          </span>
        </div>

        <div className="det-row">
          <span className="det-icon">📏</span>
          <span className="det-label">Height</span>
          <span className="det-value">
            {person?.height_m ? (
              <>
                {person.height_m.toFixed(2)} m
                {person.height_spread_m != null && (
                  <span className="det-margin"> ±{(person.height_spread_m * 100).toFixed(0)} cm</span>
                )}
              </>
            ) : "—"}
          </span>
        </div>
        {person && !person.height_m && person.height_status && (
          <div className="det-note">{titleCase(person.height_status)}</div>
        )}
        {person?.height_m && (
          <div className="det-note">
            median of {person.height_samples} measurements
            {person.distance_m ? ` · ${person.distance_m} m away` : ""}
          </div>
        )}

        <div className="det-divider" />

        <div className="det-row">
          <span className="det-icon">⚡</span>
          <span className="det-label">Risk</span>
          <span className={`badge ${riskClass(risk_level)}`} style={{ fontSize: 10, padding: "2px 8px" }}>
            {risk_level}
          </span>
        </div>
        {risk_reasons.length > 0 && (
          <div className="det-reasons">
            {risk_reasons.map((r, i) => <span key={i} className="det-reason">{r}</span>)}
          </div>
        )}

        {person && (
          <div className="det-row">
            <span className="det-icon">🔍</span>
            <span className="det-label">Confidence</span>
            <span className="det-value">{(person.confidence * 100).toFixed(0)}%</span>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Video feed with demo fallback ──
const DEMO_VIDEO = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4";

function DemoFeedVideo({ src, alt }) {
  const [useFallback, setUseFallback] = useState(false);
  const [failed, setFailed] = useState(false);
  const videoRef = useRef(null);

  useEffect(() => {
    if (useFallback) {
      const interval = setInterval(() => {
        if (videoRef.current) {
          window.__demoVideoTime = videoRef.current.currentTime;
        }
      }, 200);
      return () => clearInterval(interval);
    }
  }, [useFallback]);

  if (useFallback && DEMO_MODE) {
    return (
      <video
        ref={videoRef}
        src={DEMO_VIDEO}
        autoPlay
        loop
        muted
        playsInline
        style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
      />
    );
  }
  if (failed) {
    return (
      <div className="feed-error">
        <div style={{ fontSize: 22, marginBottom: 4 }}>◉</div>
        <div style={{ fontSize: 11 }}>Video stream unavailable</div>
      </div>
    );
  }
  // The stock clip is only ever shown in demo builds; a real deployment shows the error instead.
  return <img src={src} alt={alt} onError={() => (DEMO_MODE ? setUseFallback(true) : setFailed(true))} />;
}

// ── Camera Feed Component (OUTSIDE to avoid remount) ──
function CameraFeed({ cam, isMain = false, onClick, isSelected, layoutMode }) {
  return (
    <div
      onClick={onClick}
      className={`feed-container cursor-pointer ${isSelected && layoutMode === LAYOUT.SPOTLIGHT
        ? "ring-1 ring-indigo-500/40"
        : ""
        }`}
      style={!isMain ? { height: 120 } : {}}
    >
    {cam.status === "online" ? (
        <DemoFeedVideo src={`/api/cameras/${cam.id}/feed`} alt={cam.name} />
      ) : (
        <div style={{
          width: "100%", height: "100%",
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "var(--text-muted)",
        }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 24, marginBottom: 4 }}>
              {cam.status === "paused" ? "⏸" : "◉"}
            </div>
            <div style={{ fontSize: 10 }}>
              {cam.status === "paused" ? "Paused" : "Offline"}
            </div>
          </div>
        </div>
      )}
      <div className="feed-overlay">
        <span className={`cam-status-dot ${cam.status === "online" ? "online" : ""}`}
          style={cam.status !== "online" ? {
            background: cam.status === "paused" ? "var(--warning)" : "var(--danger)"
          } : {}}
        />
        <span className="cam-name">{cam.name}</span>
        <span className="cam-type">{cam.type}</span>
      </div>
    </div>
  );
}


export default function DashboardPage() {
  const [stats, setStats] = useState({ events_last_hour: 0, events_today: 0 });
  const [cameras, setCameras] = useState([]);
  const [layout, setLayout] = useState(LAYOUT.SPOTLIGHT);
  const [selectedCam, setSelectedCam] = useState(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newCamName, setNewCamName] = useState("");
  const [newCamSource, setNewCamSource] = useState("");
  const [addingCam, setAddingCam] = useState(false);
  const [addError, setAddError] = useState("");
  const backend = useBackendStatus();

  useEffect(() => {
    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission();
    }
  }, []);

  const notify = useCallback((ev) => {
    if (ev.risk_level === "LOW" || !("Notification" in window) || Notification.permission !== "granted") return;
    new Notification(`🚨 ${ev.risk_level} risk — ${ev.camera_name}`, {
      body: [ev.action, ...(ev.risk_reasons || [])].join(" · "),
      tag: `${ev.source}-${ev.unix_ts}`,
    });
  }, []);

  const { snapshots, events, transport } = useLiveData({ onEvent: notify });
  const detections = snapshots[selectedCam] || snapshots.cam_local || Object.values(snapshots)[0] || EMPTY_SNAPSHOT;
  const recentEvent = events[0];
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  const isActive = Boolean(recentEvent?.unix_ts && now / 1000 - recentEvent.unix_ts < 10);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch("/api/stats");
        if (res.ok) setStats(await res.json());
      } catch {
        /* backend offline — the sidebar shows it */
      }
    };
    fetchStats();
    const id = setInterval(fetchStats, 5000);
    return () => clearInterval(id);
  }, [events.length]);

  const selectedCamRef = useRef(selectedCam);
  selectedCamRef.current = selectedCam;

  const fetchCameras = useCallback(async () => {
    try {
      const res = await fetch("/api/cameras");
      const data = await res.json();
      setCameras(data);
      if (!selectedCamRef.current && data.length > 0) setSelectedCam(data[0].id);
    } catch (err) {
      console.error("Failed to fetch cameras", err);
    }
  }, []);

  useEffect(() => {
    fetchCameras();
    const id = setInterval(fetchCameras, 4000);
    return () => clearInterval(id);
  }, [fetchCameras]);

  // ── Quick add camera ──────────────────────────────────────────
  const addCamera = async () => {
    if (!newCamName.trim() || !newCamSource.trim()) return;
    setAddingCam(true);
    try {
      const res = await fetch("/api/cameras", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newCamName.trim(), source: newCamSource.trim() }),
      });
      if (res.ok) {
        setNewCamName("");
        setNewCamSource("");
        setAddError("");
        setShowAddModal(false);
        fetchCameras();
      } else {
        setAddError((await res.json()).error || res.statusText);
      }
    } catch (err) {
      setAddError(String(err));
    }
    setAddingCam(false);
  };

  // ── Helpers ───────────────────────────────────────────────────

  const formatTime = (ts) => {
    if (!ts) return "—";
    try {
      const d = new Date(typeof ts === "number" ? (ts < 1e12 ? ts * 1000 : ts) : ts);
      if (isNaN(d.getTime())) return "—";
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch { return "—"; }
  };

  const formatDate = (ts) => {
    if (!ts) return "—";
    try {
      const d = new Date(typeof ts === "number" ? (ts < 1e12 ? ts * 1000 : ts) : ts);
      if (isNaN(d.getTime())) return "—";
      return d.toLocaleDateString();
    } catch { return "—"; }
  };

  const riskClass = (level) => {
    if (level === "HIGH") return "badge-danger";
    if (level === "MEDIUM") return "badge-warning";
    return "badge-success";
  };

  // ── Render camera area ───────────────────────────────────────
  const renderCameraPanel = () => {
    if (cameras.length === 0) {
      return (
        <div className="card" style={{ padding: 40, textAlign: "center" }}>
          <div style={{ fontSize: 28, marginBottom: 8, opacity: 0.4 }}>◉</div>
          <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 12 }}>
            No cameras connected
          </p>
          <button className="btn-primary" onClick={() => setShowAddModal(true)}>
            + Add Camera
          </button>
        </div>
      );
    }

    if (layout === LAYOUT.SPOTLIGHT) {
      const main = cameras.find((c) => c.id === selectedCam) || cameras[0];
      return (
        <div style={{ display: "flex", gap: 10 }}>
          <div style={{ flex: 1 }}>
            <CameraFeed cam={main} isMain onClick={() => { }} isSelected={true} layoutMode={layout} />
          </div>
          {cameras.length > 1 && (
            <div style={{
              width: 150, display: "flex", flexDirection: "column",
              gap: 6,
            }}>
              {cameras.filter((c) => c.id !== main.id).map((cam) => (
                <CameraFeed
                  key={cam.id}
                  cam={cam}
                  onClick={() => setSelectedCam(cam.id)}
                  isSelected={selectedCam === cam.id}
                  layoutMode={layout}
                />
              ))}
            </div>
          )}
        </div>
      );
    }

    if (layout === LAYOUT.GRID) {
      return (
        <div style={{
          display: "grid",
          gridTemplateColumns: `repeat(${Math.min(cameras.length, 3)}, 1fr)`,
          gap: 8,
        }}>
          {cameras.map((cam) => (
            <CameraFeed
              key={cam.id}
              cam={cam}
              isMain
              onClick={() => setSelectedCam(cam.id)}
              isSelected={selectedCam === cam.id}
              layoutMode={layout}
            />
          ))}
        </div>
      );
    }

    // LIST
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {cameras.map((cam) => (
          <CameraFeed
            key={cam.id}
            cam={cam}
            isMain
            onClick={() => setSelectedCam(cam.id)}
            isSelected={selectedCam === cam.id}
            layoutMode={layout}
          />
        ))}
      </div>
    );
  };

  return (
    <div className="page-enter" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* ── Page Header ──────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <h2 style={{ fontSize: 18, fontWeight: 700, letterSpacing: -0.3 }}>Live Monitoring</h2>
          <div className="toggle-group">
            {[
              { key: LAYOUT.SPOTLIGHT, label: "Focus" },
              { key: LAYOUT.GRID, label: "Grid" },
              { key: LAYOUT.LIST, label: "Stack" },
            ].map((m) => (
              <button
                key={m.key}
                onClick={() => setLayout(m.key)}
                className={`toggle-pill ${layout === m.key ? "active" : ""}`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {cameras.length} source{cameras.length !== 1 ? "s" : ""}
          </span>
          <button className="btn-primary" onClick={() => setShowAddModal(true)}>
            + Add Camera
          </button>
        </div>
      </div>

      {/* ── Camera Feed + Detection Panel ─────────────────── */}
      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ flex: 1 }} className="layout-transition">
          {renderCameraPanel()}
        </div>
        <DetectionPanel detections={detections} riskClass={riskClass} transport={transport} />
      </div>

      {/* ── Metrics Strip ────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
        <div className="metric-tile">
          <span className="metric-label">Status</span>
          <span className="metric-value" style={{
            color: isActive ? "var(--danger)" : "var(--success)",
            fontSize: 16,
          }}>
            {isActive ? "⚠ ACTIVITY" : "● Idle"}
          </span>
        </div>
        <div className="metric-tile">
          <span className="metric-label">Events (1h)</span>
          <span className="metric-value">{stats.events_last_hour}</span>
        </div>
        <div className="metric-tile">
          <span className="metric-label">Events (today)</span>
          <span className="metric-value">{stats.events_today}</span>
        </div>
        <div className="metric-tile">
          <span className="metric-label">Risk Level</span>
          <span className="metric-value" style={{
            color: detections.risk_level === "HIGH" ? "var(--danger)"
              : detections.risk_level === "MEDIUM" ? "var(--warning)"
                : "var(--success)",
            fontSize: 16,
          }}>
            {detections.risk_level || "LOW"}
          </span>
        </div>
      </div>

      {/* ── Bottom: Status + Events ──────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 12 }}>
        {/* System Status */}
        <div className="card">
          <div className="card-header">
            <h3>System Status</h3>
            <span className={`badge ${isActive ? "badge-danger" : "badge-success"}`}>
              {isActive ? "ALERT" : "NORMAL"}
            </span>
          </div>
          <div className="card-body" style={{ padding: "10px 16px" }}>
            <div className="status-row">
              <span className="status-label">Backend</span>
              <span className="status-value" style={{ color: backend.online ? "var(--success)" : "var(--danger)" }}>
                {backend.online === null ? "…" : backend.online ? "Online" : "Offline"}
              </span>
            </div>
            <div className="status-row">
              <span className="status-label">Compute</span>
              <span className="status-value">
                {backend.perf ? (backend.perf.device === "cuda" ? "GPU" : "CPU") : "—"}
                {backend.perf?.pipeline_fps ? ` · ${backend.perf.pipeline_fps} fps` : ""}
              </span>
            </div>
            <div className="status-row">
              <span className="status-label">Activity AI</span>
              <span className="status-value" style={{ color: detections.model_ready ? "var(--success)" : "var(--warning)" }}>
                {backend.perf?.activity?.enabled === false ? "Disabled" : detections.model_ready ? "SlowFast R50" : "Loading…"}
              </span>
            </div>
            <div className="status-row">
              <span className="status-label">Faces</span>
              <span className="status-value" style={{ color: backend.perf?.face?.available ? "var(--success)" : "var(--warning)" }}>
                {backend.perf?.face ? (backend.perf.face.available ? backend.perf.face.backend : "Disabled") : "—"}
              </span>
            </div>
            <div className="status-row">
              <span className="status-label">Updates</span>
              <span className="status-value">{transport === "stream" ? "Live push" : transport === "polling" ? "Polling" : "…"}</span>
            </div>
            <div className="status-row">
              <span className="status-label">Last Event</span>
              <span className="status-value">
                {recentEvent ? `${formatDate(recentEvent.timestamp)} ${formatTime(recentEvent.timestamp)}` : "—"}
              </span>
            </div>
          </div>
        </div>

        {/* Recent Events */}
        <div className="card" style={{ maxHeight: 280, display: "flex", flexDirection: "column" }}>
          <div className="card-header">
            <h3>Recent Events</h3>
            <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
              {events.length} event{events.length !== 1 ? "s" : ""}
            </span>
          </div>
          <div style={{ flex: 1, overflowY: "auto" }}>
            {events.length === 0 ? (
              <div style={{
                padding: 24, textAlign: "center",
                fontSize: 12, color: "var(--text-muted)",
              }}>
                No events recorded yet
              </div>
            ) : (
              events.map((ev, idx) => {
                const known = (ev?.faces || []).filter((f) => f.name && f.name !== "unknown");
                const desc = known.length ? `Recognized: ${known.map((f) => f.name).join(", ")}`
                  : ev?.face_status === "unknown" || ev?.faces?.some((f) => f.score > 0) ? "Unrecognised person"
                    : ev?.face_status === "no_face_visible" ? "Person (face not visible)"
                      : "Person detected";
                const actionText = ev?.action || "";
                const heightText = ev?.height_m ? `${Number(ev.height_m).toFixed(2)} m` : "";

                return (
                  <div key={`${ev?.source}-${ev?.unix_ts ?? idx}`} className="event-row">
                    <span className="event-time">{formatTime(ev?.timestamp)}</span>
                    <div className="event-body">
                      <div className="event-title">
                        <span>{desc}</span>
                        {ev?.risk_level && (
                          <span className={`badge ${riskClass(ev.risk_level)}`}>
                            {ev.risk_level}
                          </span>
                        )}
                      </div>
                      <div className="event-desc">
                        {[actionText, heightText, ev?.camera_name, ...(ev?.risk_reasons || [])].filter(Boolean).join(" · ")}
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      {/* ── Add Camera Modal ─────────────────────────────── */}
      {showAddModal && (
        <div className="modal-backdrop" onClick={() => setShowAddModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>Add Camera Source</h3>
            <p style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 16 }}>
              A device index (e.g. 1) or an RTSP / HTTP stream URL
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 16 }}>
              <div>
                <label style={{ fontSize: 10, color: "var(--text-muted)", display: "block", marginBottom: 4, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>
                  Camera Name
                </label>
                <input
                  type="text"
                  placeholder="e.g. Front Entrance"
                  value={newCamName}
                  onChange={(e) => setNewCamName(e.target.value)}
                />
              </div>
              <div>
                <label style={{ fontSize: 10, color: "var(--text-muted)", display: "block", marginBottom: 4, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>
                  Stream URL
                </label>
                <input
                  type="text"
                  placeholder="http://192.168.1.100:8080/video"
                  value={newCamSource}
                  onChange={(e) => setNewCamSource(e.target.value)}
                />
              </div>
            </div>
            {addError && <p className="msg-error" style={{ marginBottom: 10 }}>{addError}</p>}
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button className="btn-secondary" onClick={() => setShowAddModal(false)}>
                Cancel
              </button>
              <button
                className="btn-primary"
                onClick={addCamera}
                disabled={addingCam || !newCamName.trim() || !newCamSource.trim()}
                style={addingCam || !newCamName.trim() || !newCamSource.trim() ? { opacity: 0.4 } : {}}
              >
                {addingCam ? "Adding…" : "Add Camera"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
