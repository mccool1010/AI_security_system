import { useCallback, useEffect, useRef, useState } from "react";

async function api(path, body) {
  const res = await fetch(path, body === undefined ? undefined : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

const FIELDS = [
  { key: "confidence_threshold", label: "Detection confidence", desc: "Minimum YOLO person confidence (0.05–0.95)", step: 0.01 },
  { key: "motion_threshold", label: "Motion frames", desc: "Frames with motion needed before a routine event", step: 1 },
  { key: "consecutive_required", label: "Consecutive detections", desc: "Frames with a person needed before a routine event", step: 1 },
  { key: "alert_cooldown", label: "Event cooldown (s)", desc: "Minimum seconds between routine events per camera. Falls, fights and loitering alert immediately.", step: 1 },
];

function DetectionSettings() {
  const [config, setConfig] = useState(null);
  const [state, setState] = useState({ saving: false, saved: false, error: "" });
  const cardRef = useRef(null);

  useEffect(() => {
    api("/api/config").then(setConfig).catch((e) => setState((s) => ({ ...s, error: e.message })));
  }, []);

  const save = async () => {
    setState({ saving: true, saved: false, error: "" });
    try {
      const data = await api("/api/config", Object.fromEntries(FIELDS.map((f) => [f.key, config[f.key]])));
      setConfig(data.config);
      setState({ saving: false, saved: true, error: "" });
      if (cardRef.current) {
        cardRef.current.classList.remove("flash-success");
        void cardRef.current.offsetWidth;
        cardRef.current.classList.add("flash-success");
      }
      setTimeout(() => setState((s) => ({ ...s, saved: false })), 2000);
    } catch (e) {
      setState({ saving: false, saved: false, error: e.message });
    }
  };

  return (
    <div className="card" ref={cardRef}>
      <div className="card-header">
        <h3>Event detection</h3>
        {state.saved && <span className="badge badge-success">✓ Saved</span>}
      </div>
      <div className="card-body" style={{ padding: "12px 20px 20px" }}>
        {!config ? (
          <p className={state.error ? "msg-error" : "form-note"}>{state.error || "Loading…"}</p>
        ) : (
          <>
            {FIELDS.map((f, i) => (
              <div key={f.key} className="form-row"
                style={{ borderBottom: i < FIELDS.length - 1 ? "1px solid rgba(99,102,241,0.06)" : "none" }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text-primary)", marginBottom: 2 }}>{f.label}</div>
                  <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{f.desc}</div>
                </div>
                <input
                  type="number"
                  step={f.step}
                  className="input-animated"
                  style={{ width: 90, textAlign: "center" }}
                  value={config[f.key] ?? ""}
                  onChange={(e) => setConfig((c) => ({ ...c, [f.key]: Number(e.target.value) }))}
                />
              </div>
            ))}
            {state.error && <p className="msg-error" style={{ marginTop: 8 }}>{state.error}</p>}
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 16 }}>
              <button className="btn-primary" onClick={save} disabled={state.saving}
                style={state.saving ? { opacity: 0.5 } : {}}>
                {state.saving ? "Saving…" : "Save"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function HeightCalibration() {
  const [cameras, setCameras] = useState([]);
  const [camId, setCamId] = useState("cam_local");
  const [status, setStatus] = useState(null);
  const [refHeight, setRefHeight] = useState("");
  const [camHeight, setCamHeight] = useState("");
  const [msg, setMsg] = useState({ text: "", error: false });
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setStatus(await api(`/api/calibrate/ground?camera_id=${encodeURIComponent(camId)}`));
    } catch (e) {
      setMsg({ text: e.message, error: true });
    }
  }, [camId]);

  useEffect(() => {
    api("/api/cameras").then(setCameras).catch(() => {});
  }, []);
  useEffect(() => {
    load();
  }, [load]);

  const run = async (fn, ok) => {
    setBusy(true);
    setMsg({ text: "", error: false });
    try {
      const data = await fn();
      setMsg({ text: ok(data), error: false });
      await load();
    } catch (e) {
      setMsg({ text: e.message, error: true });
    }
    setBusy(false);
  };

  const params = status?.params;
  const samples = status?.samples || [];

  return (
    <div className="card">
      <div className="card-header">
        <h3>Height calibration</h3>
        <span className={`badge ${params ? "badge-success" : "badge-warning"}`}>
          {params ? "Calibrated" : "Not calibrated"}
        </span>
      </div>
      <div className="card-body" style={{ padding: "12px 20px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
        <p className="form-note">
          Heights are measured from where a person's feet and head appear in the image, which needs the
          camera's mounting height and tilt. Have one person whose height you know stand fully in view,
          and record a sample at three or more different distances from the camera. If you measured the
          mount height, enter it for a better fit.
        </p>

        {cameras.length > 1 && (
          <div className="form-row">
            <span style={{ fontSize: 12 }}>Camera</span>
            <select value={camId} onChange={(e) => setCamId(e.target.value)}>
              {cameras.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
          </div>
        )}

        <div className="form-row">
          <span style={{ fontSize: 12 }}>Reference person height (m)</span>
          <div style={{ display: "flex", gap: 8 }}>
            <input type="number" step="0.01" min="0.5" max="2.6" placeholder="1.75" style={{ width: 90 }}
              value={refHeight} onChange={(e) => setRefHeight(e.target.value)} />
            <button className="btn-secondary" disabled={busy || !refHeight}
              onClick={() => run(
                () => api("/api/calibrate/ground/sample", { camera_id: camId, height_m: Number(refHeight) }),
                (d) => `Sample ${d.samples} recorded (feet at row ${Math.round(d.sample.v_foot)}).`,
              )}>
              Record sample
            </button>
          </div>
        </div>

        <div className="form-row">
          <span style={{ fontSize: 12 }}>Camera mount height (m, optional)</span>
          <div style={{ display: "flex", gap: 8 }}>
            <input type="number" step="0.01" placeholder="auto" style={{ width: 90 }}
              value={camHeight} onChange={(e) => setCamHeight(e.target.value)} />
            <button className="btn-primary" disabled={busy || samples.length < (camHeight ? 1 : 2)}
              onClick={() => run(
                () => api("/api/calibrate/ground/solve", {
                  camera_id: camId, camera_height_m: camHeight ? Number(camHeight) : null,
                }),
                (d) => `Calibrated from ${d.params.n_samples} samples.`,
              )}>
              Calibrate
            </button>
          </div>
        </div>

        {msg.text && <p className={msg.error ? "msg-error" : "msg-ok"}>{msg.text}</p>}

        <table className="kv-table">
          <tbody>
            <tr><td>Reference samples</td><td>{samples.length}</td></tr>
            {params && (
              <>
                <tr><td>Camera height</td><td>{params.camera_height_m} m{params.fixed_camera_height ? " (entered)" : " (fitted)"}</td></tr>
                <tr><td>Camera tilt</td><td>{params.tilt_deg}°</td></tr>
                <tr><td>Fit error (RMS)</td><td>{params.rms_cm != null ? `${params.rms_cm} cm` : "—"}</td></tr>
                <tr>
                  <td title="Each sample predicted from a calibration fitted without it — an honest estimate of measurement error">
                    Leave-one-out error (RMS)
                  </td>
                  <td>{params.loo_rms_cm != null ? `${params.loo_rms_cm} cm` : "needs more samples"}</td>
                </tr>
              </>
            )}
          </tbody>
        </table>

        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button className="btn-secondary" disabled={busy || (!params && !samples.length)}
            onClick={() => window.confirm("Delete all samples and the calibration for this camera?") && run(
              () => api("/api/calibrate/ground/reset", { camera_id: camId }),
              () => "Calibration cleared.",
            )}>
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}

export default function SettingsPage() {
  return (
    <div className="page-enter" style={{ maxWidth: 560, display: "flex", flexDirection: "column", gap: 16 }}>
      <h2 style={{ fontSize: 18, fontWeight: 700, letterSpacing: -0.3 }}>Settings</h2>
      <DetectionSettings />
      <HeightCalibration />
    </div>
  );
}
