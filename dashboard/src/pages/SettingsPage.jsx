import { useEffect, useState, useRef } from "react";

export default function SettingsPage() {
  const [config, setConfig] = useState({
    motion_threshold: 5,
    alert_cooldown: 30,
    consecutive_required: 2,
    confidence_threshold: 0.6,
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const cardRef = useRef(null);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const res = await fetch("http://127.0.0.1:5000/config");
        const data = await res.json();
        setConfig(data);
      } catch (err) {
        console.error("Failed to fetch /config", err);
      }
    };
    fetchConfig();
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      const res = await fetch("http://127.0.0.1:5000/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          motion_threshold: config.motion_threshold,
          alert_cooldown: config.alert_cooldown,
          consecutive_required: config.consecutive_required,
          confidence_threshold: config.confidence_threshold,
        }),
      });
      const data = await res.json();
      if (data.ok !== false) {
        setSaved(true);
        // Flash success on the card
        if (cardRef.current) {
          cardRef.current.classList.remove("flash-success");
          void cardRef.current.offsetWidth; // force reflow
          cardRef.current.classList.add("flash-success");
        }
        setTimeout(() => setSaved(false), 2000);
      }
    } catch (err) {
      console.error(err);
    }
    setSaving(false);
  };

  const fields = [
    { key: "motion_threshold", label: "Motion Threshold", desc: "Minimum contour area to trigger detection", step: 1 },
    { key: "consecutive_required", label: "Consecutive Frames", desc: "Required consecutive detections before alert", step: 1 },
    { key: "confidence_threshold", label: "Confidence Threshold", desc: "YOLO detection confidence cutoff", step: 0.01 },
    { key: "alert_cooldown", label: "Alert Cooldown (s)", desc: "Seconds between repeated alerts", step: 1 },
  ];

  return (
    <div className="page-enter" style={{ maxWidth: 520 }}>
      <h2 style={{ fontSize: 18, fontWeight: 700, letterSpacing: -0.3, marginBottom: 16 }}>
        Detection Settings
      </h2>

      <div className="card" ref={cardRef}>
        <div className="card-header">
          <h3>ML Pipeline Configuration</h3>
          {saved && (
            <span className="badge badge-success" style={{ animation: "scale-in 0.3s var(--ease-spring)" }}>
              ✓ Saved
            </span>
          )}
        </div>
        <div className="card-body" style={{ padding: "12px 20px 20px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {fields.map((f, i) => (
              <div
                key={f.key}
                style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  padding: "8px 0",
                  borderBottom: i < fields.length - 1 ? "1px solid rgba(99,102,241,0.06)" : "none",
                  animation: `slide-up 0.35s var(--ease-out-expo) ${i * 0.06}s both`,
                }}
              >
                <div>
                  <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text-primary)", marginBottom: 2 }}>
                    {f.label}
                  </div>
                  <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
                    {f.desc}
                  </div>
                </div>
                <input
                  type="number"
                  step={f.step}
                  className="input-animated"
                  style={{ width: 90, textAlign: "center" }}
                  value={config[f.key]}
                  onChange={(e) =>
                    setConfig((c) => ({ ...c, [f.key]: Number(e.target.value) }))
                  }
                />
              </div>
            ))}
          </div>

          <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 16, gap: 8 }}>
            <button
              className="btn-primary"
              onClick={save}
              disabled={saving}
              style={saving ? { opacity: 0.5 } : {}}
            >
              {saving ? "Saving…" : saved ? "✓ Saved!" : "Save Configuration"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
