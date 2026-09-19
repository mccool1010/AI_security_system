import { useState, useEffect } from "react";
import { Link } from "react-router-dom";

export default function LandingPage() {
  const [systemInfo, setSystemInfo] = useState(null);
  const [backendLive, setBackendLive] = useState(false);

  useEffect(() => {
    fetch("/api/system/info")
      .then((r) => r.json())
      .then((data) => {
        setSystemInfo(data);
        setBackendLive(true);
      })
      .catch(() => setBackendLive(false));
  }, []);

  const features = [
    {
      icon: "🎯",
      title: "Pose & Posture",
      desc: "YOLOv8-Pose keypoints tracked per person; posture (walking, running, falling, lying down, fighting) from body-size-normalised motion, so it holds at any resolution or frame rate",
      tech: "YOLOv8n-Pose",
    },
    {
      icon: "📏",
      title: "Height Measurement",
      desc: "Ground-plane geometry: calibrate once with a person of known height, then every person is measured independently. The calibration reports its own leave-one-out error",
      tech: "Camera geometry",
    },
    {
      icon: "🏃",
      title: "Activity Recognition",
      desc: "SlowFast R50 on person-centred clips, reduced to a few categories with a clear CCTV meaning: fighting, falls, running, climbing, spray painting",
      tech: "SlowFast R50",
    },
    {
      icon: "👤",
      title: "Face Recognition",
      desc: "Multi-sample enrollment with FaceNet embeddings. Every enrollment records the model that made it, so a model change can never silently mis-match people",
      tech: "FaceNet + MTCNN",
    },
    {
      icon: "📷",
      title: "Multi-Camera",
      desc: "USB, IP and RTSP cameras, one processing pipeline each whether or not anyone is watching, with 18 lens presets and checkerboard calibration",
      tech: "OpenCV",
    },
    {
      icon: "⚡",
      title: "Live Push & Measured Performance",
      desc: "Detections and alerts are pushed to the dashboard over Server-Sent Events; throughput and latency are measured continuously and shown in the app",
      tech: "SSE + /api/perf",
    },
  ];

  const techStack = [
    { name: "Python", color: "#3776AB" },
    { name: "PyTorch", color: "#EE4C2C" },
    { name: "OpenCV", color: "#5C3EE8" },
    { name: "Flask", color: "#999" },
    { name: "React", color: "#61DAFB" },
    { name: "MongoDB", color: "#47A248" },
    { name: "Docker", color: "#2496ED" },
    { name: "YOLOv8", color: "#FF6F00" },
  ];

  // Measured on an RTX 5050 laptop GPU (see README "Benchmarks"); the live dashboard shows current values.
  const metrics = [
    { label: "Pipeline throughput", value: "48 fps", sub: "max per-frame path, RTX 5050 laptop" },
    { label: "Camera → dashboard", value: "≈60 ms", sub: "median, live push" },
    { label: "Neural networks", value: "5", sub: "pose · activity · 2× face · depth" },
    { label: "Height error", value: "measured", sub: "per install, leave-one-out" },
  ];

  // Replace this with your actual YouTube video ID after recording
  const DEMO_VIDEO_ID = null; // e.g., "dQw4w9WgXcQ"
  // Replace with your GitHub repo URL
  const GITHUB_URL = "https://github.com/mccool1010/AI_security_system";

  return (
    <div style={styles.page}>
      {/* ── Hero Section ──────────────────────────────────────── */}
      <section style={styles.hero}>
        <div style={styles.heroGlow} />
        <div style={styles.heroGlow2} />
        <div style={styles.heroContent}>
          <div style={styles.badge}>
            <span style={styles.badgeDot} />
            {backendLive ? "LIVE SYSTEM" : "AI SECURITY SYSTEM"}
          </div>

          <h1 style={styles.heroTitle}>
            <span style={styles.heroTitleAccent}>SecureVision</span>
            <br />
            <span style={{ fontSize: "0.55em", fontWeight: 600, color: "#94a3b8" }}>
              AI-Powered Security Platform
            </span>
          </h1>

          <p style={styles.heroSubtitle}>
            Full-stack intelligent surveillance system with real-time pose estimation,
            calibrated height measurement, activity recognition, and face identification
            — five neural networks in one real-time pipeline per camera.
          </p>

          <div style={styles.heroButtons}>
            {backendLive ? (
              <Link to="/dashboard" style={styles.primaryBtn}>
                <span>Enter Live Dashboard</span>
                <span style={{ fontSize: 18 }}>→</span>
              </Link>
            ) : (
              <Link to="/dashboard" style={styles.primaryBtn}>
                <span>View Live Demo</span>
                <span style={{ fontSize: 18 }}>→</span>
              </Link>
            )}
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              style={styles.secondaryBtn}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
              View Source Code
            </a>
          </div>

          {backendLive && (
            <div style={styles.liveIndicator}>
              <span style={styles.livePulse} />
              <span>ML Pipeline Active — Processing {systemInfo?.demo_mode ? "Demo Feed" : "Live Camera"}</span>
            </div>
          )}
        </div>
      </section>

      {/* ── Metrics Bar ───────────────────────────────────────── */}
      <section style={styles.metricsBar}>
        {metrics.map((m, i) => (
          <div key={i} style={styles.metricItem}>
            <div style={styles.metricValue}>{m.value}</div>
            <div style={styles.metricLabel}>{m.label}</div>
            <div style={styles.metricSub}>{m.sub}</div>
          </div>
        ))}
      </section>

      {/* ── Demo Video ────────────────────────────────────────── */}
      {DEMO_VIDEO_ID && (
        <section id="demo" style={styles.section}>
          <h2 style={styles.sectionTitle}>Live Demo</h2>
          <p style={styles.sectionSubtitle}>Watch the system processing real-time video</p>
          <div style={styles.videoContainer}>
            <iframe
              width="100%"
              height="100%"
              src={`https://www.youtube.com/embed/${DEMO_VIDEO_ID}?rel=0`}
              title="SecureVision AI Demo"
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              style={{ borderRadius: 12 }}
            />
          </div>
        </section>
      )}

      {/* ── Architecture ──────────────────────────────────────── */}
      <section style={styles.section}>
        <h2 style={styles.sectionTitle}>System Architecture</h2>
        <p style={styles.sectionSubtitle}>
          One pipeline per camera; activity, faces and depth run on shared background workers
        </p>

        <div style={styles.pipelineContainer}>
          {[
            { icon: "📹", name: "Camera Feed", detail: "USB / IP / RTSP", gradient: "linear-gradient(135deg, #667eea, #764ba2)" },
            { icon: "🦴", name: "YOLOv8-Pose", detail: "17 Keypoints", gradient: "linear-gradient(135deg, #f093fb, #f5576c)" },
            { icon: "🧠", name: "SlowFast R50", detail: "Activity Class", gradient: "linear-gradient(135deg, #4facfe, #00f2fe)" },
            { icon: "📏", name: "Height", detail: "Ground geometry", gradient: "linear-gradient(135deg, #43e97b, #38f9d7)" },
            { icon: "🛡️", name: "Risk Engine", detail: "Alert + Log", gradient: "linear-gradient(135deg, #fa709a, #fee140)" },
          ].map((step, i, arr) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={styles.pipelineStep}>
                <div style={{ ...styles.pipelineIcon, background: step.gradient }}>{step.icon}</div>
                <div style={styles.pipelineName}>{step.name}</div>
                <div style={styles.pipelineDetail}>{step.detail}</div>
              </div>
              {i < arr.length - 1 && <div style={styles.pipelineArrow}>→</div>}
            </div>
          ))}
        </div>
      </section>

      {/* ── Features Grid ─────────────────────────────────────── */}
      <section id="features" style={styles.section}>
        <h2 style={styles.sectionTitle}>Key Features</h2>
        <div style={styles.featuresGrid}>
          {features.map((f, i) => (
            <div key={i} style={styles.featureCard}>
              <div style={styles.featureIcon}>{f.icon}</div>
              <h3 style={styles.featureTitle}>{f.title}</h3>
              <p style={styles.featureDesc}>{f.desc}</p>
              <span style={styles.featureTech}>{f.tech}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Height Estimation Deep Dive ───────────────────────── */}
      <section style={{ ...styles.section, background: "rgba(99,102,241,0.03)", borderRadius: 20, margin: "0 auto", maxWidth: 1000, padding: "50px 30px" }}>
        <h2 style={styles.sectionTitle}>Height Estimation Pipeline</h2>
        <p style={styles.sectionSubtitle}>No assumed heights — every person is measured from where they stand</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16, marginTop: 30 }}>
          {[
            { step: "1", title: "Calibrate once", desc: "A person of known height stands at 3+ distances; camera height and tilt are fitted" },
            { step: "2", title: "Error you can trust", desc: "Leave-one-out error over the reference samples is reported with the calibration" },
            { step: "3", title: "Track", desc: "Each person is tracked; only upright, fully visible, large-enough views are measured" },
            { step: "4", title: "Feet & head rows", desc: "Where the feet and head appear in the image, smoothed over a short window" },
            { step: "5", title: "Ground-plane geometry", desc: "Feet row → distance along the ground; head row → height above it" },
            { step: "6", title: "Robust per person", desc: "Median of all measurements, with the spread shown as ±" },
          ].map((s, i) => (
            <div key={i} style={{ padding: "18px 16px", borderRadius: 12, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
              <div style={{ width: 28, height: 28, borderRadius: 8, background: "linear-gradient(135deg, #6366f1, #8b5cf6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 700, marginBottom: 10 }}>{s.step}</div>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 6, color: "#f1f5f9" }}>{s.title}</div>
              <div style={{ fontSize: 12, lineHeight: 1.5, color: "#64748b" }}>{s.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Tech Stack ────────────────────────────────────────── */}
      <section style={styles.section}>
        <h2 style={styles.sectionTitle}>Built With</h2>
        <div style={styles.techGrid}>
          {techStack.map((t, i) => (
            <div key={i} style={styles.techBadge}>
              <span style={{ ...styles.techDot, background: t.color, boxShadow: `0 0 8px ${t.color}55` }} />
              {t.name}
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA ───────────────────────────────────────────────── */}
      <section style={{ ...styles.section, textAlign: "center", paddingBottom: 60 }}>
        <div style={{ display: "flex", gap: 14, justifyContent: "center", flexWrap: "wrap" }}>
          {backendLive && (
            <Link to="/dashboard" style={{ ...styles.primaryBtn, fontSize: 16, padding: "14px 36px" }}>
              Enter Live Dashboard →
            </Link>
          )}
          <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" style={{ ...styles.secondaryBtn, fontSize: 16, padding: "14px 36px" }}>
            View Source on GitHub
          </a>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────── */}
      <footer style={styles.footer}>
        <p style={{ margin: 0 }}>SecureVision AI — Full-stack security system built with PyTorch, OpenCV, React & Flask</p>
      </footer>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
const styles = {
  page: {
    minHeight: "100vh",
    background: "#0a0a0f",
    color: "#e2e8f0",
    fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
    overflowX: "hidden",
  },
  hero: {
    position: "relative",
    padding: "100px 24px 60px",
    textAlign: "center",
    overflow: "hidden",
  },
  heroGlow: {
    position: "absolute",
    top: "-40%",
    left: "50%",
    transform: "translateX(-50%)",
    width: 800,
    height: 800,
    borderRadius: "50%",
    background: "radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%)",
    pointerEvents: "none",
  },
  heroGlow2: {
    position: "absolute",
    top: "-20%",
    right: "-10%",
    width: 400,
    height: 400,
    borderRadius: "50%",
    background: "radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 70%)",
    pointerEvents: "none",
  },
  heroContent: {
    position: "relative",
    maxWidth: 720,
    margin: "0 auto",
  },
  badge: {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    padding: "6px 16px",
    borderRadius: 20,
    background: "rgba(99,102,241,0.1)",
    border: "1px solid rgba(99,102,241,0.2)",
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: 2,
    color: "#818cf8",
    marginBottom: 28,
  },
  badgeDot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    background: "#22c55e",
    boxShadow: "0 0 8px rgba(34,197,94,0.6)",
    animation: "pulse 2s infinite",
  },
  heroTitle: {
    fontSize: 56,
    fontWeight: 800,
    lineHeight: 1.1,
    margin: "0 0 24px",
    letterSpacing: -2,
  },
  heroTitleAccent: {
    background: "linear-gradient(135deg, #818cf8, #6366f1, #a78bfa)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
  },
  heroSubtitle: {
    fontSize: 17,
    lineHeight: 1.7,
    color: "#94a3b8",
    maxWidth: 600,
    margin: "0 auto 36px",
  },
  heroButtons: {
    display: "flex",
    gap: 14,
    justifyContent: "center",
    flexWrap: "wrap",
  },
  primaryBtn: {
    display: "inline-flex",
    alignItems: "center",
    gap: 10,
    padding: "13px 28px",
    borderRadius: 10,
    background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
    color: "#fff",
    fontWeight: 600,
    fontSize: 15,
    textDecoration: "none",
    border: "none",
    cursor: "pointer",
    boxShadow: "0 4px 20px rgba(99,102,241,0.3)",
    transition: "transform 0.15s, box-shadow 0.15s",
  },
  secondaryBtn: {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    padding: "13px 28px",
    borderRadius: 10,
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.1)",
    color: "#cbd5e1",
    fontWeight: 500,
    fontSize: 15,
    textDecoration: "none",
    cursor: "pointer",
  },
  liveIndicator: {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    marginTop: 28,
    padding: "8px 18px",
    borderRadius: 8,
    background: "rgba(34,197,94,0.08)",
    border: "1px solid rgba(34,197,94,0.2)",
    fontSize: 13,
    color: "#4ade80",
  },
  livePulse: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#22c55e",
    boxShadow: "0 0 8px rgba(34,197,94,0.6)",
  },
  metricsBar: {
    display: "flex",
    justifyContent: "center",
    gap: 1,
    maxWidth: 700,
    margin: "0 auto 40px",
    padding: "0 24px",
    flexWrap: "wrap",
  },
  metricItem: {
    flex: "1 1 140px",
    textAlign: "center",
    padding: "24px 16px",
    background: "rgba(255,255,255,0.02)",
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.04)",
  },
  metricValue: {
    fontSize: 28,
    fontWeight: 800,
    background: "linear-gradient(135deg, #818cf8, #a78bfa)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    letterSpacing: -1,
  },
  metricLabel: {
    fontSize: 12,
    fontWeight: 600,
    color: "#e2e8f0",
    marginTop: 4,
  },
  metricSub: {
    fontSize: 11,
    color: "#475569",
    marginTop: 2,
  },
  section: {
    maxWidth: 1000,
    margin: "0 auto",
    padding: "60px 24px",
  },
  sectionTitle: {
    fontSize: 30,
    fontWeight: 700,
    textAlign: "center",
    marginBottom: 8,
    letterSpacing: -0.5,
  },
  sectionSubtitle: {
    textAlign: "center",
    color: "#64748b",
    fontSize: 15,
    marginBottom: 40,
  },
  videoContainer: {
    maxWidth: 800,
    margin: "0 auto",
    aspectRatio: "16/9",
    borderRadius: 14,
    overflow: "hidden",
    border: "1px solid rgba(99,102,241,0.2)",
    background: "#000",
  },
  pipelineContainer: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexWrap: "wrap",
    gap: 4,
    padding: "20px 0",
  },
  pipelineStep: {
    textAlign: "center",
    width: 120,
  },
  pipelineIcon: {
    width: 52,
    height: 52,
    borderRadius: 14,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 22,
    margin: "0 auto 10px",
  },
  pipelineName: {
    fontSize: 13,
    fontWeight: 600,
    color: "#e2e8f0",
  },
  pipelineDetail: {
    fontSize: 11,
    color: "#64748b",
    marginTop: 3,
  },
  pipelineArrow: {
    fontSize: 20,
    color: "#334155",
    padding: "0 2px",
  },
  featuresGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
    gap: 16,
  },
  featureCard: {
    padding: "24px 22px",
    borderRadius: 14,
    background: "rgba(255,255,255,0.02)",
    border: "1px solid rgba(255,255,255,0.05)",
    transition: "border-color 0.2s",
  },
  featureIcon: {
    fontSize: 28,
    marginBottom: 12,
  },
  featureTitle: {
    fontSize: 16,
    fontWeight: 600,
    marginBottom: 8,
    color: "#f1f5f9",
  },
  featureDesc: {
    fontSize: 13,
    lineHeight: 1.65,
    color: "#94a3b8",
    marginBottom: 14,
  },
  featureTech: {
    fontSize: 11,
    padding: "4px 10px",
    borderRadius: 6,
    background: "rgba(99,102,241,0.08)",
    color: "#818cf8",
    fontWeight: 600,
  },
  techGrid: {
    display: "flex",
    flexWrap: "wrap",
    gap: 12,
    justifyContent: "center",
  },
  techBadge: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "10px 18px",
    borderRadius: 10,
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.06)",
    fontSize: 14,
    fontWeight: 500,
  },
  techDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
  },
  footer: {
    padding: "30px 24px",
    textAlign: "center",
    borderTop: "1px solid rgba(255,255,255,0.05)",
    fontSize: 13,
    color: "#475569",
  },
};
