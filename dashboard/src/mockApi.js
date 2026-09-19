/**
 * Mock API for the static portfolio demo (build with VITE_DEMO_MODE=true).
 * Intercepts fetch() and returns simulated data in the same shape as the real
 * backend. Everything here is made up and the UI says so with a banner.
 */

function person(id, pose, confidence, dwell_s) {
  return {
    track_id: id, box: [0, 0, 0, 0], confidence, pose, stable_pose: pose, truncated: false,
    dwell: dwell_s < 5 ? "passing" : dwell_s < 15 ? "visitor" : "lingering", dwell_s,
    identity: null, height_m: null, height_spread_m: null, height_samples: 0,
    distance_m: null, height_status: "camera not calibrated", relative_depth: null,
  };
}

const started = Date.now();

// Timeline matched to intel-iot-devkit person-bicycle-car-detection.mp4
function mockDetections() {
  const t = window.__demoVideoTime || 0;
  const since = (s) => Math.max(0, t - s);
  const persons = [];
  const activities = [];
  if (t >= 2 && t < 8) persons.push(person("p1", "walking", 0.89, since(2)));
  else if (t >= 26 && t < 30) {
    persons.push(person("p2", "running", 0.82, since(26)));
    activities.push({ action: "Running", category: "running", severity: "MEDIUM", confidence: 0.41, raw_label: "jogging" });
  } else if (t >= 41 && t < 50) persons.push(person("p3", "walking", 0.91, since(41)));

  const risk_reasons = activities.map((a) => `${a.action.toLowerCase()} (${a.confidence})`);
  return {
    camera_id: "cam_local",
    ts: Date.now() / 1000,
    persons,
    person_count: persons.length,
    activities,
    risk_level: "LOW",
    risk_score: activities.length ? 1 : 0,
    risk_reasons,
    motion: persons.length > 0,
    model_ready: true,
  };
}

const eventCache = [];
let lastKey = null;

function maybeEvent() {
  const d = mockDetections();
  const key = d.persons.map((p) => p.track_id).join(",");
  if (!key || key === lastKey) return;
  lastKey = key;
  const p = d.persons[0];
  const now = Date.now() / 1000;
  eventCache.unshift({
    timestamp: new Date().toISOString(),
    unix_ts: now,
    source: "cam_local",
    camera_name: "Main Entrance",
    trigger: "activity",
    action: [d.activities[0]?.action, p.pose.replace(/^\w/, (c) => c.toUpperCase()), p.dwell].filter(Boolean).join(" — "),
    persons: d.persons,
    activities: d.activities,
    faces: [],
    face_status: "no_face_visible",
    detections: d.persons.map((x) => ({ class: "person", confidence: x.confidence, box: x.box })),
    height_m: null,
    risk_score: d.risk_score,
    risk_level: d.risk_level,
    risk_reasons: d.risk_reasons,
    pipeline_version: "demo",
  });
  if (eventCache.length > 30) eventCache.pop();
}

setInterval(maybeEvent, 500);

const mockStats = () => ({
  events_last_hour: eventCache.length,
  events_today: eventCache.length,
});

const mockCameras = () => [{
  id: "cam_local", name: "Main Entrance", source: "demo", type: "demo",
  status: "online", paused: false, error: null, viewers: 1, height_calibrated: false,
}];

const mockPerf = () => ({
  device: "demo", gpu: null, torch: "—",
  activity: { enabled: true, ready: true, device: "demo", last_inference_ms: null },
  depth: { enabled: false, ready: false },
  face: { backend: "demo", available: false },
  pipeline_fps: null,
  cameras: {},
  uptime_s: (Date.now() - started) / 1000,
});

const mockSystemInfo = () => ({
  demo_mode: true,
  models: {
    pose: "YOLOv8n-Pose (COCO 17 keypoints)",
    activity: "SlowFast R50 (Kinetics-400, curated categories)",
    depth: "MiDaS small (optional, relative depth)",
    face: "FaceNet InceptionResnetV1 (VGGFace2) + MTCNN",
  },
  features: [],
});

const mockMetrics = () => ({
  events_by_hour: Array.from({ length: 24 }, (_, h) => ({ hour: h, count: h === 23 ? eventCache.length : 0 })),
  detection_distribution: [{ action: "activity", count: eventCache.length }],
  confidence_histogram: Array.from({ length: 10 }, (_, b) => ({ range: `${b * 10}-${b * 10 + 10}%`, count: b === 8 ? eventCache.length : 0 })),
  risk_distribution: [{ level: "LOW", count: eventCache.length }, { level: "MEDIUM", count: 0 }, { level: "HIGH", count: 0 }],
  face_match_rate: 0,
  face_observations: 0,
  face_match_rate_note: "demo data",
  total_events_24h: eventCache.length,
});

const DEMO_CONFIG = { confidence_threshold: 0.5, motion_threshold: 5, consecutive_required: 2, alert_cooldown: 5 };

const ROUTES = [
  ["/api/auth", () => ({ required: false, authenticated: true })],
  ["/api/detections", mockDetections],
  ["/api/activity_status", () => ({ active: eventCache.length > 0, risk_level: "LOW", last_event: eventCache[0]?.timestamp ?? null })],
  ["/api/metrics", mockMetrics],
  ["/api/perf", mockPerf],
  ["/api/system/info", mockSystemInfo],
  ["/api/users", () => ({ backend: "demo", threshold: 0.6, matchable: [], needs_reenrollment: [] })],
  ["/api/calibrate/ground", () => ({ calibrated: false, params: null, samples: [] })],
  ["/api/events", () => [...eventCache]],
  ["/api/stats", mockStats],
  ["/api/status", () => ({ service: "ok", mongo: { ok: true, error: "" }, camera_open: true })],
  ["/api/health", () => ({ status: "ok", demo_mode: true })],
  ["/api/config", () => DEMO_CONFIG],
];

const _originalFetch = window.fetch;

export function installMockAPI() {
  console.log("Mock API installed — demo mode, all data is simulated");
  window.fetch = async function (url, options) {
    const path = typeof url === "string" ? url : url?.url || "";
    if ((options?.method || "GET").toUpperCase() !== "GET") {
      if (path.startsWith("/api/login") || path.startsWith("/api/logout")) {
        return jsonResponse({ ok: true, required: false });
      }
      if (path.startsWith("/api/config")) {
        return jsonResponse({ ok: true, config: { ...DEMO_CONFIG, ...JSON.parse(options.body || "{}") } });
      }
      return jsonResponse({ error: "Not available in the demo" }, 503);
    }
    if (path.includes("/api/cameras") && !path.includes("/feed")) return jsonResponse(mockCameras());
    for (const [prefix, handler] of ROUTES) {
      if (path.startsWith(prefix)) return jsonResponse(handler());
    }
    return _originalFetch.apply(this, arguments);
  };
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
