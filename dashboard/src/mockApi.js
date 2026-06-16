/**
 * Mock API — Simulates the Flask backend for the portfolio demo.
 * Intercepts fetch() calls and returns realistic security system data.
 * This allows the REAL dashboard to work on Vercel without a backend.
 */

const ACTIVITIES = [
  "walking", "walking", "walking", "walking", "standing", "carrying object"
];

const NAMES = [null, null, null, null, null, null];
const POSES = ["full_body", "walking"];

let eventCounter = 0;

function rand(min, max) {
  return min + Math.random() * (max - min);
}

// Helper for mock data
function createPerson(height, pose, confidence) {
  return {
    confidence, class: "person", height_m: height, margin_m: 0.03,
    distance_m: 5.5, pose, visibility: "full_body", face: null,
    quality_score: 0.85, kalman_updates: 24, raw_height_m: height
  };
}

// ── Mock: /api/detections ─────────────────────────────────────
function mockDetections() {
  const t = window.__demoVideoTime || 0;
  let persons = [];
  let activities = [];
  let risk = "LOW";

  // Timeline matched exactly to intel-iot-devkit person-bicycle-car-detection.mp4
  if (t >= 2 && t < 8) {
    // 2s - 8s: Person walking
    persons.push(createPerson(1.74, "full_body", 0.89));
    activities.push({ action: "walking", confidence: 0.92 });
  } 
  else if (t >= 17 && t < 20) {
    // 17s - 20s: Car driving by
    activities.push({ action: "driving vehicle", confidence: 0.96 });
  } 
  else if (t >= 26 && t < 30) {
    // 26s - 30s: Cyclist
    persons.push(createPerson(1.68, "riding", 0.82));
    activities.push({ action: "riding bicycle", confidence: 0.88 });
  } 
  else if (t >= 41 && t < 44) {
    // 41s - 44s: 1 Person
    persons.push(createPerson(1.72, "walking", 0.91));
    activities.push({ action: "walking", confidence: 0.90 });
  }
  else if (t >= 44 && t < 47.5) {
    // 44s - 47.5s: Person + Car (no bicycle yet)
    persons.push(createPerson(1.72, "walking", 0.91));
    activities.push({ action: "walking", confidence: 0.88 });
    activities.push({ action: "driving vehicle", confidence: 0.85 });
  }
  else if (t >= 47.5 && t < 50) {
    // 47.5s - 50s: Busy scene (Person, Car, AND Bicycle)
    persons.push(createPerson(1.72, "walking", 0.91));
    persons.push(createPerson(1.65, "riding", 0.84));
    activities.push({ action: "riding bicycle", confidence: 0.91 });
    activities.push({ action: "driving vehicle", confidence: 0.85 });
    activities.push({ action: "walking", confidence: 0.82 });
  }

  return {
    persons,
    activities,
    risk_level: risk,
    person_count: persons.length,
    model_ready: true,
  };
}

// ── Mock: /activity_status ────────────────────────────────────
function mockActivityStatus() {
  const d = mockDetections();
  const active = d.activities.length > 0 || d.persons.length > 0;
  return {
    active,
    risk_level: "LOW",
    last_event: active ? new Date().toISOString() : null,
    last_detections: active && d.activities.length > 0
      ? [{ class: d.activities[0].action, class_name: d.activities[0].action, confidence: 0.92 }]
      : [],
  };
}

// ── Mock: /events ─────────────────────────────────────────────
const eventCache = [];
let lastEventAction = null;

function generateEvent() {
  const d = mockDetections();
  if (d.activities.length === 0) return null;
  
  const action = d.activities[0].action;
  // Only generate a new event if the action changes (e.g. walking -> cycling -> driving)
  if (action === lastEventAction) return null;
  lastEventAction = action;

  eventCounter++;
  const person = d.persons.length > 0 ? d.persons[0] : null;
  
  return {
    _id: `evt_${eventCounter}`,
    timestamp: Date.now() / 1000,
    risk_level: "LOW",
    risk_score: 1,
    action: action,
    height_m: person ? person.height_m : null,
    camera_name: "Main Entrance",
    faces: person ? [{ name: "unknown", confidence: 0 }] : [],
    detections: [{ class: person ? "person" : "vehicle", confidence: 0.9 }],
  };
}

// Seed initial event
const initialEvent = generateEvent();
if (initialEvent) eventCache.push(initialEvent);

// Check video timeline frequently for event changes
setInterval(() => {
  const e = generateEvent();
  if (e) {
    eventCache.unshift(e); // prepend to show newest first in /events
    if (eventCache.length > 30) eventCache.pop();
  }
}, 500);

function mockEvents() {
  return [...eventCache]; // already sorted newest first
}

// ── Mock: /stats ──────────────────────────────────────────────
function mockStats() {
  return {
    events_last_hour: eventCache.filter(
      (e) => e.timestamp > Date.now() / 1000 - 3600
    ).length,
    events_today: eventCache.length + Math.floor(rand(20, 80)),
  };
}

// ── Mock: /api/cameras ────────────────────────────────────────
function mockCameras() {
  return [
    {
      id: 0,
      name: "Main Entrance",
      source: "rtsp://demo/cam0",
      type: "USB",
      status: "online",
    },
  ];
}

// ── Mock: /api/system/info ────────────────────────────────────
function mockSystemInfo() {
  return {
    demo_mode: true,
    models: {
      pose: "YOLOv8n-Pose (COCO 17-keypoint)",
      activity: "SlowFast R50 (Kinetics-400)",
      depth: "MiDaS v3.1",
      face: "ArcFace (DeepFace)",
    },
    features: [
      "Real-time pose estimation & action recognition",
      "Precision height measurement (±3-5cm)",
      "Monocular depth estimation",
      "Face recognition & enrollment",
      "Multi-camera support (USB/IP/RTSP)",
    ],
  };
}

// ── Intercept fetch ───────────────────────────────────────────
const _originalFetch = window.fetch;

export function installMockAPI() {
  console.log("🎬 Mock API installed — demo mode active");

  window.fetch = async function (url, options) {
    const path = typeof url === "string" ? url : url?.url || "";

    // Only intercept our API routes
    if (path.includes("/api/detections")) {
      return jsonResponse(mockDetections());
    }
    if (path.includes("/activity_status")) {
      return jsonResponse(mockActivityStatus());
    }
    if (path.includes("/events")) {
      return jsonResponse(mockEvents());
    }
    if (path.includes("/stats")) {
      return jsonResponse(mockStats());
    }
    if (path.includes("/api/cameras") && !path.includes("/feed")) {
      return jsonResponse(mockCameras());
    }
    if (path.includes("/api/system/info")) {
      return jsonResponse(mockSystemInfo());
    }
    if (path.includes("/health")) {
      return jsonResponse({ status: "ok", demo_mode: true });
    }
    if (path.includes("/config")) {
      return jsonResponse({});
    }

    // Pass through everything else (real fetch)
    return _originalFetch.apply(this, arguments);
  };
}

function jsonResponse(data) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
