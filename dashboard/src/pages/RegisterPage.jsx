import { useRef, useState, useEffect, useCallback } from "react";
import * as faceapi from "face-api.js";

// ── Constants ────────────────────────────────────────────────────────
const OVAL_RX_RATIO = 0.32; // oval width  as fraction of container width
const OVAL_RY_RATIO = 0.42; // oval height as fraction of container height
const REQUIRED_CAPTURES = 5;
const SCAN_DURATION_MS = 4000; // total progress-ring sweep time
const DETECTION_INTERVAL_MS = 150;
const MIN_FACE_RATIO = 0.06; // face must be ≥ 6% of frame (relaxed for laptop cams)
const MAX_FACE_RATIO = 0.85;

// ── Quality thresholds ───────────────────────────────────────────────
const QUALITY = {
  centerTolerance: 0.35, // fraction of oval dimension for centering (relaxed for laptops)
  minBrightness: 30,
  maxBrightness: 235,
  minSharpness: 3, // very low — laptop webcams are noisy
};

// ── Phase enum ─────────────────────────────────────────────────
const PHASE = {
  IDLE: "idle",
  POSITIONING: "positioning",
  SCANNING: "scanning",
  REVIEW: "review",
  SUBMITTING: "submitting",
  DONE: "done",
};

export default function RegisterPage() {
  // ── Refs ─────────────────────────────────────────────────────
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const detectionLoop = useRef(null);
  const scanTimer = useRef(null);
  const scanStart = useRef(null);
  const capturedDuringRun = useRef([]);
  const phaseRef = useRef(PHASE.IDLE); // shadow phase state for use inside intervals

  // ── State ────────────────────────────────────────────────────
  const [name, setName] = useState("");
  const [phase, _setPhase] = useState(PHASE.IDLE);
  const setPhase = (p) => { phaseRef.current = p; _setPhase(p); };
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [feedback, setFeedback] = useState(""); // positioning prompt
  const [progress, setProgress] = useState(0); // 0-100
  const [captured, setCaptured] = useState([]); // base64 thumbnails
  const [qualityChecks, setQualityChecks] = useState({ face: false, center: false, brightness: false, sharpness: false });
  const [statusMsg, setStatusMsg] = useState(null);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState(null);

  // ── Load face-api models on mount ───────────────────────────
  useEffect(() => {
    async function loadModels() {
      const MODEL_URL = "/models";
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
        ]);
        setModelsLoaded(true);
      } catch (e) {
        console.error("Failed to load face-api models:", e);
        setStatusMsg({ ok: false, text: "Failed to load face detection models." });
      }
    }
    loadModels();
  }, []);

  // ── Enumerate cameras ──────────────────────────────────────
  useEffect(() => {
    async function loadDevices() {
      try {
        // need a temp stream to get permission & labels
        const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
        tmp.getTracks().forEach((t) => t.stop());
        const list = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = list.filter((d) => d.kind === "videoinput");
        setDevices(videoInputs);
        const front = videoInputs.find((d) => /front|user/i.test(d.label));
        if (front) setSelectedDeviceId(front.deviceId);
        else if (videoInputs.length === 1) setSelectedDeviceId(videoInputs[0].deviceId);
      } catch (_) { }
    }
    loadDevices();
  }, []);

  // ── Cleanup on unmount ──────────────────────────────────────
  useEffect(() => {
    return () => {
      stopEverything();
    };
  }, []);

  // ── Helper: stop camera + loops ─────────────────────────────
  function stopEverything() {
    clearInterval(detectionLoop.current);
    clearInterval(scanTimer.current);
    try {
      const s = videoRef.current?.srcObject;
      if (s) { s.getTracks().forEach((t) => t.stop()); videoRef.current.srcObject = null; }
    } catch (_) { }
    // try to resume backend camera
    fetch("/camera/resume", { method: "POST" }).catch(() => { });
  }

  // ── Start camera ────────────────────────────────────────────
  async function startCamera() {
    // pause backend camera so browser can use it
    try {
      await fetch("/camera/pause", { method: "POST" });
    } catch (_) { }

    const constraints = { audio: false };
    if (selectedDeviceId) {
      constraints.video = { deviceId: { exact: selectedDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } };
    } else {
      constraints.video = { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } };
    }

    // Retry up to 4 times with increasing delays — the backend thread
    // may take a few seconds to actually release the camera device
    const MAX_RETRIES = 4;
    let lastError = null;
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      const delayMs = 1000 + attempt * 1000; // 1s, 2s, 3s, 4s
      await new Promise((r) => setTimeout(r, delayMs));
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        console.log(`Camera opened on attempt ${attempt + 1}`);
        return; // success
      } catch (e) {
        lastError = e;
        console.warn(`Camera attempt ${attempt + 1}/${MAX_RETRIES} failed: ${e.name} - ${e.message}`);
      }
    }
    throw lastError; // all retries failed
  }

  // ── Image quality helpers (browser-side) ────────────────────
  function estimateBrightness(imageData) {
    const d = imageData.data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 16) { // sample every 4th pixel
      sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    }
    return sum / (d.length / 16);
  }

  function estimateSharpness(ctx, x, y, w, h) {
    const gray = ctx.getImageData(x, y, w, h);
    const d = gray.data;
    const width = w;
    let lap = 0, count = 0;
    for (let row = 1; row < h - 1; row++) {
      for (let col = 1; col < width - 1; col++) {
        const idx = (row * width + col) * 4;
        const val = 0.299 * d[idx] + 0.587 * d[idx + 1] + 0.114 * d[idx + 2];
        const top = 0.299 * d[((row - 1) * width + col) * 4] + 0.587 * d[((row - 1) * width + col) * 4 + 1] + 0.114 * d[((row - 1) * width + col) * 4 + 2];
        const bot = 0.299 * d[((row + 1) * width + col) * 4] + 0.587 * d[((row + 1) * width + col) * 4 + 1] + 0.114 * d[((row + 1) * width + col) * 4 + 2];
        const lft = 0.299 * d[(row * width + col - 1) * 4] + 0.587 * d[(row * width + col - 1) * 4 + 1] + 0.114 * d[(row * width + col - 1) * 4 + 2];
        const rgt = 0.299 * d[(row * width + col + 1) * 4] + 0.587 * d[(row * width + col + 1) * 4 + 1] + 0.114 * d[(row * width + col + 1) * 4 + 2];
        lap += Math.abs(top + bot + lft + rgt - 4 * val);
        count++;
      }
    }
    return count > 0 ? lap / count : 0;
  }

  // ── Oval geometry ───────────────────────────────────────────
  function getOvalGeom() {
    const container = overlayRef.current;
    if (!container) return { cx: 320, cy: 240, rx: 100, ry: 130, w: 640, h: 480 };
    const w = container.clientWidth, h = container.clientHeight;
    return { cx: w / 2, cy: h / 2, rx: w * OVAL_RX_RATIO, ry: h * OVAL_RY_RATIO, w, h };
  }

  // ── Evaluate face position relative to oval ─────────────────
  function evaluatePosition(detection) {
    const { cx, cy, rx, ry, w, h } = getOvalGeom();
    const box = detection.detection.box;
    const faceCx = box.x + box.width / 2;
    const faceCy = box.y + box.height / 2;
    const faceRatio = (box.width * box.height) / (w * h);

    const checks = { face: true, center: false, brightness: false, sharpness: false };
    let prompt = "";

    // Size check
    if (faceRatio < MIN_FACE_RATIO) {
      prompt = "Move closer";
      return { ok: false, prompt, checks };
    }
    if (faceRatio > MAX_FACE_RATIO) {
      prompt = "Move back";
      return { ok: false, prompt, checks };
    }

    // Center check
    const dx = Math.abs(faceCx - cx) / rx;
    const dy = Math.abs(faceCy - cy) / ry;
    if (dx > QUALITY.centerTolerance || dy > QUALITY.centerTolerance) {
      if (faceCx < cx - rx * QUALITY.centerTolerance) prompt = "Move right";
      else if (faceCx > cx + rx * QUALITY.centerTolerance) prompt = "Move left";
      else if (faceCy < cy - ry * QUALITY.centerTolerance) prompt = "Move down";
      else prompt = "Move up";
      return { ok: false, prompt, checks };
    }
    checks.center = true;

    // Brightness (from canvas)
    try {
      const ctx = canvasRef.current.getContext("2d");
      const imgData = ctx.getImageData(
        Math.max(0, Math.round(box.x)),
        Math.max(0, Math.round(box.y)),
        Math.round(box.width),
        Math.round(box.height)
      );
      const brightness = estimateBrightness(imgData);
      checks.brightness = brightness >= QUALITY.minBrightness && brightness <= QUALITY.maxBrightness;
      if (!checks.brightness) {
        prompt = brightness < QUALITY.minBrightness ? "Need more light" : "Too bright";
        return { ok: false, prompt, checks };
      }
    } catch (_) {
      checks.brightness = true; // skip if error
    }

    // Sharpness
    try {
      const ctx = canvasRef.current.getContext("2d");
      const sharp = estimateSharpness(ctx, Math.max(0, Math.round(box.x)), Math.max(0, Math.round(box.y)), Math.round(box.width), Math.round(box.height));
      checks.sharpness = sharp >= QUALITY.minSharpness;
      if (!checks.sharpness) {
        prompt = "Hold still";
        return { ok: false, prompt, checks };
      }
    } catch (_) {
      checks.sharpness = true;
    }

    return { ok: true, prompt: "Hold still…", checks };
  }

  // ── Capture a single frame ──────────────────────────────────
  function captureFrame() {
    const video = videoRef.current;
    if (!video) return null;
    const c = canvasRef.current;
    c.width = video.videoWidth || 640;
    c.height = video.videoHeight || 480;
    const ctx = c.getContext("2d");
    ctx.drawImage(video, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.92);
  }

  // ── Begin registration ──────────────────────────────────────
  async function beginRegistration() {
    if (!name.trim()) {
      setStatusMsg({ ok: false, text: "Please enter a name first." });
      return;
    }
    if (!modelsLoaded) {
      setStatusMsg({ ok: false, text: "Face detection models still loading…" });
      return;
    }
    setStatusMsg(null);
    setCaptured([]);
    capturedDuringRun.current = [];
    setProgress(0);
    setFeedback("Waiting for camera…");
    setPhase(PHASE.POSITIONING);

    try {
      await startCamera();
    } catch (e) {
      console.error("Camera access failed:", e);
      let msg = `Cannot access camera (${e?.name || "unknown error"}).`;
      if (e?.name === "NotAllowedError") msg = "Camera permission denied. Please allow camera access.";
      if (e?.name === "NotReadableError" || e?.name === "TrackStartError")
        msg = "Camera is in use. Please close other apps or stop the backend camera first.";
      if (e?.name === "AbortError") msg = "Camera access was aborted. Try again.";
      if (e?.name === "NotFoundError") msg = "No camera found on this device.";
      setStatusMsg({ ok: false, text: msg });
      setPhase(PHASE.IDLE);
      return;
    }

    startDetectionLoop();
  }

  // ── Real-time detection loop ────────────────────────────────
  function startDetectionLoop() {
    let allGoodSince = null;

    detectionLoop.current = setInterval(async () => {
      const video = videoRef.current;
      if (!video || video.paused || video.ended) return;

      // draw current frame to canvas for quality analysis
      const c = canvasRef.current;
      c.width = video.videoWidth || 640;
      c.height = video.videoHeight || 480;
      const ctx = c.getContext("2d");
      ctx.drawImage(video, 0, 0, c.width, c.height);

      const detection = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.35 }))
        .withFaceLandmarks(true);

      if (!detection) {
        setFeedback("Position your face in the oval");
        setQualityChecks({ face: false, center: false, brightness: false, sharpness: false });
        allGoodSince = null;
        return;
      }

      const eval_ = evaluatePosition(detection);
      setQualityChecks(eval_.checks);
      setFeedback(eval_.prompt);

      if (eval_.ok) {
        if (!allGoodSince) allGoodSince = Date.now();
        // require face to be good for 300ms before starting scan
        if (Date.now() - allGoodSince > 300 && phaseRef.current !== PHASE.SCANNING) {
          // ─── Begin scanning phase ──────────────────────
          clearInterval(detectionLoop.current);
          startScanPhase();
        }
      } else {
        allGoodSince = null;
      }
    }, DETECTION_INTERVAL_MS);
  }

  // ── Scan phase: progress ring + auto-capture ────────────────
  function startScanPhase() {
    setPhase(PHASE.SCANNING);
    setFeedback("Hold still…");
    scanStart.current = Date.now();
    capturedDuringRun.current = [];
    let captureIndex = 0;
    const captureInterval = SCAN_DURATION_MS / REQUIRED_CAPTURES;

    scanTimer.current = setInterval(async () => {
      const elapsed = Date.now() - scanStart.current;
      const pct = Math.min(100, (elapsed / SCAN_DURATION_MS) * 100);
      setProgress(pct);

      // capture at evenly spaced intervals
      if (captureIndex < REQUIRED_CAPTURES && elapsed >= captureInterval * captureIndex) {
        captureIndex++; // increment immediately to avoid double-capture

        const video = videoRef.current;
        if (!video) return;

        // Draw full frame to main canvas
        const c = canvasRef.current;
        c.width = video.videoWidth || 640;
        c.height = video.videoHeight || 480;
        const ctx = c.getContext("2d");
        ctx.drawImage(video, 0, 0, c.width, c.height);

        // Detect face in current frame to get crop coordinates
        try {
          const detection = await faceapi
            .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.35 }));

          if (detection) {
            const box = detection.box;
            // Add 40% padding around detected face
            const pad = 0.4;
            const px = box.width * pad, py = box.height * pad;
            const cropX = Math.max(0, Math.round(box.x - px));
            const cropY = Math.max(0, Math.round(box.y - py));
            const cropW = Math.min(c.width - cropX, Math.round(box.width + px * 2));
            const cropH = Math.min(c.height - cropY, Math.round(box.height + py * 2));

            // Create a temporary canvas for the cropped face
            const cropCanvas = document.createElement("canvas");
            cropCanvas.width = cropW;
            cropCanvas.height = cropH;
            const cropCtx = cropCanvas.getContext("2d");
            cropCtx.drawImage(c, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

            const croppedDataUrl = cropCanvas.toDataURL("image/jpeg", 0.92);
            capturedDuringRun.current.push(croppedDataUrl);
            setCaptured([...capturedDuringRun.current]);
            console.log(`Capture ${capturedDuringRun.current.length}: cropped face ${cropW}x${cropH}`);
          } else {
            // Fallback: capture full frame if detection failed this frame
            const frame = c.toDataURL("image/jpeg", 0.92);
            capturedDuringRun.current.push(frame);
            setCaptured([...capturedDuringRun.current]);
            console.log(`Capture ${capturedDuringRun.current.length}: full frame (no face detected)`);
          }
        } catch (e) {
          // Fallback on error
          const frame = c.toDataURL("image/jpeg", 0.92);
          capturedDuringRun.current.push(frame);
          setCaptured([...capturedDuringRun.current]);
        }
      }

      if (elapsed >= SCAN_DURATION_MS) {
        clearInterval(scanTimer.current);
        setProgress(100);
        setFeedback("✓ Face captured!");
        setPhase(PHASE.REVIEW);
      }
    }, 50);
  }

  // ── Submit to backend ───────────────────────────────────────
  async function handleSubmit() {
    if (captured.length === 0) return;
    setPhase(PHASE.SUBMITTING);
    setStatusMsg({ ok: null, text: "Registering…" });
    try {
      const res = await fetch("/api/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim(), images_base64: captured }),
      });
      const j = await res.json().catch(() => null);
      if (!res.ok) throw j || { error: `HTTP ${res.status}` };
      setPhase(PHASE.DONE);
      setStatusMsg({ ok: true, text: `✓ ${j.name} registered (${j.samples} samples${j.rejected ? `, ${j.rejected} rejected` : ""})` });
      stopEverything();
    } catch (err) {
      setStatusMsg({ ok: false, text: err?.error || String(err) });
      setPhase(PHASE.REVIEW);
    }
  }

  // ── Redo ────────────────────────────────────────────────────
  function handleRedo() {
    setCaptured([]);
    capturedDuringRun.current = [];
    setProgress(0);
    setStatusMsg(null);
    setPhase(PHASE.POSITIONING);
    startDetectionLoop();
  }

  // ── Cancel ──────────────────────────────────────────────────
  function handleCancel() {
    stopEverything();
    setCaptured([]);
    setProgress(0);
    setFeedback("");
    setStatusMsg(null);
    setPhase(PHASE.IDLE);
  }

  // ═════════════════════════════════════════════
  //  RENDER
  // ═════════════════════════════════════════════
  const showCamera = [PHASE.POSITIONING, PHASE.SCANNING, PHASE.REVIEW].includes(phase);
  const { cx, cy, rx, ry, w: ovalW, h: ovalH } = getOvalGeom();

  // SVG progress ring params (slightly larger than oval)
  const ringRx = rx + 8, ringRy = ry + 8;
  const circumference = Math.PI * 2 * Math.max(ringRx, ringRy); // approximate
  const dashOffset = circumference - (progress / 100) * circumference;

  return (
    <div className="min-h-full flex flex-col items-center py-6 px-4">
      {/* ── Title ──────────────────────────────────────────── */}
      <h1 className="text-2xl font-bold mb-1 tracking-tight">Face Registration</h1>
      <p className="text-sm text-gray-400 mb-6">
        {phase === PHASE.IDLE && "Enter your name and begin to register your face."}
        {phase === PHASE.POSITIONING && "Position your face inside the oval."}
        {phase === PHASE.SCANNING && "Hold still while we capture your face…"}
        {phase === PHASE.REVIEW && "Review your captures, then register or redo."}
        {phase === PHASE.SUBMITTING && "Submitting…"}
        {phase === PHASE.DONE && "Registration complete!"}
      </p>

      {/* ── Name input + Begin button (idle phase) ─────────── */}
      {phase === PHASE.IDLE && (
        <div className="w-full max-w-sm space-y-4">
          <div>
            <label className="block text-sm text-gray-300 mb-1">Name</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter your name"
              className="w-full px-4 py-3 rounded-xl bg-gray-800 border border-gray-700 text-white text-lg placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500/60 transition"
            />
          </div>

          {devices.length > 1 && (
            <div>
              <label className="block text-sm text-gray-300 mb-1">Camera</label>
              <select
                value={selectedDeviceId || ""}
                onChange={(e) => setSelectedDeviceId(e.target.value || null)}
                className="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-200 text-sm"
              >
                <option value="">Auto (front camera)</option>
                {devices.map((d) => (
                  <option key={d.deviceId} value={d.deviceId}>
                    {d.label || `Camera ${d.deviceId.slice(0, 8)}`}
                  </option>
                ))}
              </select>
            </div>
          )}

          <button
            onClick={beginRegistration}
            disabled={!modelsLoaded}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold text-lg hover:from-blue-500 hover:to-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-900/30"
          >
            {modelsLoaded ? "Begin Registration" : "Loading models…"}
          </button>
        </div>
      )}

      {/* ── Camera + Oval overlay ──────────────────────────── */}
      {showCamera && (
        <div
          ref={overlayRef}
          className="relative w-[480px] h-[380px] max-w-full rounded-2xl overflow-hidden bg-black mx-auto"
          style={{ aspectRatio: "4/3" }}
        >
          {/* Live video */}
          <video
            ref={videoRef}
            playsInline
            muted
            className="absolute inset-0 w-full h-full object-cover"
            style={{ transform: "scaleX(-1)" }}
          />

          {/* Dark overlay mask with oval cutout */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 480 380`}
            preserveAspectRatio="none"
          >
            <defs>
              <mask id="ovalMask">
                <rect width="480" height="380" fill="white" />
                <ellipse cx="240" cy="190" rx={480 * OVAL_RX_RATIO} ry={380 * OVAL_RY_RATIO} fill="black" />
              </mask>
            </defs>
            {/* dark overlay */}
            <rect width="480" height="380" fill="rgba(0,0,0,0.65)" mask="url(#ovalMask)" />

            {/* Oval border */}
            <ellipse
              cx="240"
              cy="190"
              rx={480 * OVAL_RX_RATIO}
              ry={380 * OVAL_RY_RATIO}
              fill="none"
              stroke={phase === PHASE.SCANNING ? "#3b82f6" : (qualityChecks.face && qualityChecks.center) ? "#22c55e" : "rgba(255,255,255,0.4)"}
              strokeWidth="2.5"
              style={{ transition: "stroke 0.3s" }}
            />

            {/* Progress ring (scanning phase) */}
            {phase === PHASE.SCANNING && (
              <ellipse
                cx="240"
                cy="190"
                rx={480 * OVAL_RX_RATIO + 5}
                ry={380 * OVAL_RY_RATIO + 5}
                fill="none"
                stroke="#3b82f6"
                strokeWidth="4"
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={dashOffset}
                style={{
                  transition: "stroke-dashoffset 0.1s linear",
                  transform: "rotate(-90deg)",
                  transformOrigin: "240px 190px",
                  filter: "drop-shadow(0 0 6px rgba(59,130,246,0.5))",
                }}
              />
            )}
          </svg>

          {/* Feedback text overlay */}
          <div className="absolute bottom-6 left-0 right-0 flex flex-col items-center z-10">
            <span
              className={`px-4 py-1.5 rounded-full text-sm font-medium backdrop-blur-md transition-all duration-300 ${phase === PHASE.SCANNING
                ? "bg-blue-500/25 text-blue-200 border border-blue-400/30"
                : qualityChecks.face && qualityChecks.center
                  ? "bg-emerald-500/20 text-emerald-200 border border-emerald-400/30"
                  : "bg-black/40 text-white/80 border border-white/10"
                }`}
            >
              {feedback || "Position your face in the oval"}
            </span>
          </div>

          {/* Quality indicators */}
          {(phase === PHASE.POSITIONING || phase === PHASE.SCANNING) && (
            <div className="absolute top-4 right-4 flex flex-col gap-1.5 z-10">
              {[
                { key: "face", label: "Face" },
                { key: "center", label: "Position" },
                { key: "brightness", label: "Light" },
                { key: "sharpness", label: "Focus" },
              ].map(({ key, label }) => (
                <div key={key} className="flex items-center gap-1.5">
                  <div
                    className={`w-2 h-2 rounded-full transition-colors duration-300 ${qualityChecks[key] ? "bg-emerald-400 shadow-[0_0_4px_rgba(34,197,94,0.5)]" : "bg-gray-500/60"
                      }`}
                  />
                  <span className="text-[10px] text-gray-300/80">{label}</span>
                </div>
              ))}
            </div>
          )}

          {/* ✓ Complete overlay */}
          {phase === PHASE.REVIEW && (
            <div className="absolute inset-0 flex items-center justify-center z-10 bg-black/20">
              <div className="w-16 h-16 rounded-full bg-emerald-500/20 border-2 border-emerald-400 flex items-center justify-center animate-[scaleIn_0.3s_ease-out]">
                <svg className="w-8 h-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Hidden canvas for capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* ── Progress bar (below camera during scanning) ────── */}
      {phase === PHASE.SCANNING && (
        <div className="w-[480px] max-w-full mt-3">
          <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full transition-all duration-100"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-gray-500 text-center mt-1">{Math.round(progress)}% — Capturing {captured.length}/{REQUIRED_CAPTURES}</p>
        </div>
      )}

      {/* ── Capture thumbnails ─────────────────────────────── */}
      {captured.length > 0 && (
        <div className="mt-4 w-[480px] max-w-full">
          <p className="text-xs text-gray-500 mb-2">{captured.length} capture{captured.length !== 1 ? "s" : ""}</p>
          <div className="flex gap-2 justify-center">
            {captured.map((src, i) => (
              <div key={i} className="w-16 h-16 rounded-lg overflow-hidden border border-gray-700 bg-gray-800">
                <img src={src} alt={`capture-${i}`} className="w-full h-full object-cover" style={{ transform: "scaleX(-1)" }} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Action buttons ─────────────────────────────────── */}
      {phase === PHASE.REVIEW && (
        <div className="mt-5 flex gap-3">
          <button
            onClick={handleSubmit}
            className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-green-600 text-white font-semibold hover:from-emerald-500 hover:to-green-500 transition-all shadow-lg shadow-emerald-900/20"
          >
            Register
          </button>
          <button
            onClick={handleRedo}
            className="px-6 py-2.5 rounded-xl bg-gray-700 text-gray-200 font-medium hover:bg-gray-600 transition-all"
          >
            Redo
          </button>
          <button
            onClick={handleCancel}
            className="px-6 py-2.5 rounded-xl bg-gray-800 text-gray-400 font-medium hover:bg-gray-700 hover:text-gray-200 transition-all"
          >
            Cancel
          </button>
        </div>
      )}

      {(phase === PHASE.POSITIONING || phase === PHASE.SCANNING) && (
        <div className="mt-4">
          <button
            onClick={handleCancel}
            className="px-5 py-2 rounded-lg bg-gray-800 text-gray-400 text-sm hover:bg-gray-700 hover:text-gray-200 transition-all"
          >
            Cancel
          </button>
        </div>
      )}

      {/* ── Status messages ────────────────────────────────── */}
      {statusMsg && (
        <div className={`mt-5 px-5 py-3 rounded-xl text-sm max-w-sm text-center ${statusMsg.ok === true ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
          : statusMsg.ok === false ? "bg-red-500/15 text-red-300 border border-red-500/30"
            : "bg-gray-800 text-gray-300 border border-gray-700"
          }`}>
          {statusMsg.text}
        </div>
      )}

      {/* ── Done: register another ─────────────────────────── */}
      {phase === PHASE.DONE && (
        <button
          onClick={() => { setPhase(PHASE.IDLE); setName(""); setCaptured([]); setProgress(0); setStatusMsg(null); }}
          className="mt-4 px-5 py-2 rounded-lg bg-gray-800 text-gray-300 text-sm hover:bg-gray-700 transition-all"
        >
          Register Another Person
        </button>
      )}
    </div>
  );
}
