import { useEffect, useRef, useState } from "react";
import { DEMO_MODE } from "../config.js";

/** fetch that reports an expired/missing session to AuthGate. */
export async function apiFetch(path, options) {
  const res = await fetch(path, options);
  if (res.status === 401) {
    window.dispatchEvent(new CustomEvent("sv-unauthorized"));
    throw new Error("authentication required");
  }
  return res;
}

export const EMPTY_SNAPSHOT = {
  camera_id: null,
  ts: null,
  persons: [],
  person_count: 0,
  activities: [],
  risk_level: "LOW",
  risk_score: 0,
  risk_reasons: [],
  model_ready: false,
};

/**
 * Live detections and events.
 *
 * Uses the backend's Server-Sent Events stream (/api/stream): detections are
 * pushed as they are produced and events the moment they are stored. Falls back
 * to polling when the stream is unavailable (demo mode, old backend, proxy that
 * buffers responses) and switches back once the stream reconnects.
 */
export function useLiveData({ onEvent } = {}) {
  const [snapshots, setSnapshots] = useState({});
  const [events, setEvents] = useState([]);
  const [transport, setTransport] = useState(DEMO_MODE ? "polling" : "connecting");
  const [connected, setConnected] = useState(false);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  // initial event list (and periodic refresh while polling)
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await apiFetch("/api/events?limit=20");
        if (!res.ok) throw new Error(res.statusText);
        const data = await res.json();
        if (!cancelled) setEvents(data);
      } catch {
        /* retried below */
      }
    };
    load();
    if (transport !== "stream") {
      const id = setInterval(load, 2000);
      return () => {
        cancelled = true;
        clearInterval(id);
      };
    }
    return () => {
      cancelled = true;
    };
  }, [transport]);

  // stream
  useEffect(() => {
    if (DEMO_MODE || typeof EventSource === "undefined") {
      setTransport("polling");
      return undefined;
    }
    let failures = 0;
    const es = new EventSource("/api/stream");
    es.onopen = () => {
      failures = 0;
      setConnected(true);
      setTransport("stream");
    };
    es.onerror = () => {
      failures += 1;
      setConnected(false);
      if (failures >= 2) setTransport("polling"); // EventSource keeps retrying on its own
    };
    es.addEventListener("detections", (e) => {
      const snap = JSON.parse(e.data);
      setSnapshots((prev) => ({ ...prev, [snap.camera_id]: snap }));
    });
    es.addEventListener("event", (e) => {
      const ev = JSON.parse(e.data);
      setEvents((prev) => [ev, ...prev].slice(0, 50));
      onEventRef.current?.(ev);
    });
    return () => es.close();
  }, []);

  // polling fallback for detections
  useEffect(() => {
    if (transport !== "polling") return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await apiFetch("/api/detections");
        if (!res.ok) throw new Error(res.statusText);
        const snap = await res.json();
        if (!cancelled) {
          setConnected(true);
          setSnapshots((prev) => ({ ...prev, [snap.camera_id || "cam_local"]: snap }));
        }
      } catch {
        if (!cancelled) setConnected(false);
      }
    };
    poll();
    const id = setInterval(poll, 500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [transport]);

  return { snapshots, events, transport, connected };
}

/** Backend health for status indicators. */
export function useBackendStatus(intervalMs = 5000) {
  const [status, setStatus] = useState({ online: null, mongo: null, perf: null });
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [s, p] = await Promise.all([
          apiFetch("/api/status").then((r) => (r.ok ? r.json() : Promise.reject(r))),
          apiFetch("/api/perf").then((r) => (r.ok ? r.json() : null)).catch(() => null),
        ]);
        if (!cancelled) setStatus({ online: true, mongo: s.mongo?.ok ?? null, camera: s.camera_open, perf: p });
      } catch {
        if (!cancelled) setStatus({ online: false, mongo: null, perf: null });
      }
    };
    load();
    const id = setInterval(load, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs]);
  return status;
}

export const titleCase = (s) =>
  (s || "").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
