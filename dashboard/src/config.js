// Demo mode serves simulated data from mockApi.js (for the static portfolio deploy).
// It is only ever enabled by an explicit build flag — never as a silent fallback,
// because a security dashboard must not show made-up activity when the backend is down.
export const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === "true";
