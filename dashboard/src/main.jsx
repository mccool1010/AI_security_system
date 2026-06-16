import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { installMockAPI } from './mockApi.js'

// Auto-detect: if backend is unreachable, activate mock API for demo
// NEVER run the mock API locally. Only run it in production (Vercel)
if (!import.meta.env.DEV) {
  fetch("/api/health")
    .then((res) => {
      if (!res.ok) throw new Error("Backend offline");
      console.log("Backend online — using real API");
    })
    .catch(() => {
      console.warn("Backend unreachable — installing Mock API for portfolio demo");
      installMockAPI();
    });
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
