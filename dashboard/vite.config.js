import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backend = process.env.BACKEND_URL || 'http://127.0.0.1:5000'

// The dashboard only talks to /api/* (including the SSE stream and camera feeds).
// Un-prefixed legacy routes like /events or /camera are NOT proxied: they collide
// with dashboard pages (/events, /cameras) and would break page reloads.
const routes = ['/api', '/video_feed', '/health']

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(routes.map((r) => [r, { target: backend, changeOrigin: true }])),
  },
})
