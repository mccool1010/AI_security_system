import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api/register': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/api/cameras': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/api/metrics': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/api/screenshots': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/api/detections': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/api/system': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/api/calibrate': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/activity_status': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/camera': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/enroll': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/users': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/events': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/status': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/stats': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/config': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/video_feed': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/set_reference': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
})
