import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { DEMO_MODE } from './config.js'
import { installMockAPI } from './mockApi.js'

if (DEMO_MODE) {
  installMockAPI();
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
