import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { TouchProvider } from './contexts/TouchContext'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <TouchProvider>
      <App />
    </TouchProvider>
  </StrictMode>,
)
