import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      // Relé FastAPI (cv-ide-relay), por defecto puerto 8765.
      '/api': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
      },
      '/ws': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
        ws: true,
      },
    },
  },
})
