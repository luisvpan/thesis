import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'
import Terminal from 'vite-plugin-terminal'

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    tailwindcss(),
    // Terminal plugin solo en desarrollo (usa módulos virtuales que fallan en build)
    mode === 'development' && Terminal({
      output: ['terminal', 'console'],
      console: 'terminal'
    }),
  ].filter(Boolean),
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      // TypeScript API (Elysia) - compile/execute endpoints
      '/api': {
        target: 'http://127.0.0.1:3000',
        changeOrigin: true,
      },
      // Python IDE relay (cv-stack) - vision/touch WebSockets
      '/ws': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
        ws: true,
      },
    },
  },
}))
