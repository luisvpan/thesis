import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'
import Terminal from 'vite-plugin-terminal'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    Terminal({
      output: ['terminal', 'console']
    }),
  ],
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
})
