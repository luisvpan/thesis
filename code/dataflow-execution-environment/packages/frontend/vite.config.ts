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
      // Mismo origen en dev: el cliente Eden usa `window.location.origin` y las rutas /api/* llegan a Elysia (puerto 3000).
      '/api': {
        target: 'http://127.0.0.1:3000',
        changeOrigin: true,
      },
      // WebSocket visión (Python → Elysia → navegador)
      '/ws': {
        target: 'http://127.0.0.1:3000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
})
