import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// `process` exists at Vite config load time (Node). Declared locally so the
// config type-checks without pulling in @types/node.
declare const process: { env: Record<string, string | undefined> }

// Ports and the backend proxy target come from the environment so many
// instances can run at once without hardcoding a server URL.
const PORT = Number(process.env.PORT ?? 9002)
const BACKEND_PORT = Number(process.env.BACKEND_PORT ?? 9001)
const backend = `http://localhost:${BACKEND_PORT}`

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: PORT,
    strictPort: true,
    host: 'localhost',
    // Requests arrive through the Cloudflare tunnel with a public Host header;
    // allow the whole family domain so Vite does not reject them.
    allowedHosts: ['.oursilverfamily.com', 'localhost'],
    proxy: {
      '/api': { target: backend, changeOrigin: true },
      '/openapi.json': { target: backend, changeOrigin: true },
      '/docs': { target: backend, changeOrigin: true },
    },
  },
})
