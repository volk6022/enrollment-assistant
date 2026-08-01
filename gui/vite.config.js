import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// No hardcoded backend URL here: in dev, the WS/HTTP calls go to same-origin
// relative paths (/ws/dialogue, /health, ...) and Vite's dev server proxies
// them to whatever VITE_DEV_BACKEND points at, purely for local development
// convenience. The production nginx container (gui/Dockerfile) does the
// equivalent proxying via envsubst'd BACKEND_HOST/BACKEND_PORT — see
// gui/nginx/default.conf.template. Neither path bakes a backend URL into the
// JS bundle.
const devBackend = process.env.VITE_DEV_BACKEND || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.VITE_DEV_PORT) || 5173,
    proxy: {
      "/ws": {
        target: devBackend.replace(/^http/, "ws"),
        ws: true,
        changeOrigin: true,
      },
      "/health": { target: devBackend, changeOrigin: true },
      "/metrics": { target: devBackend, changeOrigin: true },
      "/answer": { target: devBackend, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
