import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// `vite build --mode static` produces a read-only showcase that reads the JSON
// snapshot written by scripts/export_static.py instead of calling the API.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [react()],
    base: env.VITE_BASE || "/",
    define: { __STATIC__: JSON.stringify(mode === "static" || env.VITE_STATIC === "1") },
    server: {
      port: 5173,
      proxy: { "/api": { target: env.API_URL || "http://127.0.0.1:8000", changeOrigin: true } },
    },
  };
});
