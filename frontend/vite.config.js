import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/models": "http://localhost:8000",
      "/predict": "http://localhost:8000",
      "/attack": "http://localhost:8000",
      "/tree": "http://localhost:8000",
    },
  },
});
