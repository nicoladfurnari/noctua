import path from "node:path"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "vite"

// One self-contained bundle (window.NoctuaUI) next to the plain-JS GUI: the build output is committed, so an
// installed Noctua needs no Node. Tailwind is prefixed (tw:) and has no preflight: it must not touch the GUI's CSS.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
  build: {
    outDir: path.resolve(__dirname, "../astroforge/gui/static/ui"),
    emptyOutDir: true,
    cssCodeSplit: false,
    sourcemap: false,
    lib: {
      entry: path.resolve(__dirname, "src/main.tsx"),
      name: "NoctuaUI",
      formats: ["iife"],
      fileName: () => "noctua-ui.js",
      cssFileName: "noctua-ui",
    },
  },
})
