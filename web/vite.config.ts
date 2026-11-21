import { resolve } from 'node:path'
import { defineConfig } from 'vite'

// Relative base: the static build works from any sub-path
// (for example https://<user>.github.io/<repo>/) without knowing it at build time.
export default defineConfig({
  base: './',
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        measure: resolve(import.meta.dirname, 'measure/index.html'),
      },
    },
  },
})
