import { configDefaults, defineConfig } from 'vitest/config'

// `npm test`: everything that runs on a clean checkout. Tests that need the git-ignored
// fixture folders are named *.fixtures.test.ts and run with `npm run test:fixtures`.
export default defineConfig({
  test: {
    environment: 'node',
    exclude: [...configDefaults.exclude, '**/*.fixtures.test.ts'],
  },
})
