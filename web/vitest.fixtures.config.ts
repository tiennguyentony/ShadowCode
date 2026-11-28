import { defineConfig } from 'vitest/config'

// `npm run test:fixtures`: the tests that need training/export/demo-fixtures
// (TILEMARK_DEMO_FIXTURES), the source bundle (TILEMARK_BUNDLE_SRC) or the bundle's
// fixtures/ copied by `npm run copy-bundle`. A missing folder fails with its path.
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.fixtures.test.ts'],
    testTimeout: 120_000,
  },
})
