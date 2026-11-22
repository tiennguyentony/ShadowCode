import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { expect, it } from 'vitest'

const WEB_ROOT = fileURLToPath(new URL('../../../', import.meta.url))

it(
  'vector generator --check matches Format_Document',
  () => {
    const r = spawnSync(process.execPath, ['scripts/gen-message-vectors.mjs', '--check'], {
      cwd: WEB_ROOT,
      timeout: 10_000,
      encoding: 'utf8',
    })
    const report = `status=${r.status} signal=${r.signal} error=${r.error?.message}\nstdout:\n${r.stdout}\nstderr:\n${r.stderr}`
    if (r.error || r.signal || r.status !== 0) console.error(report)
    expect(r.error, report).toBeUndefined()
    expect(r.signal, report).toBeNull()
    expect(r.status, report).toBe(0)
  },
  15_000,
)
