// Requirement 10: jsQR (Apache-2.0) ships with its license text; qrcode stays a dev dependency.
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const read = (rel: string) => readFileSync(new URL(rel, import.meta.url), 'utf8')
const pkg = JSON.parse(read('../../../package.json')) as {
  dependencies?: Record<string, string>
  devDependencies?: Record<string, string>
}

describe('dependencies and licenses (10.1–10.4)', () => {
  it('public/licenses/jsQR-LICENSE.txt equals the license in node_modules/jsqr', () => {
    const shipped = read('../../../public/licenses/jsQR-LICENSE.txt')
    expect(shipped).toBe(read('../../../node_modules/jsqr/LICENSE'))
    expect(shipped).toMatch(/Apache License\s+Version 2\.0/)
  })

  it('jsqr and onnxruntime-web (tile reader) are the only runtime dependencies; qrcode is dev-only', () => {
    expect(Object.keys(pkg.dependencies ?? {}).sort()).toEqual(['jsqr', 'onnxruntime-web'])
    expect(pkg.devDependencies).toHaveProperty('qrcode')
    expect(pkg.dependencies).not.toHaveProperty('qrcode')
  })

  it('jsqr and qrcode have the licenses listed in the design', () => {
    expect(JSON.parse(read('../../../node_modules/jsqr/package.json')).license).toBe('Apache-2.0')
    expect(JSON.parse(read('../../../node_modules/qrcode/package.json')).license).toBe('MIT')
    expect(JSON.parse(read('../../../node_modules/onnxruntime-web/package.json')).license).toBe('MIT')
  })
})
