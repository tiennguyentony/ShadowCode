// Requirement 9.1: the pure reader files import no UI module and use no browser API.
// Requirement 8.1: WebGL is never used.
import { readdirSync, readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const IMPORT_RE = /(?:import|export)\s[^'"]*?from\s*['"]([^'"]+)['"]|import\s*['"]([^'"]+)['"]|import\s*\(\s*['"]([^'"]+)['"]\s*\)/g
const dir = new URL('../', import.meta.url)
const read = (name: string) => readFileSync(new URL(name, dir), 'utf8')
const specifiers = (src: string) => [...src.matchAll(IMPORT_RE)].map((m) => m[1] ?? m[2] ?? m[3])
const stripComments = (src: string) => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1')
const stripStrings = (src: string) => src.replace(/'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"|`(?:[^`\\]|\\.)*`/g, "''")

const BROWSER_ONLY = new Set(['browser.ts'])
const ALLOWED_PACKAGES = new Set(['onnxruntime-web'])
const sources = readdirSync(dir).filter((f) => f.endsWith('.ts') && !f.endsWith('.d.ts'))

describe('reader isolation (9.1)', () => {
  it('has the expected files', () => {
    for (const f of ['contract.ts', 'flatten.ts', 'frame.ts', 'homography.ts', 'layout.ts', 'locate.ts', 'logitWindow.ts', 'onnx.ts', 'pipeline.ts', 'tags.ts', 'types.ts']) {
      expect(sources).toContain(f)
    }
  })

  it.each(sources.filter((f) => !BROWSER_ONLY.has(f)))('%s imports only reader files, the codec and allowed packages, and uses no browser API', (file) => {
    const src = read(file)
    for (const s of specifiers(src)) {
      const ok = s.startsWith('./') || s === '../codec/message' || ALLOWED_PACKAGES.has(s)
      expect(ok, `${file} imports ${s}`).toBe(true)
      expect(s.includes('/ui/') || s.includes('/race/'), `${file} imports ${s}`).toBe(false)
    }
    const code = stripStrings(stripComments(src))
    for (const word of ['window', 'document', 'navigator', 'getUserMedia', 'fetch', 'HTMLVideoElement', 'OffscreenCanvas']) {
      expect(new RegExp(`\\b${word}\\b`).test(code), `${file}: uses ${word}`).toBe(false)
    }
  })

  it('onnx.ts imports only types from onnxruntime-web', () => {
    expect(read('onnx.ts')).toMatch(/import type \{[^}]*\} from 'onnxruntime-web'/)
    expect(read('onnx.ts')).not.toMatch(/^import \* as ort/m)
  })
})

describe('execution providers (8.1, 8.2)', () => {
  it('never names webgl', () => {
    for (const f of sources) expect(stripComments(read(f)).toLowerCase(), f).not.toContain('webgl')
  })

  it('lists webgpu then wasm', async () => {
    const { EXECUTION_PROVIDERS } = await import('../onnx')
    expect(EXECUTION_PROVIDERS).toEqual(['webgpu', 'wasm'])
  })

  it('browser.ts sets wasmPaths from an absolute URL', () => {
    expect(read('browser.ts')).toMatch(/ort\.env\.wasm\.wasmPaths = \{ wasm: new URL\(jsepWasmUrl, import\.meta\.url\)\.href \}/)
  })
})
