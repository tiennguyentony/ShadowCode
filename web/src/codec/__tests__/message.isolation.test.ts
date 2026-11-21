// Requirements 13.2-13.4 and 14.2: the codec has no UI, camera, DOM or package imports,
// and the Vector_Generator imports only Node built-ins.
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

const CODEC_DIR = fileURLToPath(new URL('../', import.meta.url))
const WEB_ROOT = fileURLToPath(new URL('../../../', import.meta.url))

const IMPORT_RE = /(?:import|export)\s[^'"]*?from\s*['"]([^'"]+)['"]|import\s*['"]([^'"]+)['"]|import\s*\(\s*['"]([^'"]+)['"]\s*\)/g

function specifiers(source: string): string[] {
  return [...source.matchAll(IMPORT_RE)].map((m) => m[1] ?? m[2] ?? m[3])
}

function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1')
}

describe('module isolation', () => {
  it('message.ts, crc.ts and bch.ts import only from /web/src/codec and use no browser APIs', () => {
    const seen = new Set<string>()
    const queue = ['message.ts', 'crc.ts', 'bch.ts'].map((f) => path.join(CODEC_DIR, f))
    while (queue.length > 0) {
      const file = queue.shift()!
      if (seen.has(file)) continue
      seen.add(file)
      const src = readFileSync(file, 'utf8')
      for (const spec of specifiers(src)) {
        expect(spec.startsWith('.'), `${file}: package import ${spec}`).toBe(true)
        expect(spec.includes('/ui'), `${file}: UI import ${spec}`).toBe(false)
        const target = path.resolve(path.dirname(file), spec)
        expect(path.relative(CODEC_DIR, target).startsWith('..'), `${file}: import outside codec ${spec}`).toBe(false)
        queue.push(target.endsWith('.ts') ? target : `${target}.ts`)
      }
      const code = stripComments(src)
      for (const word of ['window', 'document', 'navigator', 'getUserMedia']) {
        expect(new RegExp(`\\b${word}\\b`).test(code), `${file}: uses ${word}`).toBe(false)
      }
    }
    expect(seen.size).toBe(3)
  })

  it('the Vector_Generator imports only node: built-ins', () => {
    const src = readFileSync(path.join(WEB_ROOT, 'scripts', 'gen-message-vectors.mjs'), 'utf8')
    const specs = specifiers(src)
    expect(specs.length).toBeGreaterThan(0)
    for (const spec of specs) expect(spec.startsWith('node:'), `generator import ${spec}`).toBe(true)
  })
})
