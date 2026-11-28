// The Scanner and the app configuration are pure: no UI, camera, DOM or package imports.
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const IMPORT_RE = /(?:import|export)\s[^'"]*?from\s*['"]([^'"]+)['"]|import\s*['"]([^'"]+)['"]|import\s*\(\s*['"]([^'"]+)['"]\s*\)/g

const read = (rel: string) => readFileSync(new URL(rel, import.meta.url), 'utf8')
const specifiers = (src: string) => [...src.matchAll(IMPORT_RE)].map((m) => m[1] ?? m[2] ?? m[3])
const stripComments = (src: string) => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1')

function expectNoBrowserApis(file: string, src: string) {
  const code = stripComments(src)
  for (const word of ['window', 'document', 'navigator', 'getUserMedia']) {
    expect(new RegExp(`\\b${word}\\b`).test(code), `${file}: uses ${word}`).toBe(false)
  }
}

describe('scanner and config isolation', () => {
  it('scanner.ts imports only the DecodeResult type from the codec', () => {
    const src = read('../scanner.ts')
    expect(specifiers(src)).toEqual(['../codec/message'])
    expect(src).toMatch(/import type \{ DecodeResult \} from '\.\.\/codec\/message'/)
    expectNoBrowserApis('scanner.ts', src)
  })

  it('config.ts imports only the Scanner types', () => {
    const src = read('../../config.ts')
    expect(specifiers(src)).toEqual(['./scanner/scanner'])
    expect(src).toMatch(/import type \{ ScannerConfiguration \} from '\.\/scanner\/scanner'/)
    expectNoBrowserApis('config.ts', src)
  })
})
