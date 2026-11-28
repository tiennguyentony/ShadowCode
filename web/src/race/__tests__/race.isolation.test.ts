// Race_Logic and its helpers are pure (Requirement 11.1), and product names live
// only in config.ts (Requirement 1.4).
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { PRODUCTS } from '../../config'

const IMPORT_RE = /(?:import|export)\s[^'"]*?from\s*['"]([^'"]+)['"]|import\s*['"]([^'"]+)['"]|import\s*\(\s*['"]([^'"]+)['"]\s*\)/g

const read = (rel: string) => readFileSync(new URL(rel, import.meta.url), 'utf8')
const specifiers = (src: string) => [...src.matchAll(IMPORT_RE)].map((m) => m[1] ?? m[2] ?? m[3])
const stripComments = (src: string) => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1')

const PURE_FILES: Record<string, string[]> = {
  '../race.ts': [],
  '../qrText.ts': [],
  '../frameRate.ts': [],
  '../tileReader.ts': ['../codec/message'],
}

describe('race isolation (11.1)', () => {
  for (const [file, allowed] of Object.entries(PURE_FILES)) {
    it(`${file} imports only ${allowed.join(', ') || 'nothing'} and uses no browser API`, () => {
      const src = read(file)
      expect(specifiers(src)).toEqual(allowed)
      const code = stripComments(src)
      for (const word of ['window', 'document', 'navigator', 'getUserMedia', 'performance']) {
        expect(new RegExp(`\\b${word}\\b`).test(code), `${file}: uses ${word}`).toBe(false)
      }
    })
  }

  it('tileReader.ts imports the codec as a type only', () => {
    expect(read('../tileReader.ts')).toMatch(/import type \{ DecodeResult \} from '\.\.\/codec\/message'/)
  })
})

describe('product names only in config.ts (1.4)', () => {
  const files = ['../race.ts', '../qrText.ts', '../frameRate.ts', '../tileReader.ts', '../../ui/race.ts', '../../../race/index.html']
  for (const file of files) {
    it(`${file} holds no product name`, () => {
      const src = read(file)
      for (const p of PRODUCTS) expect(src.includes(p.name), `${file} contains "${p.name}"`).toBe(false)
      expect(src).not.toMatch(/Product 100\d/)
    })
  }
})
