// @vitest-environment happy-dom
// Label_Sheets (Requirement 9): print/labels.js loaded into a DOM, as the browser runs it.
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import fc from 'fast-check'
import { beforeAll, describe, expect, it } from 'vitest'
import { PRODUCTS } from '../../config'

interface PlanItem {
  id: number
  copy: number
  of: number
}
interface TilemarkLabelsApi {
  LABEL_W_MM: number
  LABEL_H_MM: number
  DEFAULT_COPIES: number
  MAX_COPIES: number
  clampCopies(value: unknown): number
  planSheets(entries: { id: number; copies: number }[]): PlanItem[][]
  labelSize(): { width: string; height: string }
  renderSheets(doc: Document, container: HTMLElement, plan: PlanItem[][], available: Set<number>): void
}

const FC_SEED = 0x7ace2026
// Paths via node:path: under happy-dom the global URL is not Node's, and readFileSync rejects it.
const PRINT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../../print')
const readPrint = (name: string) => readFileSync(path.join(PRINT, name), 'utf8')
const IDS = PRODUCTS.map((p) => p.id)

let api: TilemarkLabelsApi

beforeAll(() => {
  new Function(readPrint('labels.js'))()
  api = (globalThis as unknown as { TilemarkLabels: TilemarkLabelsApi }).TilemarkLabels
})

function render(copies: number[], available: Set<number>) {
  const container = document.createElement('div')
  const plan = api.planSheets(IDS.map((id, i) => ({ id, copies: copies[i] })))
  api.renderSheets(document, container, plan, available)
  return { container, plan }
}

describe('sheet plan (9.4–9.6)', () => {
  // Feature: race, Property 5: Sheet plan keeps every copy, two per Sheet
  it('Property 5: Sheet plan keeps every copy, two per Sheet', () => {
    console.log(`race Property 5 seed 0x${FC_SEED.toString(16)}`)
    const entries = fc.array(
      fc.record({ id: fc.integer({ min: 1, max: 0xffffffff }), copies: fc.integer({ min: 0, max: 60 }) }),
      { maxLength: 6 },
    )
    fc.assert(
      fc.property(entries, (list) => {
        const sheets = api.planSheets(list)
        const flat = sheets.flat()
        const expected = list.flatMap((e) => Array.from({ length: e.copies }, (_, c) => ({ id: e.id, copy: c + 1, of: e.copies })))
        expect(flat).toEqual(expected)
        const total = expected.length
        expect(sheets.length).toBe(Math.ceil(total / 2))
        sheets.forEach((s, i) => {
          expect(s.length).toBe(i === sheets.length - 1 && total % 2 === 1 ? 1 : 2)
        })
      }),
      { seed: FC_SEED, numRuns: 1000 },
    )
  })

  it('default 6 copies of 3 products: 18 labels on 9 sheets of 2', () => {
    const sheets = api.planSheets(IDS.map((id) => ({ id, copies: api.DEFAULT_COPIES })))
    expect(api.DEFAULT_COPIES).toBe(6)
    expect(sheets).toHaveLength(9)
    expect(sheets.every((s) => s.length === 2)).toBe(true)
  })

  it('9.5 clampCopies keeps integers in [0, 60], default 6', () => {
    expect(api.clampCopies('6')).toBe(6)
    expect(api.clampCopies(0)).toBe(0)
    expect(api.clampCopies('61')).toBe(60)
    expect(api.clampCopies(-3)).toBe(0)
    expect(api.clampCopies('2.6')).toBe(3)
    expect(api.clampCopies('')).toBe(6)
    expect(api.clampCopies('x')).toBe(6)
    expect(api.clampCopies(undefined)).toBe(6)
  })
})

describe('label elements (9.1, 9.2, 9.8, 9.9)', () => {
  it('9.2 every Label element is 114mm x 149mm in CSS millimetres, never pixels', () => {
    expect(api.labelSize()).toEqual({ width: '114mm', height: '149mm' })
    const { container } = render([6, 6, 6], new Set(IDS))
    const labels = [...container.querySelectorAll<HTMLElement>('.label')]
    expect(labels).toHaveLength(18)
    for (const label of labels) {
      expect(label.style.width).toBe('114mm')
      expect(label.style.height).toBe('149mm')
      expect(label.getAttribute('style') ?? '').not.toMatch(/px/)
    }
  })

  it('9.1 images come from labels/<Product_ID>.png', () => {
    const { container } = render([1, 1, 1], new Set(IDS))
    const srcs = [...container.querySelectorAll('img.label')].map((img) => img.getAttribute('src'))
    expect(srcs).toEqual(IDS.map((id) => `labels/${id}.png`))
  })

  it('9.8 a missing image becomes an outline of the same size with its Product_ID', () => {
    const { container } = render([2, 2, 2], new Set([1001, 1003]))
    const outlines = [...container.querySelectorAll<HTMLElement>('.label.outline')]
    expect(outlines).toHaveLength(2)
    for (const o of outlines) {
      expect(o.tagName).toBe('DIV')
      expect(o.style.width).toBe('114mm')
      expect(o.style.height).toBe('149mm')
      expect(o.querySelector('.outline-id')?.textContent).toBe('1002')
    }
    expect(container.querySelectorAll('img.label')).toHaveLength(4)
  })

  it('9.4 two slots per sheet at the design positions; 9.9 caption with ID and copy number', () => {
    const { container } = render([3, 0, 0], new Set(IDS))
    const sheets = [...container.querySelectorAll<HTMLElement>('.sheet')]
    expect(sheets).toHaveLength(2)
    const slots = [...sheets[0].querySelectorAll<HTMLElement>('.slot')]
    expect(slots.map((s) => [s.style.left, s.style.top])).toEqual([
      ['30.5mm', '12mm'],
      ['30.5mm', '136mm'],
    ])
    expect(sheets[1].querySelectorAll('.slot')).toHaveLength(1)
    expect([...container.querySelectorAll('.caption')].map((c) => c.textContent)).toEqual([
      'ID 1001 · copy 1 of 3',
      'ID 1001 · copy 2 of 3',
      'ID 1001 · copy 3 of 3',
    ])
  })

  it('9.7 every sheet has a 100 mm ruler and "print at 100% (actual size)"', () => {
    const { container } = render([5, 6, 7], new Set())
    const sheets = [...container.querySelectorAll('.sheet')]
    expect(sheets).toHaveLength(9)
    for (const sheet of sheets) {
      const ruler = sheet.querySelector('svg.ruler')!
      expect(ruler.getAttribute('width')).toBe('100mm')
      expect(ruler.getAttribute('viewBox')).toBe('0 0 100 8')
      expect(ruler.querySelectorAll('line')).toHaveLength(12) // base line + 11 ticks, 0 to 100 mm
      expect(sheet.querySelector('.actual-size')?.textContent).toBe('print at 100% (actual size)')
    }
  })
})

describe('labels.html (9.3, 9.10–9.12)', () => {
  const html = readPrint('labels.html')

  it('9.3 declares an A4 page and A4 sheets', () => {
    expect(html).toMatch(/@page \{ size: A4; margin: 0; \}/)
    expect(html).toMatch(/\.sheet \{[^}]*width: 210mm; height: 297mm;/)
    expect(html).toMatch(/\.label \{[^}]*width: 114mm; height: 149mm; transform: translate\(-50%, -50%\) rotate\(90deg\);/)
  })

  it('9.10 the size check is screen-only; 9.11 classic scripts only', () => {
    expect(html).toMatch(/@media print \{\s*\.screen-only \{ display: none; \}/)
    expect(html).toMatch(/<div class="screen-only">[\s\S]*id="check"[\s\S]*<\/div>\s*<div id="sheets">/)
    expect(html).toContain('<script src="products.js"></script>')
    expect(html).toContain('<script src="labels.js"></script>')
    expect(html).not.toMatch(/type="module"/)
  })

  it('9.12 no reference to /training', () => {
    expect(html).not.toMatch(/training/)
    expect(readPrint('labels.js')).not.toMatch(/training/)
  })
})
