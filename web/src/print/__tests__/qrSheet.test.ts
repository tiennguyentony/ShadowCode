import { readFileSync } from 'node:fs'
import jsQR from 'jsqr'
import { describe, expect, it } from 'vitest'
import { PRODUCTS } from '../../config'
import { QR_SYMBOL_MM, QUIET_ZONE_MODULES, productsScript, qrMatrix, qrSheetHtml } from '../qrSheet'

const PRINT = new URL('../../../../print/', import.meta.url)
const readPrint = (name: string) => readFileSync(new URL(name, PRINT), 'utf8').replace(/\r\n/g, '\n')
const committedHtml = readPrint('qr-codes.html')

interface ParsedSvg {
  text: string
  modules: number
  moduleMm: number
  quiet: number
  widthMm: number
  heightMm: number
  viewBox: number[]
  dark: boolean[][] // in viewBox modules, Quiet_Zone included
}

/** Reads each <svg> of the page back into a module grid, from its path data. */
function parseSvgs(html: string): ParsedSvg[] {
  return [...html.matchAll(/<svg [^>]*>[\s\S]*?<\/svg>/g)].map(([svg]) => {
    const attr = (name: string) => svg.match(new RegExp(`${name}="([^"]*)"`))![1]
    const viewBox = attr('viewBox').split(' ').map(Number)
    const total = viewBox[2]
    const dark = Array.from({ length: total }, () => Array<boolean>(total).fill(false))
    const d = svg.match(/<path [^>]*d="([^"]*)"/)![1]
    for (const [, x, y, len] of d.matchAll(/M(\d+) (\d+)h(\d+)v1h-\3z/g)) {
      for (let i = 0; i < Number(len); i++) dark[Number(y)][Number(x) + i] = true
    }
    expect(d.replace(/M(\d+) (\d+)h(\d+)v1h-\3z/g, '')).toBe('')
    return {
      text: attr('data-text'),
      modules: Number(attr('data-modules')),
      moduleMm: Number(attr('data-module-mm')),
      quiet: Number(attr('data-quiet-modules')),
      widthMm: parseFloat(attr('width')),
      heightMm: parseFloat(attr('height')),
      viewBox,
      dark,
    }
  })
}

/** Draws a module grid as RGBA pixels, 4 px per module, white = 255. */
function toRgba(dark: boolean[][], px = 4) {
  const size = dark.length * px
  const data = new Uint8ClampedArray(size * size * 4).fill(255)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (dark[Math.floor(y / px)][Math.floor(x / px)]) {
        const i = (y * size + x) * 4
        data[i] = data[i + 1] = data[i + 2] = 0
      }
    }
  }
  return { data, size }
}

const svgs = parseSvgs(committedHtml)

describe('QR_Sheet (Requirement 8)', () => {
  it('8.1 one code per product, in catalog order', () => {
    expect(svgs.map((s) => s.text)).toEqual(PRODUCTS.map((p) => String(p.id)))
  })

  it('8.2, 8.3 each printed code decodes with jsQR to the decimal Product_ID and nothing else', () => {
    for (const svg of svgs) {
      const { data, size } = toRgba(svg.dark)
      const code = jsQR(data, size, size)
      expect(code?.data).toBe(svg.text)
    }
  })

  it('8.2 the drawn modules equal the qrcode matrix', () => {
    for (const svg of svgs) {
      const m = qrMatrix(svg.text)
      const inner = svg.dark.slice(svg.quiet, svg.quiet + m.size).map((row) => row.slice(svg.quiet, svg.quiet + m.size))
      expect(inner).toEqual(m.dark)
    }
  })

  it('8.4 the QR_Symbol is at least 30 mm wide', () => {
    for (const svg of svgs) {
      expect(svg.modules * svg.moduleMm).toBeCloseTo(QR_SYMBOL_MM, 3)
      expect(svg.modules * svg.moduleMm).toBeGreaterThanOrEqual(30)
    }
  })

  it('8.5 a Quiet_Zone of 4 white modules on each side', () => {
    for (const svg of svgs) {
      const total = svg.modules + 2 * QUIET_ZONE_MODULES
      expect(svg.quiet).toBe(4)
      expect(svg.viewBox).toEqual([0, 0, total, total])
      expect(svg.widthMm).toBeCloseTo(total * svg.moduleMm, 3)
      expect(svg.heightMm).toBeCloseTo(total * svg.moduleMm, 3)
      for (let i = 0; i < total; i++) {
        for (let q = 0; q < 4; q++) {
          expect(svg.dark[q][i] || svg.dark[total - 1 - q][i] || svg.dark[i][q] || svg.dark[i][total - 1 - q]).toBe(false)
        }
      }
    }
  })

  it('8.6 name and ID under each code', () => {
    for (const p of PRODUCTS) {
      const cell = committedHtml.match(new RegExp(`<div class="cell" data-product-id="${p.id}">([\\s\\S]*?)\\n      </div>`))![1]
      expect(cell.indexOf('<svg')).toBeLessThan(cell.indexOf(`<div class="name">${p.name}</div>`))
      expect(cell.indexOf(`<div class="name">`)).toBeLessThan(cell.indexOf(`<div class="id">ID ${p.id}</div>`))
    }
  })

  it('8.7 dashed cut lines around cells larger than the code', () => {
    expect(committedHtml.match(/<div class="cell" /g)).toHaveLength(PRODUCTS.length)
    expect(committedHtml).toMatch(/\.cell \{[^}]*width: 80mm; height: 85mm; border: 0\.3mm dashed #000;/)
    for (const svg of svgs) expect(svg.widthMm).toBeLessThan(80)
  })

  it('8.1, 8.8 an A4 page with no scripts or external files', () => {
    expect(committedHtml).toMatch(/@page \{ size: A4; margin: 0; \}/)
    expect(committedHtml).toMatch(/\.sheet \{[^}]*width: 210mm; height: 297mm;/)
    expect(committedHtml).not.toMatch(/<script|<link|src=/)
  })

  it('8.9 print/qr-codes.html and print/products.js equal the generator output', () => {
    expect(committedHtml, 'print/qr-codes.html is out of date: run `npm run print:gen` in /web').toBe(qrSheetHtml(PRODUCTS))
    expect(readPrint('products.js'), 'print/products.js is out of date: run `npm run print:gen` in /web').toBe(
      productsScript(PRODUCTS),
    )
  })

  it('escapes HTML in product names', () => {
    const html = qrSheetHtml([{ id: 7, name: 'A <b> & "c"' }])
    expect(html).toContain('A &#60;b&#62; &#38; &#34;c&#34;')
  })
})
