import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { PRODUCTS } from '../../config'
import { createQrTextParser } from '../qrText'

const FC_SEED = 0x7ace2026
const CATALOG = PRODUCTS.map((p) => p.id)
const parse = createQrTextParser(CATALOG)

describe('QR_Text_Parser (4.4, 4.5)', () => {
  it('accepts 1001, 1002 and 1003', () => {
    expect(parse('1001')).toBe(1001)
    expect(parse('1002')).toBe(1002)
    expect(parse('1003')).toBe(1003)
  })

  it.each(['1004', '0', ' 1001x', '', '01001', '+1001', '1001 ', ' 1001', '1001.0', '1e3', '１００１', '1001\n', 'Product 1001'])(
    'ignores %j',
    (text) => {
      expect(parse(text)).toBeNull()
    },
  )

  it('never accepts the reserved ID 0, even if the catalog lists it', () => {
    expect(createQrTextParser([0, 1001])('0')).toBeNull()
  })

  // Feature: race, Property 4: The QR_Text_Parser accepts exactly the catalog's decimal texts
  it("Property 4: accepts exactly the catalog's decimal texts", () => {
    console.log(`race Property 4 seed 0x${FC_SEED.toString(16)}`)
    const catalogText = fc.constantFrom(...CATALOG.map(String))
    const textArb = fc.oneof(
      fc.string({ unit: 'grapheme' }),
      fc.stringMatching(/^[0-9]{0,6}$/),
      catalogText,
      fc.tuple(catalogText, fc.string({ minLength: 1, maxLength: 3 })).map(([t, s]) => t + s),
      fc.tuple(fc.string({ minLength: 1, maxLength: 3 }), catalogText).map(([s, t]) => s + t),
      catalogText.chain((t) => fc.integer({ min: 0, max: t.length - 1 }).map((i) => t.slice(0, i) + t.slice(i + 1))),
    )
    fc.assert(
      fc.property(textArb, (text) => {
        const id = parse(text)
        const expected = CATALOG.find((c) => String(c) === text) ?? null
        expect(id).toBe(expected)
      }),
      { seed: FC_SEED, numRuns: 1000 },
    )
  })
})
