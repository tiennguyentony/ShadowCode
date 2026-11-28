import * as fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { createScanner, type ScannerConfiguration } from '../scanner'
import { CATALOG, FC_SEED, hit } from './scanner.helpers'

const withCount = (value: unknown): ScannerConfiguration =>
  ({ productCatalog: CATALOG, confirmationCount: value }) as unknown as ScannerConfiguration

describe('Confirmation_Count configuration', () => {
  it.each([
    ['0', 0],
    ['1', 1],
    ['4', 4],
    ['-1', -1],
    ['2.5', 2.5],
    ['NaN', NaN],
    ['Infinity', Infinity],
    ['-Infinity', -Infinity],
    ['a string "2"', '2'],
    ['null', null],
  ])('3.3 rejects %s at configuration time, naming Confirmation_Count and [2, 3]', (_label, value) => {
    let scanner: unknown
    expect(() => {
      scanner = createScanner(withCount(value))
    }).toThrow(RangeError)
    expect(() => createScanner(withCount(value))).toThrow(/Confirmation_Count must be an integer in \[2, 3\]/)
    // No Scanner exists, so no Decode_Attempt can be processed with this configuration.
    expect(scanner).toBeUndefined()
  })

  it('3.4 rejects a missing Confirmation_Count (property absent), naming it as missing', () => {
    const config = { productCatalog: CATALOG } as unknown as ScannerConfiguration
    expect(() => createScanner(config)).toThrow(RangeError)
    expect(() => createScanner(config)).toThrow(/Confirmation_Count is missing/)
  })

  it('3.4 rejects an explicit undefined Confirmation_Count, naming it as missing', () => {
    expect(() => createScanner(withCount(undefined))).toThrow(/Confirmation_Count is missing/)
  })

  it('3.4 has no implicit default: a missing value is never treated as 2 or 3', () => {
    expect(() => createScanner({} as unknown as ScannerConfiguration)).toThrow(/Confirmation_Count is missing/)
  })

  it('rejects a missing Scanner_Configuration', () => {
    expect(() => createScanner(undefined as unknown as ScannerConfiguration)).toThrow(TypeError)
    expect(() => createScanner(null as unknown as ScannerConfiguration)).toThrow(TypeError)
  })

  it.each([2, 3] as const)('3.1, 3.2 accepts %i and uses it as N', (n) => {
    const s = createScanner({ productCatalog: CATALOG, confirmationCount: n })
    const out = Array.from({ length: n }, () => s.push(hit(1001)))
    expect(out[n - 1]).toBe(1001)
    expect(out.slice(0, n - 1).every((x) => x === null)).toBe(true)
  })

  // Feature: scanner, Property 6: Only 2 and 3 are accepted as Confirmation_Count
  it('Property 6: any value other than 2 and 3 is rejected with a RangeError naming Confirmation_Count', () => {
    console.log(`[scanner] property 6 fast-check seed=0x${FC_SEED.toString(16)}`)
    const arb = fc
      .oneof(fc.anything(), fc.double(), fc.integer(), fc.integer({ min: -10, max: 10 }))
      .filter((v) => v !== 2 && v !== 3)
    fc.assert(
      fc.property(arb, (v) => {
        expect(() => createScanner(withCount(v))).toThrow(RangeError)
        expect(() => createScanner(withCount(v))).toThrow(/Confirmation_Count/)
      }),
      { seed: FC_SEED, numRuns: 2000 },
    )
  })
})
