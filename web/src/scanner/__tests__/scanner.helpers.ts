import * as fc from 'fast-check'
import type { DecodeResult } from '../../codec/message'
import { createScanner, type Scanner } from '../scanner'

export const FC_SEED = 0x5ca77e12
export const RUNS = 2000

export const CATALOG = new Set([1001, 1002, 1003])

export function scannerFor(n: 2 | 3, catalog: ReadonlySet<number> = CATALOG): Scanner {
  return createScanner({ productCatalog: catalog, confirmationCount: n })
}

export const hit = (id: number, correctedBits = 0): DecodeResult => ({ id, correctedBits })

/** Push every result and return what push returned for each. */
export function run(scanner: Scanner, results: readonly DecodeResult[]): (number | null)[] {
  return results.map((r) => scanner.push(r))
}

export const arbN = fc.constantFrom<2 | 3>(2, 3)

const arbU32 = fc.integer({ min: 0, max: 0xffffffff })

/** 1 to 5 unique IDs; 0 is added in about one run in four. */
export const arbCatalog: fc.Arbitrary<number[]> = fc
  .tuple(fc.uniqueArray(fc.integer({ min: 1, max: 0xffffffff }), { minLength: 1, maxLength: 5 }), fc.integer({ min: 0, max: 3 }))
  .map(([ids, r]) => (r === 0 ? [...ids, 0] : ids))

export function arbResult(catalog: readonly number[]): fc.Arbitrary<DecodeResult> {
  const bits = fc.integer({ min: 0, max: 4 })
  // Draw catalog IDs from at most 2 entries so that runs of equal IDs are common.
  const common = catalog.slice(0, 2)
  return fc.oneof(
    { weight: 2, arbitrary: fc.constant(null) },
    { weight: 6, arbitrary: fc.record({ id: fc.constantFrom(...common), correctedBits: bits }) },
    { weight: 1, arbitrary: fc.record({ id: fc.constantFrom(...catalog), correctedBits: bits }) },
    { weight: 1, arbitrary: fc.record({ id: fc.constant(0), correctedBits: bits }) },
    { weight: 1, arbitrary: fc.record({ id: arbU32, correctedBits: bits }) },
  )
}

export interface Case {
  n: 2 | 3
  catalog: number[]
  results: DecodeResult[]
}

export function arbCase(arbNValue: fc.Arbitrary<2 | 3>): fc.Arbitrary<Case> {
  return fc.tuple(arbNValue, arbCatalog).chain(([n, catalog]) =>
    fc.array(arbResult(catalog), { maxLength: 60 }).map((results) => ({ n, catalog, results })),
  )
}

export const isCatalogId = (catalog: readonly number[], id: number): boolean => id !== 0 && catalog.includes(id)
