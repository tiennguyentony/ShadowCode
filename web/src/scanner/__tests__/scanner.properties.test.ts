import * as fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import type { DecodeResult } from '../../codec/message'
import { arbCase, arbCatalog, arbResult, FC_SEED, isCatalogId, RUNS, run, scannerFor } from './scanner.helpers'

const logSeed = (p: number, label: string) =>
  console.log(`[scanner] property ${p} (${label}) fast-check seed=0x${FC_SEED.toString(16)}`)

const params = { seed: FC_SEED, numRuns: RUNS }

describe.each([
  ['N in {2, 3}', fc.constantFrom<2 | 3>(2, 3)],
  ['N = 2', fc.constant<2 | 3>(2)],
  ['N = 3', fc.constant<2 | 3>(3)],
] as const)('scanner properties, %s', (label, arbN) => {
  // Feature: scanner, Property 1: Only catalog IDs are shown, never the Reserved_ID
  it('Property 1: only catalog IDs are shown, never the Reserved_ID', () => {
    logSeed(1, label)
    fc.assert(
      fc.property(arbCase(arbN), ({ n, catalog, results }) => {
        const out = run(scannerFor(n, new Set(catalog)), results)
        for (const p of out) {
          if (p === null) continue
          expect(p).not.toBe(0)
          expect(catalog).toContain(p)
        }
      }),
      params,
    )
  })

  // Feature: scanner, Property 2: Shown only after N consecutive equal Catalog_IDs
  it('Property 2: shown only after N consecutive equal Catalog_IDs (2.10)', () => {
    logSeed(2, label)
    fc.assert(
      fc.property(arbCase(arbN), ({ n, catalog, results }) => {
        const out = run(scannerFor(n, new Set(catalog)), results)
        out.forEach((p, i) => {
          if (p === null) return
          expect(i).toBeGreaterThanOrEqual(n - 1)
          expect(isCatalogId(catalog, p)).toBe(true)
          for (let j = i - n + 1; j <= i; j++) expect(results[j]?.id).toBe(p)
        })
      }),
      params,
    )
  })

  // Feature: scanner, Property 3: N consecutive equal Catalog_IDs are shown at the Nth
  it('Property 3: N consecutive equal Catalog_IDs from no Candidate_ID are shown at the Nth (2.11)', () => {
    logSeed(3, label)
    const arb = fc.tuple(arbN, arbCatalog).chain(([n, catalog]) => {
      const valid = catalog.filter((id) => id !== 0)
      const breaker: fc.Arbitrary<DecodeResult> = fc.oneof(
        fc.constant(null),
        fc.record({ id: fc.constant(0), correctedBits: fc.nat(4) }),
        fc.record({
          id: fc.integer({ min: 0, max: 0xffffffff }).filter((id) => !catalog.includes(id)),
          correctedBits: fc.nat(4),
        }),
      )
      const prefix = fc
        .tuple(fc.array(arbResult(catalog), { maxLength: 30 }), breaker, fc.boolean())
        .map(([p, b, empty]) => (empty ? [] : [...p, b]))
      return fc.record({
        n: fc.constant(n),
        catalog: fc.constant(catalog),
        prefix,
        c: fc.constantFrom(...valid),
        bits: fc.array(fc.nat(4), { minLength: n, maxLength: n }),
      })
    })
    fc.assert(
      fc.property(arb, ({ n, catalog, prefix, c, bits }) => {
        const s = scannerFor(n, new Set(catalog))
        run(s, prefix)
        expect(s.state()).toEqual({ candidateId: null, matchRun: 0 })
        const out = run(s, bits.map((b) => ({ id: c, correctedBits: b })))
        expect(out).toEqual([...Array(n - 1).fill(null), c])
      }),
      params,
    )
  })

  // Feature: scanner, Property 4: correctedBits is ignored
  it('Property 4: correctedBits is ignored (1.5, 2.9)', () => {
    logSeed(4, label)
    const arb = arbCase(arbN).chain((c) =>
      fc.record({ c: fc.constant(c), other: fc.array(fc.integer(), { minLength: c.results.length, maxLength: c.results.length }) }),
    )
    fc.assert(
      fc.property(arb, ({ c: { n, catalog, results }, other }) => {
        const changed = results.map((r, i) => (r === null ? null : { id: r.id, correctedBits: other[i] }))
        const s = scannerFor(n, new Set(catalog))
        const t = scannerFor(n, new Set(catalog))
        results.forEach((r, i) => {
          expect(t.push(changed[i])).toBe(s.push(r))
          expect(t.state()).toEqual(s.state())
        })
      }),
      params,
    )
  })

  // Feature: scanner, Property 5: Agreement with the reference model
  it('Property 5: state agrees with the reference model (2.1, 2.2, 2.5, 2.6, 2.7)', () => {
    logSeed(5, label)
    fc.assert(
      fc.property(arbCase(arbN), ({ n, catalog, results }) => {
        const s = scannerFor(n, new Set(catalog))
        results.forEach((r, i) => {
          s.push(r)
          // Reference: computed from results[0..i] alone.
          let candidateId: number | null = null
          let matchRun = 0
          if (r !== null && isCatalogId(catalog, r.id)) {
            candidateId = r.id
            for (let j = i; j >= 0 && results[j]?.id === r.id; j--) matchRun++
          }
          expect(s.state()).toEqual({ candidateId, matchRun })
        })
      }),
      params,
    )
  })
})
