// Requirement 7.1 and Property 1: the Logit_Window.
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { LogitWindow, logitsToBits, meanOf } from '../logitWindow'

const FC_SEED = 0x711e4ead

describe('LogitWindow', () => {
  it('averages 1, 2, then 3 frames and then slides', () => {
    const w = new LogitWindow(3)
    expect(w.mean()).toBeNull()
    w.push([3, -3])
    expect(Array.from(w.mean()!)).toEqual([3, -3])
    w.push([6, 0])
    expect(Array.from(w.mean()!)).toEqual([4.5, -1.5])
    w.push([0, 3])
    expect(Array.from(w.mean()!)).toEqual([3, 0])
    w.push([9, 9])
    expect(w.length).toBe(3)
    expect(Array.from(w.mean()!)).toEqual([5, 4])
    w.clear()
    expect(w.length).toBe(0)
    expect(w.mean()).toBeNull()
  })

  it('bit i is 1 only for a logit greater than 0', () => {
    expect(Array.from(logitsToBits([-1, 0, 1e-9, 5]))).toEqual([0, 0, 1, 1])
  })

  it('rejects a size below 1 and a length change', () => {
    expect(() => new LogitWindow(0)).toThrow(RangeError)
    const w = new LogitWindow(2)
    w.push([1, 2])
    expect(() => w.push([1])).toThrow(RangeError)
  })

  // Feature: tile-reader, Property 1: The window decodes the mean of the last frames
  it('Property 1: The window decodes the mean of the last frames', () => {
    console.log(`fast-check seed ${FC_SEED}`)
    const vec = fc.array(fc.float({ min: -50, max: 50, noNaN: true }), { minLength: 64, maxLength: 64 })
    const op = fc.oneof({ arbitrary: vec, weight: 5 }, { arbitrary: fc.constant(null), weight: 1 })
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 5 }), fc.array(op, { maxLength: 30 }), (n, ops) => {
        const w = new LogitWindow(n)
        let since: number[][] = []
        for (const o of ops) {
          if (o === null) {
            w.clear()
            since = []
          } else {
            w.push(o)
            since.push(o)
          }
          const last = since.slice(-n)
          if (last.length === 0) {
            expect(w.mean()).toBeNull()
            continue
          }
          const want = meanOf(last.map((v) => Float32Array.from(v)))
          const got = w.mean()!
          for (let i = 0; i < 64; i++) expect(Math.abs(got[i] - want[i])).toBeLessThanOrEqual(1e-5)
          expect(w.length).toBe(last.length)
        }
      }),
      { seed: FC_SEED, numRuns: 500 },
    )
  })
})
