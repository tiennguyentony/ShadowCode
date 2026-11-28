import { afterEach, describe, expect, it, vi } from 'vitest'
import { CATALOG, hit, run, scannerFor } from './scanner.helpers'

const A = 1001
const B = 1002
const NOT_IN_CATALOG = 4242

describe.each([2, 3] as const)('scanner unit tests, N = %i', (n) => {
  const runOf = (id: number) => Array.from({ length: n }, () => hit(id))

  it('1.1 a Catalog_ID becomes the Candidate_ID', () => {
    const s = scannerFor(n)
    expect(s.push(hit(A))).toBeNull()
    expect(s.state()).toEqual({ candidateId: A, matchRun: 1 })
  })

  it('1.2 an ID not in the catalog is discarded and never shown', () => {
    const s = scannerFor(n)
    const out = run(s, Array.from({ length: 5 }, () => hit(NOT_IN_CATALOG)))
    expect(out).toEqual([null, null, null, null, null])
    expect(s.state()).toEqual({ candidateId: null, matchRun: 0 })
  })

  it('1.3 the Reserved_ID 0 is never shown, even if the catalog contains 0', () => {
    const s = scannerFor(n, new Set([0, A]))
    const out = run(s, Array.from({ length: 5 }, () => hit(0)))
    expect(out).toEqual([null, null, null, null, null])
    expect(s.state()).toEqual({ candidateId: null, matchRun: 0 })
  })

  it('1.4 and 4.1(a) only catalog IDs other than 0 are shown', () => {
    const s = scannerFor(n, new Set([0, A]))
    const out = run(s, [...runOf(0), ...runOf(NOT_IN_CATALOG), ...runOf(A)])
    expect(out.filter((x) => x !== null)).toEqual([A])
  })

  it('1.5 the catalog check ignores correctedBits', () => {
    const s = scannerFor(n)
    expect(s.push(hit(A, 0))).toBeNull()
    expect(s.state().candidateId).toBe(A)
    expect(s.push(hit(NOT_IN_CATALOG, 0))).toBeNull()
    expect(s.push(hit(A, 4))).toBeNull()
    expect(s.state().candidateId).toBe(A)
  })

  it('2.1 a first Catalog_ID is held with Match_Run 1 and not shown', () => {
    const s = scannerFor(n)
    expect(s.state()).toEqual({ candidateId: null, matchRun: 0 })
    expect(s.push(hit(B))).toBeNull()
    expect(s.state()).toEqual({ candidateId: B, matchRun: 1 })
  })

  it('2.2 an equal Catalog_ID increases the Match_Run by 1', () => {
    const s = scannerFor(n)
    s.push(hit(A))
    s.push(hit(A))
    expect(s.state()).toEqual({ candidateId: A, matchRun: 2 })
    s.push(hit(A))
    expect(s.state()).toEqual({ candidateId: A, matchRun: 3 })
  })

  it('2.3 the Catalog_ID is shown when the Match_Run reaches N, and again while it stays >= N', () => {
    const s = scannerFor(n)
    const out = run(s, [...runOf(A), hit(A), hit(A)])
    expect(out.slice(n - 1)).toEqual([A, A, A])
  })

  it('2.4 below N the Candidate_ID is held but not shown', () => {
    const s = scannerFor(n)
    const out = run(s, runOf(A).slice(0, n - 1))
    expect(out.every((x) => x === null)).toBe(true)
    expect(s.state()).toEqual({ candidateId: A, matchRun: n - 1 })
  })

  it('2.5 a different Catalog_ID replaces the Candidate_ID with Match_Run 1', () => {
    const s = scannerFor(n)
    s.push(hit(A))
    expect(s.push(hit(B))).toBeNull()
    expect(s.state()).toEqual({ candidateId: B, matchRun: 1 })
    // Also after A was already shown.
    const t = scannerFor(n)
    run(t, runOf(A))
    expect(t.push(hit(B))).toBeNull()
    expect(t.state()).toEqual({ candidateId: B, matchRun: 1 })
  })

  it('2.6 null clears the Candidate_ID and sets the Match_Run to 0', () => {
    const s = scannerFor(n)
    s.push(hit(A))
    expect(s.push(null)).toBeNull()
    expect(s.state()).toEqual({ candidateId: null, matchRun: 0 })
    // The run restarts: [A, null, A x (N-1)] does not show A.
    const t = scannerFor(n)
    const out = run(t, [hit(A), null, ...runOf(A).slice(0, n - 1)])
    expect(out.every((x) => x === null)).toBe(true)
  })

  it('2.7 a non-Catalog_ID clears the Candidate_ID and sets the Match_Run to 0', () => {
    for (const bad of [NOT_IN_CATALOG, 0]) {
      const s = scannerFor(n)
      s.push(hit(A))
      expect(s.push(hit(bad))).toBeNull()
      expect(s.state()).toEqual({ candidateId: null, matchRun: 0 })
    }
  })

  describe('2.8 consecutiveness is counted over Decode_Attempts', () => {
    afterEach(() => {
      vi.useRealTimers()
    })
    it('long gaps between Decode_Attempts (unprocessed frames) do not break a run', () => {
      vi.useFakeTimers()
      const s = scannerFor(n)
      const out: (number | null)[] = []
      for (let i = 0; i < n; i++) {
        out.push(s.push(hit(A)))
        vi.advanceTimersByTime(10_000)
      }
      expect(out[n - 1]).toBe(A)
    })
  })

  it('2.9 Decode_Results with the same ID and different correctedBits confirm each other', () => {
    const s = scannerFor(n)
    const out = run(s, Array.from({ length: n }, (_, i) => hit(A, i + 1)))
    expect(out[n - 1]).toBe(A)
  })

  it('2.10 nothing is shown without N consecutive equal Catalog_IDs', () => {
    const s = scannerFor(n)
    // Alternating IDs and single breaks never form a run of N.
    const seq = [hit(A), hit(B), hit(A), null, hit(A), hit(NOT_IN_CATALOG), hit(B), hit(0), hit(B)]
    if (n === 3) seq.push(hit(A), hit(A), null, hit(A))
    expect(run(s, seq).every((x) => x === null)).toBe(true)
  })

  it('2.11 N consecutive equal Catalog_IDs from no Candidate_ID are shown at the Nth', () => {
    const s = scannerFor(n)
    const out = run(s, runOf(B))
    expect(out).toEqual([...Array(n - 1).fill(null), B])
  })

  it('4.1(b) with the catalog of the app, the Nth equal read is the first shown', () => {
    const s = scannerFor(n, CATALOG)
    expect(run(s, [hit(1003), ...runOf(1003).slice(1)])[n - 1]).toBe(1003)
  })
})

describe('4.1 display rule at the chosen N = 2', () => {
  it('an ID is shown on the 2nd consecutive Decode_Attempt, not the 1st', () => {
    const s = scannerFor(2)
    expect(s.push(hit(A))).toBeNull()
    expect(s.push(hit(A))).toBe(A)
  })
  it('2 consecutive reads of a non-catalog ID are not shown', () => {
    const s = scannerFor(2)
    expect(run(s, [hit(NOT_IN_CATALOG), hit(NOT_IN_CATALOG)])).toEqual([null, null])
  })
  it('two scanners do not share state', () => {
    const s = scannerFor(2)
    const t = scannerFor(2)
    s.push(hit(A))
    expect(t.push(hit(A))).toBeNull()
  })
})
