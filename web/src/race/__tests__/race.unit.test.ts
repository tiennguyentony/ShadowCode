import { beforeEach, describe, expect, it } from 'vitest'
import {
  createRace,
  elapsedMs,
  formatSeconds,
  isRunning,
  productFound,
  reset,
  results,
  runningMode,
  start,
  type RaceState,
} from '../race'
import { createFakeClock, type FakeClock } from './fakeClock'

const CATALOG = [1001, 1002, 1003]

let clock: FakeClock
let race: RaceState

beforeEach(() => {
  clock = createFakeClock()
  race = createRace(CATALOG)
})

describe('start (3.1, 3.2, 3.3)', () => {
  it('3.1 begins a Run at the current clock time with no Finds', () => {
    race = start(race, 'qr', clock.now())
    expect(race.runs.qr).toEqual({ startedAt: 10_000, finds: [], totalMs: null })
    expect(isRunning(race.runs.qr)).toBe(true)
    expect(runningMode(race)).toBe('qr')
    expect(race.runs.tilemark).toBeNull()
  })

  it('3.2 Start again replaces the Run of that Mode and keeps the other', () => {
    race = start(race, 'tilemark', clock.now())
    for (const id of CATALOG) race = productFound(race, 'tilemark', id, clock.advance(1000))
    const tilemarkRun = race.runs.tilemark
    race = start(race, 'qr', clock.advance(5000))
    race = productFound(race, 'qr', 1001, clock.advance(700))
    race = start(race, 'qr', clock.advance(300))
    expect(race.runs.qr).toEqual({ startedAt: clock.now(), finds: [], totalMs: null })
    expect(race.runs.tilemark).toBe(tilemarkRun)
  })

  it('3.3 Start for the other Mode is ignored while one Mode is running', () => {
    race = start(race, 'qr', clock.now())
    const before = race
    expect(start(race, 'tilemark', clock.advance(100))).toBe(before)
  })
})

describe('product found (3.4–3.8)', () => {
  it('3.4 records the Find_Time from start', () => {
    race = start(race, 'qr', clock.now())
    race = productFound(race, 'qr', 1002, clock.advance(1234))
    expect(race.runs.qr?.finds).toEqual([{ id: 1002, ms: 1234 }])
    expect(race.runs.qr?.totalMs).toBeNull()
  })

  it('3.5 a duplicate keeps the first Find', () => {
    race = start(race, 'tilemark', clock.now())
    race = productFound(race, 'tilemark', 1001, clock.advance(500))
    const before = race
    race = productFound(race, 'tilemark', 1001, clock.advance(500))
    expect(race).toBe(before)
    expect(race.runs.tilemark?.finds).toEqual([{ id: 1001, ms: 500 }])
  })

  it('3.6 accepts the products in any order', () => {
    race = start(race, 'qr', clock.now())
    race = productFound(race, 'qr', 1003, clock.advance(100))
    race = productFound(race, 'qr', 1001, clock.advance(100))
    race = productFound(race, 'qr', 1002, clock.advance(100))
    expect(race.runs.qr?.finds.map((f) => f.id)).toEqual([1003, 1001, 1002])
    expect(race.runs.qr?.totalMs).toBe(300)
  })

  it('3.7 ignores a Product_ID that is not in the catalog', () => {
    race = start(race, 'qr', clock.now())
    const before = race
    for (const id of [1004, 0, -1, 1001.5, NaN]) {
      expect(productFound(race, 'qr', id, clock.advance(10))).toBe(before)
    }
  })

  it('3.8 ignores a Find for a Mode without a running Run', () => {
    const before = race
    expect(productFound(race, 'qr', 1001, clock.advance(10))).toBe(before)
    race = start(race, 'qr', clock.now())
    expect(productFound(race, 'tilemark', 1001, clock.advance(10)).runs.tilemark).toBeNull()
  })
})

describe('finish (3.9)', () => {
  it('finishes after three products with Run_Total = last Find_Time', () => {
    race = start(race, 'qr', clock.now())
    race = productFound(race, 'qr', 1001, clock.advance(2000))
    race = productFound(race, 'qr', 1002, clock.advance(3000))
    expect(isRunning(race.runs.qr)).toBe(true)
    race = productFound(race, 'qr', 1003, clock.advance(4000))
    expect(race.runs.qr?.totalMs).toBe(9000)
    expect(isRunning(race.runs.qr)).toBe(false)
    expect(runningMode(race)).toBeNull()
  })

  it('ignores Finds after the Run has finished', () => {
    race = start(race, 'qr', clock.now())
    for (const id of CATALOG) race = productFound(race, 'qr', id, clock.advance(100))
    const before = race
    expect(productFound(race, 'qr', 1001, clock.advance(100))).toBe(before)
  })

  it('elapsedMs runs while running and stops at Run_Total', () => {
    expect(elapsedMs(race.runs.qr, clock.now())).toBeNull()
    race = start(race, 'qr', clock.now())
    expect(elapsedMs(race.runs.qr, clock.advance(1500))).toBe(1500)
    for (const id of CATALOG) race = productFound(race, 'qr', id, clock.advance(1000))
    expect(elapsedMs(race.runs.qr, clock.advance(60_000))).toBe(4500)
  })

  it('the other Mode can start once the first has finished', () => {
    race = start(race, 'qr', clock.now())
    for (const id of CATALOG) race = productFound(race, 'qr', id, clock.advance(100))
    race = start(race, 'tilemark', clock.advance(100))
    expect(runningMode(race)).toBe('tilemark')
  })
})

describe('reset (6.4)', () => {
  it('clears both Modes, also while one is running', () => {
    race = start(race, 'qr', clock.now())
    for (const id of CATALOG) race = productFound(race, 'qr', id, clock.advance(100))
    race = start(race, 'tilemark', clock.advance(100))
    race = productFound(race, 'tilemark', 1002, clock.advance(100))
    race = reset(race)
    expect(race.runs).toEqual({ qr: null, tilemark: null })
    expect(race.catalog).toEqual(CATALOG)
    expect(results(race).totals).toEqual({ qr: null, tilemark: null })
  })

  it('a Find after reset is ignored', () => {
    race = start(race, 'qr', clock.now())
    race = reset(race)
    expect(productFound(race, 'qr', 1001, clock.advance(10))).toBe(race)
  })
})

describe('results (6.1–6.3)', () => {
  it('rows per product per Mode, totals and difference', () => {
    race = start(race, 'qr', clock.now())
    race = productFound(race, 'qr', 1002, clock.advance(4000))
    race = productFound(race, 'qr', 1001, clock.advance(5000))
    race = productFound(race, 'qr', 1003, clock.advance(6000))
    race = start(race, 'tilemark', clock.advance(10_000))
    race = productFound(race, 'tilemark', 1001, clock.advance(1000))
    let r = results(race)
    expect(r.rows).toEqual([
      { id: 1001, ms: { qr: 9000, tilemark: 1000 } },
      { id: 1002, ms: { qr: 4000, tilemark: null } },
      { id: 1003, ms: { qr: 15_000, tilemark: null } },
    ])
    expect(r.totals).toEqual({ qr: 15_000, tilemark: null })
    expect(r.differenceMs).toBeNull()
    expect(r.faster).toBeNull()

    race = productFound(race, 'tilemark', 1003, clock.advance(1500))
    race = productFound(race, 'tilemark', 1002, clock.advance(1500))
    r = results(race)
    expect(r.totals).toEqual({ qr: 15_000, tilemark: 4000 })
    expect(r.differenceMs).toBe(11_000)
    expect(r.faster).toBe('tilemark')
  })

  it('names QR as faster when its total is lower, and a tie when equal', () => {
    for (const [qrStep, tmStep, faster] of [
      [100, 200, 'qr'],
      [100, 100, 'tie'],
    ] as const) {
      race = reset(race)
      race = start(race, 'qr', clock.now())
      for (const id of CATALOG) race = productFound(race, 'qr', id, clock.advance(qrStep))
      race = start(race, 'tilemark', clock.now())
      for (const id of CATALOG) race = productFound(race, 'tilemark', id, clock.advance(tmStep))
      expect(results(race).faster).toBe(faster)
    }
  })
})

describe('formatSeconds (6.5)', () => {
  it('two decimals, dash for none', () => {
    expect(formatSeconds(0)).toBe('0.00 s')
    expect(formatSeconds(1234)).toBe('1.23 s')
    expect(formatSeconds(12_345.6)).toBe('12.35 s')
    expect(formatSeconds(null)).toBe('–')
    expect(formatSeconds(NaN)).toBe('–')
  })
})

describe('createRace', () => {
  it('rejects 0, duplicates, non-integers and an empty catalog', () => {
    expect(() => createRace([0, 1001])).toThrow(RangeError)
    expect(() => createRace([1001, 1001])).toThrow(/duplicate/)
    expect(() => createRace([1001.5])).toThrow(RangeError)
    expect(() => createRace([2 ** 32])).toThrow(RangeError)
    expect(() => createRace([])).toThrow(RangeError)
  })
})
