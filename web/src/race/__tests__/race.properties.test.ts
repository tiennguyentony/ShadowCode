import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { createRace, MODES, productFound, reset, results, start, type Mode, type RaceState } from '../race'

const FC_SEED = 0x7ace2026
const NUM_RUNS = 1000
const CATALOG = [1001, 1002, 1003]

type Event =
  | { kind: 'start'; mode: Mode; dt: number }
  | { kind: 'found'; mode: Mode; id: number; dt: number }
  | { kind: 'reset'; dt: number }

const modeArb = fc.constantFrom<Mode>('qr', 'tilemark')
const dtArb = fc.integer({ min: 0, max: 5000 })
const idArb = fc.oneof(
  { weight: 6, arbitrary: fc.constantFrom(...CATALOG) },
  { weight: 1, arbitrary: fc.constantFrom(0, 1004, 999, 10010) },
  { weight: 1, arbitrary: fc.integer({ min: 0, max: 0xffffffff }) },
)
const eventArb: fc.Arbitrary<Event> = fc.oneof(
  { weight: 2, arbitrary: fc.record({ kind: fc.constant('start' as const), mode: modeArb, dt: dtArb }) },
  { weight: 8, arbitrary: fc.record({ kind: fc.constant('found' as const), mode: modeArb, id: idArb, dt: dtArb }) },
  { weight: 1, arbitrary: fc.record({ kind: fc.constant('reset' as const), dt: dtArb }) },
)
const eventsArb = fc.array(eventArb, { maxLength: 80 })

function apply(state: RaceState, e: Event, now: number): RaceState {
  if (e.kind === 'start') return start(state, e.mode, now)
  if (e.kind === 'found') return productFound(state, e.mode, e.id, now)
  return reset(state)
}

/** Reference: a plain mutable simulation written from the requirements, not from race.ts. */
function reference(events: readonly Event[]) {
  type RefRun = { startedAt: number; finds: { id: number; ms: number }[]; done: boolean }
  const runs: Record<Mode, RefRun | null> = { qr: null, tilemark: null }
  let now = 0
  for (const e of events) {
    now += e.dt
    if (e.kind === 'reset') {
      runs.qr = null
      runs.tilemark = null
    } else if (e.kind === 'start') {
      const other = runs[e.mode === 'qr' ? 'tilemark' : 'qr']
      if (!(other && !other.done)) runs[e.mode] = { startedAt: now, finds: [], done: false }
    } else {
      const run = runs[e.mode]
      if (run && !run.done && CATALOG.includes(e.id) && !run.finds.some((f) => f.id === e.id)) {
        run.finds.push({ id: e.id, ms: now - run.startedAt })
        if (run.finds.length === CATALOG.length) run.done = true
      }
    }
  }
  return runs
}

describe('race properties', () => {
  // Feature: race, Property 1: Finds equal the reference model
  it('Property 1: Finds equal the reference model', () => {
    console.log(`race Property 1 seed 0x${FC_SEED.toString(16)}`)
    fc.assert(
      fc.property(eventsArb, (events) => {
        let state = createRace(CATALOG)
        let now = 0
        for (const e of events) state = apply(state, e, (now += e.dt))
        const ref = reference(events)
        for (const mode of MODES) {
          const run = state.runs[mode]
          const refRun = ref[mode]
          expect(run === null).toBe(refRun === null)
          if (run && refRun) {
            expect(run.startedAt).toBe(refRun.startedAt)
            expect(run.finds).toEqual(refRun.finds)
          }
        }
      }),
      { seed: FC_SEED, numRuns: NUM_RUNS },
    )
  })

  // Feature: race, Property 2: Ignored events change nothing
  it('Property 2: Ignored events change nothing', () => {
    console.log(`race Property 2 seed 0x${FC_SEED.toString(16)}`)
    fc.assert(
      fc.property(eventsArb, fc.integer({ min: 1005, max: 0xffffffff }), (events, foreignId) => {
        let state = createRace(CATALOG)
        let now = 0
        for (const e of events) {
          now += e.dt
          for (const mode of MODES) {
            const run = state.runs[mode]
            expect(productFound(state, mode, foreignId, now)).toBe(state)
            expect(productFound(state, mode, 0, now)).toBe(state)
            if (run === null || run.totalMs !== null) {
              for (const id of CATALOG) expect(productFound(state, mode, id, now)).toBe(state)
            } else {
              for (const f of run.finds) expect(productFound(state, mode, f.id, now)).toBe(state)
              const other: Mode = mode === 'qr' ? 'tilemark' : 'qr'
              expect(start(state, other, now)).toBe(state)
            }
          }
          state = apply(state, e, now)
        }
      }),
      { seed: FC_SEED, numRuns: NUM_RUNS },
    )
  })

  // Feature: race, Property 3: A Run finishes exactly when all Catalog_IDs are found
  it('Property 3: A Run finishes exactly when all Catalog_IDs are found', () => {
    console.log(`race Property 3 seed 0x${FC_SEED.toString(16)}`)
    fc.assert(
      fc.property(eventsArb, (events) => {
        let state = createRace(CATALOG)
        let now = 0
        for (const e of events) {
          state = apply(state, e, (now += e.dt))
          for (const mode of MODES) {
            const run = state.runs[mode]
            if (!run) continue
            const all = CATALOG.every((id) => run.finds.some((f) => f.id === id))
            expect(run.totalMs !== null).toBe(all)
            if (all) expect(run.totalMs).toBe(Math.max(...run.finds.map((f) => f.ms)))
          }
          const r = results(state)
          expect(r.totals.qr).toBe(state.runs.qr?.totalMs ?? null)
          expect(r.totals.tilemark).toBe(state.runs.tilemark?.totalMs ?? null)
          if (r.totals.qr !== null && r.totals.tilemark !== null) {
            expect(r.differenceMs).toBe(r.totals.qr - r.totals.tilemark)
          } else {
            expect(r.differenceMs).toBeNull()
          }
        }
      }),
      { seed: FC_SEED, numRuns: NUM_RUNS },
    )
  })
})
