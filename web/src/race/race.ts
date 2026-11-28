/**
 * Race_Logic: the timers and results of the QR and Tilemark race.
 * Spec: .kiro/specs/race/requirements.md (Requirements 3 and 6) and design.md.
 *
 * Pure: state in, new state out. Every time comes from a `now` argument
 * (a Run_Clock value in milliseconds), so tests drive it with a fake clock.
 * An ignored event returns the same state object.
 */

export type Mode = 'qr' | 'tilemark'
export const MODES: readonly Mode[] = Object.freeze(['qr', 'tilemark'] as const)

export interface Find {
  readonly id: number
  /** Find_Time: ms from the Run's start. */
  readonly ms: number
}

export interface Run {
  /** Run_Clock time of Start, in ms. */
  readonly startedAt: number
  /** Finds in the order found. */
  readonly finds: readonly Find[]
  /** Run_Total in ms once every Catalog_ID is found; null while running. */
  readonly totalMs: number | null
}

export interface RaceState {
  /** Catalog_IDs in display order. */
  readonly catalog: readonly number[]
  readonly runs: Readonly<Record<Mode, Run | null>>
}

export interface ResultRow {
  readonly id: number
  readonly ms: Readonly<Record<Mode, number | null>>
}

export interface Results {
  readonly rows: readonly ResultRow[]
  readonly totals: Readonly<Record<Mode, number | null>>
  /** QR total minus Tilemark total, when both Runs are finished. Positive: Tilemark faster. */
  readonly differenceMs: number | null
  readonly faster: Mode | 'tie' | null
}

const NO_RUNS: Readonly<Record<Mode, Run | null>> = Object.freeze({ qr: null, tilemark: null })

export function createRace(catalog: readonly number[]): RaceState {
  const seen = new Set<number>()
  for (const id of catalog) {
    if (!Number.isInteger(id) || id <= 0 || id > 0xffffffff) {
      throw new RangeError(`Race catalog: Product_ID must be an integer in [1, 2^32 - 1], got ${String(id)}`)
    }
    if (seen.has(id)) throw new RangeError(`Race catalog: duplicate Product_ID ${id}`)
    seen.add(id)
  }
  if (seen.size === 0) throw new RangeError('Race catalog: no Product_IDs')
  return Object.freeze({ catalog: Object.freeze([...catalog]), runs: NO_RUNS })
}

export function isRunning(run: Run | null): boolean {
  return run !== null && run.totalMs === null
}

export function runningMode(state: RaceState): Mode | null {
  for (const mode of MODES) if (isRunning(state.runs[mode])) return mode
  return null
}

function otherMode(mode: Mode): Mode {
  return mode === 'qr' ? 'tilemark' : 'qr'
}

function withRun(state: RaceState, mode: Mode, run: Run | null): RaceState {
  return Object.freeze({ catalog: state.catalog, runs: Object.freeze({ ...state.runs, [mode]: run }) })
}

/** Requirement 3.1–3.3: begin a new Run, replacing this Mode's Run; ignored while the other Mode runs. */
export function start(state: RaceState, mode: Mode, now: number): RaceState {
  if (isRunning(state.runs[otherMode(mode)])) return state
  return withRun(state, mode, Object.freeze({ startedAt: now, finds: Object.freeze([]), totalMs: null }))
}

/** Requirement 3.4–3.9: the first report of each Catalog_ID in a running Run counts. */
export function productFound(state: RaceState, mode: Mode, id: number, now: number): RaceState {
  const run = state.runs[mode]
  if (run === null || !isRunning(run)) return state
  if (!state.catalog.includes(id)) return state
  if (run.finds.some((f) => f.id === id)) return state
  const ms = now - run.startedAt
  const finds = Object.freeze([...run.finds, Object.freeze({ id, ms })])
  const done = state.catalog.every((c) => finds.some((f) => f.id === c))
  return withRun(state, mode, Object.freeze({ startedAt: run.startedAt, finds, totalMs: done ? ms : null }))
}

/** Requirement 6.4: clear both Modes. */
export function reset(state: RaceState): RaceState {
  if (state.runs.qr === null && state.runs.tilemark === null) return state
  return Object.freeze({ catalog: state.catalog, runs: NO_RUNS })
}

/** Time shown on the timer: running, now − start; finished, Run_Total; no Run, null. */
export function elapsedMs(run: Run | null, now: number): number | null {
  if (run === null) return null
  return run.totalMs ?? now - run.startedAt
}

export function results(state: RaceState): Results {
  const findMs = (mode: Mode, id: number): number | null =>
    state.runs[mode]?.finds.find((f) => f.id === id)?.ms ?? null
  const rows = state.catalog.map((id) => ({ id, ms: { qr: findMs('qr', id), tilemark: findMs('tilemark', id) } }))
  const totals = { qr: state.runs.qr?.totalMs ?? null, tilemark: state.runs.tilemark?.totalMs ?? null }
  const differenceMs = totals.qr !== null && totals.tilemark !== null ? totals.qr - totals.tilemark : null
  const faster: Mode | 'tie' | null =
    differenceMs === null ? null : differenceMs > 0 ? 'tilemark' : differenceMs < 0 ? 'qr' : 'tie'
  return { rows, totals, differenceMs, faster }
}

/** Requirement 6.5: seconds with two decimals, or a dash. */
export function formatSeconds(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return '–'
  return `${(ms / 1000).toFixed(2)} s`
}
