/** A fake Run_Clock for the race tests: time moves only when advance() is called. */
export interface FakeClock {
  now(): number
  advance(ms: number): number
}

export function createFakeClock(startMs = 10_000): FakeClock {
  let t = startMs
  return {
    now: () => t,
    advance: (ms) => (t += ms),
  }
}
