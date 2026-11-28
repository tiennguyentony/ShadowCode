/**
 * Frame_Rate (race spec, Requirement 7.3): Processed_Frames in the last
 * window (default 1000 ms) of the Run_Clock. Times come in as arguments.
 */
export interface FrameRateCounter {
  /** One Processed_Frame finished at `now` (ms). */
  record(now: number): void
  /** Processed_Frames with a time in (now − window, now]. */
  perSecond(now: number): number
  reset(): void
}

export function createFrameRateCounter(windowMs = 1000): FrameRateCounter {
  if (!(windowMs > 0) || !Number.isFinite(windowMs)) {
    throw new RangeError(`Frame_Rate windowMs must be a positive number of ms, got ${String(windowMs)}`)
  }
  let times: number[] = []
  const prune = (now: number) => {
    let i = 0
    while (i < times.length && times[i] <= now - windowMs) i++
    if (i > 0) times = times.slice(i)
  }
  return {
    record(now) {
      times.push(now)
      prune(now)
    },
    perSecond(now) {
      prune(now)
      return (times.filter((t) => t <= now).length * 1000) / windowMs
    },
    reset() {
      times = []
    },
  }
}
