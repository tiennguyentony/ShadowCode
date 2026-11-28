import { describe, expect, it } from 'vitest'
import { createFrameRateCounter } from '../frameRate'
import { createFakeClock } from './fakeClock'

describe('Frame_Rate (7.3)', () => {
  it('counts Processed_Frames in the last 1000 ms', () => {
    const clock = createFakeClock()
    const fps = createFrameRateCounter()
    expect(fps.perSecond(clock.now())).toBe(0)
    for (let i = 0; i < 12; i++) fps.record(clock.advance(100))
    expect(fps.perSecond(clock.now())).toBe(10)
  })

  it('a frame exactly 1000 ms old drops out', () => {
    const fps = createFrameRateCounter()
    fps.record(0)
    fps.record(500)
    expect(fps.perSecond(999)).toBe(2)
    expect(fps.perSecond(1000)).toBe(1)
    expect(fps.perSecond(1500)).toBe(0)
  })

  it('reset clears the count', () => {
    const fps = createFrameRateCounter()
    fps.record(10)
    fps.reset()
    expect(fps.perSecond(10)).toBe(0)
  })

  it('rejects a window that is not positive', () => {
    expect(() => createFrameRateCounter(0)).toThrow(RangeError)
    expect(() => createFrameRateCounter(NaN)).toThrow(RangeError)
  })
})
