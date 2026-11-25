import { describe, expect, it } from 'vitest'
import { frameFileName } from '../frameName'

describe('frameFileName', () => {
  it('names a saved frame by its size and the UTC time', () => {
    expect(frameFileName(1080, 1920, new Date(Date.UTC(2026, 10, 25, 14, 30, 0)))).toBe(
      'frame-1080x1920-20261125-143000.png',
    )
  })

  it('pads single-digit months, days and times', () => {
    expect(frameFileName(1920, 1080, new Date(Date.UTC(2027, 0, 2, 3, 4, 5)))).toBe('frame-1920x1080-20270102-030405.png')
  })
})
