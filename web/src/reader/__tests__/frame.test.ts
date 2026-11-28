// Requirement 2: frame checks and the shrink step.
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { areaResize, checkFrame, shrinkToShortSide } from '../frame'
import type { RgbaFrame } from '../types'

const FC_SEED = 0x711e4ead

function solid(width: number, height: number, rgb: readonly [number, number, number]): RgbaFrame {
  const data = new Uint8ClampedArray(width * height * 4)
  for (let p = 0; p < width * height; p++) data.set([rgb[0], rgb[1], rgb[2], 255], p * 4)
  return { width, height, data }
}

describe('checkFrame (2.1)', () => {
  it('rejects a wrong data length, naming the expected length', () => {
    expect(() => checkFrame({ width: 4, height: 4, data: new Uint8ClampedArray(63) })).toThrow(/expected 64/)
  })
  it.each([
    [0, 4],
    [4, -1],
    [2.5, 4],
    [Number.NaN, 4],
  ])('rejects size %s x %s', (w, h) => {
    expect(() => checkFrame({ width: w, height: h, data: new Uint8ClampedArray(0) })).toThrow(RangeError)
  })
})

describe('shrinkToShortSide (2.2, 2.3)', () => {
  it('2.3: leaves a 1080 x 1920 frame unchanged', () => {
    const f = solid(1080, 1920, [1, 2, 3])
    expect(shrinkToShortSide(f, 1080)).toBe(f)
  })

  it('2.2: shrinks 1440 x 2560 to 1080 x 1920 and 2160 x 3840 to 1080 x 1920', () => {
    expect(shrinkToShortSide(solid(1440, 2560, [0, 0, 0]), 1080)).toMatchObject({ width: 1080, height: 1920 })
    expect(shrinkToShortSide(solid(2160, 3840, [0, 0, 0]), 1080)).toMatchObject({ width: 1080, height: 1920 })
  })

  it('2.2: a landscape frame keeps its short side as height', () => {
    expect(shrinkToShortSide(solid(2560, 1440, [0, 0, 0]), 1080)).toMatchObject({ width: 1920, height: 1080 })
  })

  it('2.2: area averaging, a 2 x 2 checker of 0 and 200 halves to 100', () => {
    const data = new Uint8ClampedArray([0, 0, 0, 255, 200, 200, 200, 255, 200, 200, 200, 255, 0, 0, 0, 255])
    const out = areaResize({ width: 2, height: 2, data }, 1, 1)
    expect(Array.from(out.data)).toEqual([100, 100, 100, 255])
  })

  it('2.2: area averaging by 3:2 weights the shared middle pixel by half', () => {
    // Row of 3 pixels 0, 90, 180 to 2 pixels: [0 + 90/2] / 1.5 = 30, [90/2 + 180] / 1.5 = 150.
    const data = new Uint8ClampedArray([0, 0, 0, 255, 90, 90, 90, 255, 180, 180, 180, 255])
    const out = areaResize({ width: 3, height: 1, data }, 2, 1)
    expect(Array.from(out.data)).toEqual([30, 30, 30, 255, 150, 150, 150, 255])
  })

  // Feature: tile-reader, Property 4: Area shrink keeps a flat colour
  it('Property 4: Area shrink keeps a flat colour', () => {
    console.log(`fast-check seed ${FC_SEED}`)
    fc.assert(
      fc.property(
        // Scaled down by 10 for speed: short side 109 to 150 shrunk to 108 (1081 to 1500 to 1080).
        fc.integer({ min: 109, max: 150 }),
        fc.integer({ min: 0, max: 40 }),
        fc.boolean(),
        fc.tuple(fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 })),
        (s, extra, portrait, rgb) => {
          const long = s + extra
          const f = portrait ? solid(s, long, rgb) : solid(long, s, rgb)
          const out = shrinkToShortSide(f, 108)
          expect(Math.min(out.width, out.height)).toBe(108)
          let wrong = 0
          for (let p = 0; p < out.width * out.height; p++) {
            const q = p * 4
            if (out.data[q] !== rgb[0] || out.data[q + 1] !== rgb[1] || out.data[q + 2] !== rgb[2] || out.data[q + 3] !== 255) wrong++
          }
          expect(wrong).toBe(0)
        },
      ),
      { seed: FC_SEED, numRuns: 100 },
    )
  })
})
