// Properties 2 and 3: the homography and the tile corner order.
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { homography, project } from '../homography'
import { LABEL_LAYOUT, tileCornersMm } from '../layout'
import { locateTiles, meanSide } from '../locate'
import type { Point } from '../types'

const FC_SEED = 0x711e4ead

function convex(q: readonly Point[]): boolean {
  let sign = 0
  for (let k = 0; k < 4; k++) {
    const a = q[k]
    const b = q[(k + 1) % 4]
    const c = q[(k + 2) % 4]
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    if (Math.abs(cross) < 1e-6) return false
    if (sign === 0) sign = Math.sign(cross)
    else if (Math.sign(cross) !== sign) return false
  }
  return true
}

// A convex quadrilateral in clockwise screen order: one corner per quadrant around a centre.
const quadArb = fc
  .tuple(
    fc.tuple(fc.double({ min: 0, max: 1920, noNaN: true }), fc.double({ min: 0, max: 1920, noNaN: true })),
    fc.array(fc.tuple(fc.double({ min: 5, max: 85, noNaN: true }), fc.double({ min: 20, max: 800, noNaN: true })), { minLength: 4, maxLength: 4 }),
    fc.double({ min: 0, max: 360, noNaN: true }),
  )
  .map(([[cx, cy], parts, turn]) =>
    parts.map(([deg, r], k): Point => {
      const a = ((turn + 180 + k * 90 + deg) * Math.PI) / 180
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)]
    }),
  )

describe('geometry properties', () => {
  // Feature: tile-reader, Property 2: Homography fits its four points
  it('Property 2: Homography fits its four points', () => {
    console.log(`fast-check seed ${FC_SEED}`)
    const from: Point[] = [[0, 0], [384, 0], [384, 384], [0, 384]]
    fc.assert(
      fc.property(quadArb, (to) => {
        fc.pre(convex(to) && to.every((p, k) => Math.hypot(p[0] - to[(k + 1) % 4][0], p[1] - to[(k + 1) % 4][1]) > 10))
        const h = homography(from, to)
        for (let k = 0; k < 4; k++) {
          const [x, y] = project(h, from[k])
          expect(Math.abs(x - to[k][0])).toBeLessThan(1e-6)
          expect(Math.abs(y - to[k][1])).toBeLessThan(1e-6)
        }
      }),
      { seed: FC_SEED, numRuns: 500 },
    )
  })

  // Feature: tile-reader, Property 3: Tile corner order follows the tags
  it('Property 3: Tile corner order follows the tags', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0, max: 360, maxExcluded: true, noNaN: true }),
        fc.double({ min: 2, max: 8, noNaN: true }),
        fc.tuple(fc.double({ min: 300, max: 800, noNaN: true }), fc.double({ min: 500, max: 1400, noNaN: true })),
        (deg, pxPerMm, [cx, cy]) => {
          const a = (deg * Math.PI) / 180
          const [lw, lh] = LABEL_LAYOUT.labelMm
          // Label mm to frame px: turn about the label centre, scale, move.
          const T = ([x, y]: Point): Point => {
            const dx = (x - lw / 2) * pxPerMm
            const dy = (y - lh / 2) * pxPerMm
            return [cx + dx * Math.cos(a) - dy * Math.sin(a), cy + dx * Math.sin(a) + dy * Math.cos(a)]
          }
          const centres = new Map([...LABEL_LAYOUT.tagCentresMm].map(([id, c]) => [id, T(c)]))
          const located = locateTiles(centres, LABEL_LAYOUT)
          expect(located.ok).toBe(true)
          if (!located.ok) return
          LABEL_LAYOUT.wholeTiles.forEach((tile, i) => {
            const want = tileCornersMm(tile, LABEL_LAYOUT.tileMm).map(T)
            located.tileCorners[i].forEach((p, k) => {
              expect(Math.abs(p[0] - want[k][0])).toBeLessThan(1e-6)
              expect(Math.abs(p[1] - want[k][1])).toBeLessThan(1e-6)
            })
            expect(meanSide(located.tileCorners[i])).toBeCloseTo(LABEL_LAYOUT.tileMm * pxPerMm, 6)
          })
          expect(located.pxPerTile).toBeCloseTo(LABEL_LAYOUT.tileMm * pxPerMm, 6)
        },
      ),
      { seed: FC_SEED, numRuns: 500 },
    )
  })

  it('locateTiles names missing tags in increasing order', () => {
    const centres = new Map<number, Point>([[1, [0, 0]], [3, [5, 5]]])
    expect(locateTiles(centres, LABEL_LAYOUT)).toEqual({ ok: false, missingTags: [0, 2] })
  })
})
