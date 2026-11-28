/**
 * From tag centres to flatten corners (tile-reader Requirements 4 and 5).
 * Pure: numbers in, numbers out.
 */
import { homography, project } from './homography'
import { tileCornersMm, type LabelLayout } from './layout'
import type { Point, Quad } from './types'

export type Located =
  | { readonly ok: true; readonly tileCorners: readonly Quad[]; readonly pxPerTile: number }
  | { readonly ok: false; readonly missingTags: readonly number[] }

/** Mean length of the four sides of a quadrilateral. */
export function meanSide(q: Quad): number {
  let s = 0
  for (let k = 0; k < 4; k++) {
    const a = q[k]
    const b = q[(k + 1) % 4]
    s += Math.hypot(b[0] - a[0], b[1] - a[1])
  }
  return s / 4
}

/**
 * Fit the homography from the tag centres in label mm to their centres in frame
 * pixels (exact four-point solve) and push each whole tile's printed corners through
 * it. The corner order follows the tag IDs, so a label held upside down still gives
 * the printed top-left corner first.
 */
export function locateTiles(tagCentres: ReadonlyMap<number, Point>, layout: LabelLayout): Located {
  const ids = [...layout.tagCentresMm.keys()]
  const missingTags = ids.filter((id) => !tagCentres.has(id))
  if (missingTags.length > 0) return { ok: false, missingTags }
  const from = ids.map((id) => layout.tagCentresMm.get(id) as Point)
  const to = ids.map((id) => tagCentres.get(id) as Point)
  const h = homography(from, to)
  const tileCorners = layout.wholeTiles.map((tile) => {
    const [a, b, c, d] = tileCornersMm(tile, layout.tileMm).map((p) => project(h, p))
    return [a, b, c, d] as Quad
  })
  const pxPerTile = tileCorners.reduce((s, q) => s + meanSide(q), 0) / tileCorners.length
  return { ok: true, tileCorners, pxPerTile }
}
