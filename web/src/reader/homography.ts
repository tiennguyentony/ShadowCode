/**
 * Four-point homography, as in docs/model-contract.md ("What the reader is given").
 * The solve is the contract's JavaScript code, typed.
 */
import type { Point } from './types'

/** Nine numbers, row by row, with h[8] = 1. */
export type Homography = readonly number[]

/** Homography that takes from[k] to to[k] for k = 0..3 (exact four-point solve). */
export function homography(from: readonly Point[], to: readonly Point[]): Homography {
  const a: number[][] = []
  for (let k = 0; k < 4; k++) {
    const [x, y] = from[k]
    const [u, v] = to[k]
    a.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u])
    a.push([0, 0, 0, x, y, 1, -v * x, -v * y, v])
  }
  for (let col = 0; col < 8; col++) {
    let pivot = col
    for (let r = col + 1; r < 8; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r
    ;[a[col], a[pivot]] = [a[pivot], a[col]]
    for (let r = 0; r < 8; r++) {
      if (r === col) continue
      const f = a[r][col] / a[col][col]
      for (let c = col; c < 9; c++) a[r][c] -= f * a[col][c]
    }
  }
  return [...a.map((row, i) => row[8] / row[i]), 1]
}

/** Push one point through a homography. */
export function project(h: Homography, [x, y]: Point): Point {
  const w = h[6] * x + h[7] * y + h[8]
  return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w]
}
