/**
 * AprilTag 36h11 detection (tile-reader Requirement 3).
 *
 * js-aruco2 2.0.0 (MIT; vendored as ES modules in ./vendor/js-aruco2, design D1; the
 * 36h11 code list is from the AprilTag project, BSD-2-Clause) finds the tags on a
 * half-size copy of the frame. Its corners are whole contour pixel indices, so each
 * tag's four outer edges are then located to a fraction of a pixel on the full frame
 * and the corners are taken as the intersections of those edge lines. Corners and centres use the contract's pixel convention: the centre of
 * the pixel in column j, row i is (j + 0.5, i + 0.5).
 *
 * Pure: plain { width, height, data } RGBA pixels, no camera or DOM.
 */
import { AR } from './vendor/js-aruco2/aruco.js'
import './vendor/js-aruco2/apriltag_36h11.js'
import { areaResize } from './frame'
import type { Point, Quad, RgbaFrame } from './types'

export const TAG_DICTIONARY = 'APRILTAG_36h11'

/**
 * Largest Hamming distance accepted, plus one (js-aruco2 accepts distance < this).
 * 36h11 codes are at least 11 bits apart; up to 4 wrong bits are accepted, so a
 * wrong ID needs at least 7 wrong bits.
 */
export const TAG_MAX_HAMMING = 5

/** Cells across a 36h11 tag including its black border (6 data cells + 2 border cells). */
const TAG_CELLS = 8

export interface DetectedTag {
  readonly id: number
  /** Outer corners of the black border, in the detector's order (clockwise on screen). */
  readonly corners: Quad
  /** Mean of the four corners. */
  readonly centre: Point
  readonly hammingDistance: number
}

export interface DetectOptions {
  /** Tag IDs to keep; others are ignored. */
  readonly ids: readonly number[]
}

let detector: InstanceType<typeof AR.Detector> | null = null
function getDetector() {
  detector ??= new AR.Detector({ dictionaryName: TAG_DICTIONARY, maxHammingDistance: TAG_MAX_HAMMING })
  return detector
}

/** Luminance of the frame at (x, y), bilinear, contract pixel convention. Outside reads as 0. */
function greyAt(frame: RgbaFrame, x: number, y: number): number {
  const fx = x - 0.5
  const fy = y - 0.5
  const x0 = Math.floor(fx)
  const y0 = Math.floor(fy)
  const ax = fx - x0
  const ay = fy - y0
  const { width, height, data } = frame
  const g = (c: number, r: number): number => {
    if (c < 0 || r < 0 || c >= width || r >= height) return 0
    const p = (r * width + c) * 4
    return 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]
  }
  return (1 - ay) * ((1 - ax) * g(x0, y0) + ax * g(x0 + 1, y0)) + ay * ((1 - ax) * g(x0, y0 + 1) + ax * g(x0 + 1, y0 + 1))
}

interface Line {
  /** A point on the line and its unit direction. */
  readonly p: Point
  readonly d: Point
}

/** Total least squares line through points. */
function fitLine(points: readonly Point[]): Line {
  let mx = 0
  let my = 0
  for (const [x, y] of points) {
    mx += x
    my += y
  }
  mx /= points.length
  my /= points.length
  let sxx = 0
  let sxy = 0
  let syy = 0
  for (const [x, y] of points) {
    sxx += (x - mx) * (x - mx)
    sxy += (x - mx) * (y - my)
    syy += (y - my) * (y - my)
  }
  const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy)
  return { p: [mx, my], d: [Math.cos(angle), Math.sin(angle)] }
}

function intersect(a: Line, b: Line): Point | null {
  const den = a.d[0] * b.d[1] - a.d[1] * b.d[0]
  if (Math.abs(den) < 1e-9) return null
  const t = ((b.p[0] - a.p[0]) * b.d[1] - (b.p[1] - a.p[1]) * b.d[0]) / den
  return [a.p[0] + t * a.d[0], a.p[1] + t * a.d[1]]
}

/**
 * Edge points of one side of the black border: along the outward normal, the place
 * where the luminance crosses halfway between the dark and the light level.
 */
function edgePoints(frame: RgbaFrame, a: Point, b: Point, centre: Point, range: number): Point[] {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  const len = Math.hypot(dx, dy)
  const t: Point = [dx / len, dy / len]
  let n: Point = [-t[1], t[0]]
  const mid: Point = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
  if ((mid[0] - centre[0]) * n[0] + (mid[1] - centre[1]) * n[1] < 0) n = [-n[0], -n[1]]
  const step = 0.25
  const count = Math.round((2 * range) / step) + 1
  const values = new Float64Array(count)
  const points: Point[] = []
  const samples = 24
  for (let s = 0; s < samples; s++) {
    const u = 0.15 + (0.7 * s) / (samples - 1)
    const bx = a[0] + u * dx
    const by = a[1] + u * dy
    let lo = Infinity
    let hi = -Infinity
    for (let k = 0; k < count; k++) {
      const d = -range + k * step
      const v = greyAt(frame, bx + d * n[0], by + d * n[1])
      values[k] = v
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
    if (hi - lo < 20) continue
    const thr = (lo + hi) / 2
    // Dark inside, light outside: the crossing nearest to the current edge.
    let best: number | null = null
    for (let k = 0; k + 1 < count; k++) {
      if (values[k] < thr && values[k + 1] >= thr) {
        const d = -range + k * step + ((thr - values[k]) / (values[k + 1] - values[k])) * step
        if (best === null || Math.abs(d) < Math.abs(best)) best = d
      }
    }
    if (best !== null) points.push([bx + best * n[0], by + best * n[1]])
  }
  return points
}

/** Locate the four outer edges to a fraction of a pixel and return their intersections. */
export function refineCorners(frame: RgbaFrame, rough: Quad): Quad {
  let corners: Point[] = rough.slice()
  for (let pass = 0; pass < 2; pass++) {
    const centre = meanPoint(corners)
    const side = (Math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1]) +
      Math.hypot(corners[2][0] - corners[1][0], corners[2][1] - corners[1][1]) +
      Math.hypot(corners[3][0] - corners[2][0], corners[3][1] - corners[2][1]) +
      Math.hypot(corners[0][0] - corners[3][0], corners[0][1] - corners[3][1])) / 4
    const cell = side / TAG_CELLS
    const range = pass === 0 ? Math.max(1.5, 0.6 * cell) : Math.max(1, 0.3 * cell)
    const lines: Line[] = []
    for (let k = 0; k < 4; k++) {
      const pts = edgePoints(frame, corners[k], corners[(k + 1) % 4], centre, range)
      if (pts.length < 6) return quad(corners)
      lines.push(fitLine(pts))
    }
    const next: Point[] = []
    for (let k = 0; k < 4; k++) {
      const p = intersect(lines[(k + 3) % 4], lines[k])
      if (!p || Math.hypot(p[0] - corners[k][0], p[1] - corners[k][1]) > cell) return quad(corners)
      next.push(p)
    }
    corners = next
  }
  return quad(corners)
}

function quad(p: readonly Point[]): Quad {
  return [p[0], p[1], p[2], p[3]]
}

export function meanPoint(points: readonly Point[]): Point {
  let x = 0
  let y = 0
  for (const p of points) {
    x += p[0]
    y += p[1]
  }
  return [x / points.length, y / points.length]
}

/** Run js-aruco2 on an image and map its contour corners to frame coordinates. */
function rawDetect(image: RgbaFrame, sx: number, sy: number, ids: readonly number[]): Map<number, { corners: Quad; hammingDistance: number }> {
  const found = new Map<number, { corners: Quad; hammingDistance: number }>()
  for (const m of getDetector().detect(image)) {
    if (!ids.includes(m.id)) continue
    const prev = found.get(m.id)
    if (prev && prev.hammingDistance <= m.hammingDistance) continue
    // Contour index q is the pixel whose centre is q + 0.5; scaled back to the frame.
    const corners = quad(m.corners.map((c): Point => [(c.x + 0.5) * sx, (c.y + 0.5) * sy]))
    found.set(m.id, { corners, hammingDistance: m.hammingDistance })
  }
  return found
}

/**
 * Find the tags with the given IDs. Detection runs on a half-size copy first; if a
 * wanted tag is missing there, it runs again on the full frame. Each tag found is
 * refined on the full frame.
 */
export function detectTags(frame: RgbaFrame, options: DetectOptions): Map<number, DetectedTag> {
  const { ids } = options
  const halfW = Math.floor(frame.width / 2)
  const halfH = Math.floor(frame.height / 2)
  let raw = rawDetect(areaResize(frame, halfW, halfH), frame.width / halfW, frame.height / halfH, ids)
  if (ids.some((id) => !raw.has(id))) {
    const full = rawDetect(frame, 1, 1, ids)
    for (const [id, t] of full) if (!raw.has(id)) raw.set(id, t)
    raw = new Map([...raw].sort((a, b) => a[0] - b[0]))
  }
  const tags = new Map<number, DetectedTag>()
  for (const [id, t] of raw) {
    const corners = refineCorners(frame, t.corners)
    tags.set(id, { id, corners, centre: meanPoint(corners), hammingDistance: t.hammingDistance })
  }
  return tags
}
