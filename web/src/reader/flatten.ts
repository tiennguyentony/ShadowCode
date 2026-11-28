/**
 * The flatten step and the reader input, as in docs/model-contract.md
 * ("What the reader is given"). The loops are the contract's JavaScript code, typed.
 */
import { homography } from './homography'
import type { Point, RgbaFrame } from './types'

/** The parts of contract.json `flatten` this code uses. */
export interface FlattenSpec {
  readonly target_size: readonly [number, number]
  readonly target_corners: readonly Point[]
}

/**
 * Flatten the quadrilateral `corners` of `frame` to target_size.
 * Returns the reader input: Float32Array, NCHW, RGB, 0..1.
 * roundToLevels: only for checking against a fixture patch.
 */
export function flatten(frame: RgbaFrame, corners: readonly Point[], spec: FlattenSpec, roundToLevels = false): Float32Array {
  const [tw, th] = spec.target_size
  const h = homography(spec.target_corners, corners)
  const plane = tw * th
  const input = new Float32Array(3 * plane)
  const { width, height, data } = frame
  const at = (col: number, row: number, c: number): number =>
    col < 0 || row < 0 || col >= width || row >= height ? 0 : data[(row * width + col) * 4 + c]
  for (let i = 0; i < th; i++) {
    for (let j = 0; j < tw; j++) {
      const sx = j + 0.5
      const sy = i + 0.5
      const w = h[6] * sx + h[7] * sy + h[8]
      const fx = (h[0] * sx + h[1] * sy + h[2]) / w - 0.5
      const fy = (h[3] * sx + h[4] * sy + h[5]) / w - 0.5
      const x0 = Math.floor(fx)
      const y0 = Math.floor(fy)
      const ax = fx - x0
      const ay = fy - y0
      for (let c = 0; c < 3; c++) {
        let value =
          (1 - ay) * ((1 - ax) * at(x0, y0, c) + ax * at(x0 + 1, y0, c)) +
          ay * ((1 - ax) * at(x0, y0 + 1, c) + ax * at(x0 + 1, y0 + 1, c))
        if (roundToLevels) value = Math.round(value)
        input[c * plane + i * tw + j] = value / 255
      }
    }
  }
  return input
}

/** Already-flattened RGBA pixels (a fixture patch) to the reader input: NCHW, RGB, 0..1. */
export function imageToInput(pixels: RgbaFrame): Float32Array {
  const plane = pixels.width * pixels.height
  const input = new Float32Array(3 * plane)
  for (let p = 0; p < plane; p++) {
    for (let c = 0; c < 3; c++) input[c * plane + p] = pixels.data[p * 4 + c] / 255
  }
  return input
}
