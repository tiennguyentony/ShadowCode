/**
 * Frame checks and the shrink step (tile-reader Requirement 2).
 * Pure: plain { width, height, data } RGBA pixels, no camera or DOM.
 */
import type { RgbaFrame } from './types'

/** Throws a RangeError unless width and height are positive integers and data holds 4 bytes per pixel. */
export function checkFrame(frame: RgbaFrame): void {
  const { width, height, data } = frame
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError(`RgbaFrame: width and height must be positive integers, got ${width} x ${height}`)
  }
  if (data.length !== width * height * 4) {
    throw new RangeError(`RgbaFrame: data has ${data.length} bytes, expected ${width * height * 4} for ${width} x ${height}`)
  }
}

/** Per output index: the input indices it covers and their weights (overlap / scale), summing to 1. */
function areaWeights(inSize: number, outSize: number): { index: Int32Array; weight: Float64Array; start: Int32Array } {
  const scale = inSize / outSize
  const start = new Int32Array(outSize + 1)
  const index: number[] = []
  const weight: number[] = []
  for (let o = 0; o < outSize; o++) {
    start[o] = index.length
    const lo = o * scale
    const hi = (o + 1) * scale
    for (let i = Math.floor(lo); i < Math.min(inSize, Math.ceil(hi)); i++) {
      const overlap = Math.min(hi, i + 1) - Math.max(lo, i)
      if (overlap > 1e-12) {
        index.push(i)
        weight.push(overlap / scale)
      }
    }
  }
  start[outSize] = index.length
  return { index: Int32Array.from(index), weight: Float64Array.from(weight), start }
}

/**
 * Resize by area averaging: each output pixel is the mean of the input area it covers.
 * Output pixel centre (j + 0.5) sits at input (j + 0.5) * inWidth / outWidth, the same
 * pixel convention as the contract. Alpha is set to 255.
 */
export function areaResize(frame: RgbaFrame, outWidth: number, outHeight: number): RgbaFrame {
  checkFrame(frame)
  const { width, height, data } = frame
  const wx = areaWeights(width, outWidth)
  const wy = areaWeights(height, outHeight)
  // Horizontal pass: height rows x outWidth columns x 3 channels.
  const mid = new Float64Array(height * outWidth * 3)
  for (let r = 0; r < height; r++) {
    const rowIn = r * width * 4
    const rowMid = r * outWidth * 3
    for (let o = 0; o < outWidth; o++) {
      let s0 = 0
      let s1 = 0
      let s2 = 0
      for (let k = wx.start[o]; k < wx.start[o + 1]; k++) {
        const p = rowIn + wx.index[k] * 4
        const w = wx.weight[k]
        s0 += w * data[p]
        s1 += w * data[p + 1]
        s2 += w * data[p + 2]
      }
      const q = rowMid + o * 3
      mid[q] = s0
      mid[q + 1] = s1
      mid[q + 2] = s2
    }
  }
  const out = new Uint8ClampedArray(outWidth * outHeight * 4)
  for (let o = 0; o < outHeight; o++) {
    for (let c = 0; c < outWidth; c++) {
      let s0 = 0
      let s1 = 0
      let s2 = 0
      for (let k = wy.start[o]; k < wy.start[o + 1]; k++) {
        const q = (wy.index[k] * outWidth + c) * 3
        const w = wy.weight[k]
        s0 += w * mid[q]
        s1 += w * mid[q + 1]
        s2 += w * mid[q + 2]
      }
      const p = (o * outWidth + c) * 4
      out[p] = Math.round(s0)
      out[p + 1] = Math.round(s1)
      out[p + 2] = Math.round(s2)
      out[p + 3] = 255
    }
  }
  return { width: outWidth, height: outHeight, data: out }
}

/**
 * Requirement 2: a frame whose short side is over maxShortSide pixels is shrunk by
 * area averaging so that its short side is exactly maxShortSide; the long side is
 * scaled by the same factor and rounded. Other frames are returned unchanged.
 */
export function shrinkToShortSide(frame: RgbaFrame, maxShortSide: number): RgbaFrame {
  checkFrame(frame)
  const short = Math.min(frame.width, frame.height)
  if (short <= maxShortSide) return frame
  const factor = maxShortSide / short
  const outWidth = frame.width <= frame.height ? maxShortSide : Math.round(frame.width * factor)
  const outHeight = frame.width <= frame.height ? Math.round(frame.height * factor) : maxShortSide
  return areaResize(frame, outWidth, outHeight)
}
