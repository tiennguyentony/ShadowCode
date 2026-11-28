/**
 * The Logit_Window: the Frame_Logits of the last Reads of one scan (tile-reader Requirement 7).
 * Pure.
 */
export class LogitWindow {
  readonly size: number
  private readonly frames: Float32Array[] = []

  constructor(size: number) {
    if (!Number.isInteger(size) || size < 1) throw new RangeError(`LogitWindow: size must be a positive integer, got ${size}`)
    this.size = size
  }

  /** Frames held now (0 to size). */
  get length(): number {
    return this.frames.length
  }

  /** Add one frame's logits, dropping the oldest beyond `size`. The values are copied. */
  push(logits: ArrayLike<number>): void {
    if (this.frames.length > 0 && logits.length !== this.frames[0].length) {
      throw new RangeError(`LogitWindow: expected ${this.frames[0].length} logits, got ${logits.length}`)
    }
    this.frames.push(Float32Array.from(logits))
    if (this.frames.length > this.size) this.frames.shift()
  }

  clear(): void {
    this.frames.length = 0
  }

  /** Logit-by-logit mean over the frames held, or null when empty. */
  mean(): Float32Array | null {
    if (this.frames.length === 0) return null
    return meanOf(this.frames)
  }
}

/** Logit-by-logit mean of equal-length vectors, summed in double precision. */
export function meanOf(vectors: readonly ArrayLike<number>[]): Float32Array {
  const n = vectors[0].length
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (const v of vectors) s += v[i]
    out[i] = s / vectors.length
  }
  return out
}

/** Bit i is 1 if logit i is greater than 0. */
export function logitsToBits(logits: ArrayLike<number>): Uint8Array {
  const bits = new Uint8Array(logits.length)
  for (let i = 0; i < logits.length; i++) bits[i] = logits[i] > 0 ? 1 : 0
  return bits
}
