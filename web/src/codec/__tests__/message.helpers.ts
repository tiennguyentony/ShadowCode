// Shared helpers for the message-codec test suite.
// The reference CRC-7 and g(x) long division here are written independently
// of crc.ts and bch.ts, so the tests do not check the codec against itself.

import { readFileSync } from 'node:fs'
import { encodeMessage } from '../message'

export const FC_SEED = 0x7113a5c1
export const MEASUREMENT_SEED = 0x2545f491
export const GENUINE_SEED = 0x6a09e667

export const hex = (v: number, digits: number): string =>
  '0x' + (v >>> 0).toString(16).toUpperCase().padStart(digits, '0')

/** mulberry32: returns a function producing uint32 values. Public-domain algorithm. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return (t ^ (t >>> 14)) >>> 0
  }
}

/** Returns a copy of `bits` with the listed indices flipped. */
export function withFlips(bits: Uint8Array, flips: readonly number[]): Uint8Array {
  const out = bits.slice()
  for (const i of flips) out[i] ^= 1
  return out
}

/**
 * Partial Fisher–Yates draw of k distinct indices from [0, 63] (design D6).
 * Flips them in `m` in place and returns the drawn indices in draw order.
 */
export function drawAndFlip(m: Uint8Array, k: number, next: () => number): number[] {
  const idx = Array.from({ length: 64 }, (_, i) => i)
  for (let j = 0; j < k; j++) {
    const s = j + Math.floor((next() * (64 - j)) / 2 ** 32)
    const t = idx[j]
    idx[j] = idx[s]
    idx[s] = t
    m[idx[j]] ^= 1
  }
  return idx.slice(0, k)
}

/** Reads `width` bits MSB first starting at `start`, as an unsigned integer. */
export function readField(bits: ArrayLike<number>, start: number, width: number): number {
  let v = 0
  for (let i = 0; i < width; i++) v = v * 2 + bits[start + i]
  return v
}

/** Reference CRC-7 by bitwise long division of the augmented 39-bit dividend, then XOR 0x7F. */
export function refCrc7(id: number): number {
  const d: number[] = []
  for (let i = 31; i >= 0; i--) d.push(Math.floor(id / 2 ** i) % 2)
  for (let i = 0; i < 7; i++) d.push(0)
  const p = [1, 0, 0, 0, 1, 0, 0, 1]
  for (let i = 0; i <= d.length - p.length; i++) {
    if (d[i]) for (let j = 0; j < p.length; j++) d[i + j] ^= p[j]
  }
  return readField(d, d.length - 7, 7) ^ 0x7f
}

/** g(x) = 0x1DB2777 as 25 coefficients, highest degree first. */
export const G_COEFFS: readonly number[] = Array.from(
  { length: 25 },
  (_, i) => (0x1db2777 >>> (24 - i)) & 1,
)

/** Remainder of Σ bits[i]·x^(len−1−i) divided by g(x), as 24 coefficients (x^23 first). */
export function refModG(bits: ArrayLike<number>): number[] {
  const w = Array.from(bits)
  for (let i = 0; i <= w.length - 25; i++) {
    if (w[i]) for (let j = 0; j < 25; j++) w[i + j] ^= G_COEFFS[j]
  }
  return w.slice(w.length - 24)
}

/** Reference 64-bit Message for an ID and CRC_Field value (CRC may be wrong on purpose). */
export function refMessage(id: number, crc: number): Uint8Array {
  const m = new Uint8Array(64)
  for (let i = 0; i < 32; i++) m[i] = Math.floor(id / 2 ** (31 - i)) % 2
  for (let i = 0; i < 7; i++) m[32 + i] = (crc >>> (6 - i)) & 1
  const data = Array.from(m.subarray(0, 39))
  const parity = refModG([...data, ...new Array(24).fill(0)])
  for (let j = 0; j < 24; j++) m[39 + j] = parity[j]
  return m
}

export const bitString = (bits: ArrayLike<number>): string => Array.from(bits).join('')

// ---------------------------------------------------------------------------
// Format_Document vectors
// ---------------------------------------------------------------------------

export interface CleanVector {
  id: number
  idHex: string
  crcHex: string
  parityHex: string
  bits: string
  expected: { id: number; correctedBits: number }
}
export interface CorruptedVector {
  originalId: number
  originalIdHex: string
  flipped: number[]
  bits: string
  expected: { id: number; correctedBits: number } | null
}
export interface FormatVectors {
  blockCount: number
  fenceCount: number
  data: { formatVersion: number; clean: CleanVector[]; corrupted: CorruptedVector[] }
}

const BEGIN = '<!-- tilemark-test-vectors:begin -->'
const END = '<!-- tilemark-test-vectors:end -->'

export function loadFormatVectors(): FormatVectors {
  const url = new URL('../../../../docs/message-format.md', import.meta.url)
  const lines = readFileSync(url, 'utf8').replace(/\r\n/g, '\n').split('\n')
  const begins = lines.flatMap((l, i) => (l === BEGIN ? [i] : []))
  const ends = lines.flatMap((l, i) => (l === END ? [i] : []))
  const blockCount = Math.min(begins.length, ends.length)
  if (begins.length !== 1 || ends.length !== 1 || ends[0] < begins[0]) {
    throw new Error(`expected exactly one Vector_Block, found ${begins.length} begin / ${ends.length} end markers`)
  }
  const block = lines.slice(begins[0] + 1, ends[0]).join('\n')
  const fences = [...block.matchAll(/```json tilemark-test-vectors\n([\s\S]*?)\n```/g)]
  if (fences.length !== 1) throw new Error(`expected one json tilemark-test-vectors fence, found ${fences.length}`)
  return { blockCount, fenceCount: fences.length, data: JSON.parse(fences[0][1]) }
}

// ---------------------------------------------------------------------------
// Measurement output formatting
// ---------------------------------------------------------------------------

export interface Counts {
  n: number
  correct: number
  null: number
  falseAccept: number
}

export const emptyCounts = (): Counts => ({ n: 0, correct: 0, null: 0, falseAccept: 0 })

export function addCounts(a: Counts, b: Counts): Counts {
  return {
    n: a.n + b.n,
    correct: a.correct + b.correct,
    null: a.null + b.null,
    falseAccept: a.falseAccept + b.falseAccept,
  }
}

export const rate = (c: Counts): number => (c.n === 0 ? 0 : c.falseAccept / c.n)

export const fmtRate = (r: number): string => (r === 0 ? '0' : r.toExponential(2))

const withRate = (c: Counts) => ({ ...c, rate: rate(c) })

export function formatFalseAccept(
  label: string,
  seed: number,
  perK: Map<number, Counts>,
  k8: Counts,
  total: Counts,
): string {
  const json = {
    perK: Object.fromEntries([...perK].map(([k, c]) => [String(k), withRate(c)])),
    'k>=8': withRate(k8),
    total: withRate(total),
  }
  const rows = [
    ...[...perK].map(([k, c]) => [String(k), c] as const),
    ['k ≥ 8', k8] as const,
    ['all', total] as const,
  ].map(([k, c]) => `| ${k} | ${c.n} | ${c.correct} | ${c.null} | ${c.falseAccept} | ${fmtRate(rate(c))} |`)
  return [
    `[message-codec] ${label} seed=${hex(seed, 8)} ${JSON.stringify(json)}`,
    '| k | trials | correct | null | false accept | rate |',
    '|---|---|---|---|---|---|',
    ...rows,
    'note: k = 5, 6, 7 exceed the Correction_Limit (t = 4); a higher false-accept rate than k ≥ 8 is expected. No bound applies to them or to m = 3.',
  ].join('\n')
}

// ---------------------------------------------------------------------------
// Seeded draw procedures (design D6), shared by the measurement and spare-bit tests
// ---------------------------------------------------------------------------

export const K_VALUES = [5, 6, 7, 8, 12, 16, 32] as const
export const TRIALS_PER_K = 25_000
export const RANDOM_TRIALS = 100_000

export interface HeavyTrial {
  k: number
  id: number
  flips: number[]
  bits: Uint8Array
}

/** Requirement 7 draw procedure (design D6). Calls `visit` once per trial in draw order. */
export function forEachHeavyTrial(visit: (t: HeavyTrial) => void): void {
  const next = mulberry32(MEASUREMENT_SEED)
  for (const k of K_VALUES) {
    for (let t = 0; t < TRIALS_PER_K; t++) {
      const id = next()
      const bits = encodeMessage(id)
      const flips = drawAndFlip(bits, k, next)
      visit({ k, id, flips, bits })
    }
  }
}

/** Requirement 8 draw procedure (design D6). */
export function forEachRandomMessage(visit: (bits: Uint8Array) => void): void {
  const next = mulberry32(MEASUREMENT_SEED)
  for (let t = 0; t < RANDOM_TRIALS; t++) {
    const a = next()
    const b = next()
    const bits = new Uint8Array(64)
    for (let i = 0; i < 32; i++) {
      bits[i] = (a >>> (31 - i)) & 1
      bits[32 + i] = (b >>> (31 - i)) & 1
    }
    visit(bits)
  }
}

