// Tilemark Message test-vector generator (Vector_Generator).
//
// Independent of /web/src/codec by design (Requirement 14.2): this file imports
// only Node built-ins and computes CRC-7 and BCH(63,39) parity with explicit
// 0/1 coefficient arrays and augmented long division, a different technique
// from the codec's packed-integer shift registers.
//
// Coefficient arrays: plain JS arrays of 0/1, index 0 = highest degree.
//
// Usage (from /web): node scripts/gen-message-vectors.mjs [--check]

import process from 'node:process'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

// ---------------------------------------------------------------------------
// Parameters copied from the Format_Document (/docs/message-format.md)
// ---------------------------------------------------------------------------

/** CRC-7 polynomial x^7 + x^3 + 1, highest degree first. */
const CRC_POLY = [1, 0, 0, 0, 1, 0, 0, 1]
const CRC_FINAL_XOR = 0x7f

/** BCH(63,39) generator g(x) as stated in the Format_Document. */
const G_CONSTANT = 0x1db2777
const G_DEGREE = 24

/** GF(2^6) primitive polynomial p(x) = x^6 + x + 1. */
const GF_PRIM = 0x43
const GF_M = 6
const GF_N = 63 // 2^6 - 1

const ID_BITS = 32
const CRC_BITS = 7
const DATA_BITS = ID_BITS + CRC_BITS // 39
const CODEWORD_BITS = 63
const MESSAGE_BITS = 64

// ---------------------------------------------------------------------------
// Test-vector inputs (Requirement 14.4)
// ---------------------------------------------------------------------------

export const CLEAN_IDS = [0, 1, 0x12345678, 0x80000000, 0xdeadbeef, 0xffffffff]

export const CORRUPTED = [
  {
    originalId: 0x12345678,
    flipped: [3, 17, 35, 58],
    expected: { id: 0x12345678, correctedBits: 4 },
  },
  {
    originalId: 0xdeadbeef,
    flipped: [0, 9, 21, 33, 44, 52, 60, 63],
    expected: null,
  },
]

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export class SelfCheckError extends Error {
  constructor(detail) {
    super(`generator self-check failed: ${detail}`)
    this.name = 'SelfCheckError'
    this.exitCode = 3
  }
}

function check(condition, detail) {
  if (!condition) throw new SelfCheckError(detail)
}

// ---------------------------------------------------------------------------
// Coefficient-array helpers
// ---------------------------------------------------------------------------

/** Integer -> coefficient array of `width` entries, MSB first. */
export function intToBits(value, width) {
  const out = new Array(width)
  for (let i = 0; i < width; i++) {
    // Division instead of >>> so widths above 31 bits stay exact.
    out[i] = Math.floor(value / 2 ** (width - 1 - i)) % 2
  }
  return out
}

/** Coefficient array (MSB first) -> non-negative integer. */
export function bitsToInt(bits) {
  let v = 0
  for (const b of bits) v = v * 2 + b
  return v
}

/**
 * Remainder of dividend / divisor over GF(2) by augmented long division.
 * For i from 0 to dividend.length - divisor.length, if the working copy has a
 * 1 at i, XOR the divisor in at i. Returns the last divisor.length - 1 entries.
 */
export function polyMod(dividend, divisor) {
  check(divisor[0] === 1, 'polyMod divisor must have a leading 1')
  const work = dividend.slice()
  const d = divisor.length
  for (let i = 0; i <= work.length - d; i++) {
    if (work[i] === 1) {
      for (let j = 0; j < d; j++) work[i + j] ^= divisor[j]
    }
  }
  return work.slice(work.length - (d - 1))
}

/** Product of two GF(2) coefficient arrays (MSB first). */
function polyMulGF2(a, b) {
  const out = new Array(a.length + b.length - 1).fill(0)
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== 1) continue
    for (let j = 0; j < b.length; j++) out[i + j] ^= b[j]
  }
  return out
}

// ---------------------------------------------------------------------------
// CRC-7 and BCH parity
// ---------------------------------------------------------------------------

/** CRC-7 core (poly x^7+x^3+1, init 0, no reflection) over an MSB-first bit array. */
function crc7Core(bits) {
  return bitsToInt(polyMod([...bits, ...new Array(CRC_BITS).fill(0)], CRC_POLY))
}

/** CRC_Field value for a 32-bit Product_ID: core over the 32 ID bits, then XOR 0x7F. */
export function crc7(id) {
  return crc7Core(intToBits(id, ID_BITS)) ^ CRC_FINAL_XOR
}

/** g(x) as a 25-entry coefficient array, from the Format_Document constant. */
const G_BITS = intToBits(G_CONSTANT, G_DEGREE + 1)

/** 24-entry Parity_Field (x^23 first) for 39 data bits. */
export function bchParity(dataBits) {
  check(dataBits.length === DATA_BITS, `bchParity expects ${DATA_BITS} data bits`)
  return polyMod([...dataBits, ...new Array(G_DEGREE).fill(0)], G_BITS)
}

/** Full 64-entry Message (0/1 array) for a Product_ID. Spare bit (index 63) is 0. */
export function encodeVector(id) {
  const idBits = intToBits(id, ID_BITS)
  const crc = crc7(id)
  const crcBits = intToBits(crc, CRC_BITS)
  const data = [...idBits, ...crcBits]
  const parity = bchParity(data)
  const codeword = [...data, ...parity]
  check(codeword.length === CODEWORD_BITS, 'codeword length is not 63')
  check(
    polyMod(codeword, G_BITS).every((b) => b === 0),
    `codeword for id 0x${hex(id, 8)} is not divisible by g(x)`,
  )
  return { bits: [...codeword, 0], crc, parity: bitsToInt(parity) }
}

// ---------------------------------------------------------------------------
// g(x) re-derivation from GF(2^6) minimal polynomials
// ---------------------------------------------------------------------------

/** Build exp/log tables for GF(2^6) with p(x) = x^6 + x + 1. */
function buildGF() {
  const exp = new Array(2 * GF_N)
  const log = new Array(GF_N + 1).fill(-1)
  let x = 1
  for (let i = 0; i < GF_N; i++) {
    check(log[x] === -1, 'p(x) is not primitive')
    exp[i] = x
    log[x] = i
    x <<= 1
    if (x & (1 << GF_M)) x ^= GF_PRIM
  }
  check(x === 1, 'alpha^63 != 1')
  for (let i = GF_N; i < 2 * GF_N; i++) exp[i] = exp[i - GF_N]
  const mul = (a, b) => (a === 0 || b === 0 ? 0 : exp[log[a] + log[b]])
  return { exp, mul }
}

/** Cyclotomic coset of s modulo 63 under multiplication by 2. */
function cyclotomicCoset(s) {
  const coset = []
  let j = s % GF_N
  while (!coset.includes(j)) {
    coset.push(j)
    j = (j * 2) % GF_N
  }
  return coset
}

/**
 * Minimal polynomial of alpha^s: product of (x - alpha^j) over the coset of s,
 * with GF(2^6) coefficient arithmetic. Returned as a GF(2) array, MSB first.
 */
function minimalPolynomial(gf, s) {
  // Work low-degree-first while multiplying: poly[k] = coefficient of x^k.
  let poly = [1]
  for (const j of cyclotomicCoset(s)) {
    const root = gf.exp[j]
    const next = new Array(poly.length + 1).fill(0)
    for (let k = 0; k < poly.length; k++) {
      next[k + 1] ^= poly[k] // x * poly
      next[k] ^= gf.mul(root, poly[k]) // root * poly (minus = plus in char 2)
    }
    poly = next
  }
  check(
    poly.every((c) => c === 0 || c === 1),
    `minimal polynomial of alpha^${s} has coefficients outside GF(2)`,
  )
  return poly.reverse()
}

/** Re-derive g(x) = m1 * m3 * m5 * m7 and return it as an integer. */
export function deriveGenerator() {
  const gf = buildGF()
  let g = [1]
  for (const s of [1, 3, 5, 7]) g = polyMulGF2(g, minimalPolynomial(gf, s))
  return bitsToInt(g)
}

// ---------------------------------------------------------------------------
// Self-checks and constant validation
// ---------------------------------------------------------------------------

export function runSelfChecks() {
  // CRC core against the catalogued CRC-7/MMC check value over ASCII "123456789".
  const ascii = []
  for (const ch of '123456789') ascii.push(...intToBits(ch.charCodeAt(0), 8))
  const core = crc7Core(ascii)
  check(core === 0x75, `CRC-7 core check value is 0x${hex(core, 2)}, expected 0x75`)

  // g(x) re-derived from minimal polynomials equals the document constant.
  const derived = deriveGenerator()
  check(
    derived === G_CONSTANT,
    `derived g(x) = 0x${hex(derived, 7)} does not equal constant 0x${hex(G_CONSTANT, 7)}`,
  )

  // g(x) divides x^63 + 1.
  const x63plus1 = new Array(GF_N + 1).fill(0)
  x63plus1[0] = 1
  x63plus1[GF_N] = 1
  check(polyMod(x63plus1, G_BITS).every((b) => b === 0), 'g(x) does not divide x^63 + 1')
}

const isU32 = (v) => Number.isInteger(v) && v >= 0 && v <= 0xffffffff

export function validateConstants() {
  check(CLEAN_IDS.every(isU32), 'CLEAN_IDS contains a value outside [0, 2^32 - 1]')
  check(new Set(CLEAN_IDS).size >= 5, 'CLEAN_IDS needs at least 5 distinct IDs')
  check(new Set(CLEAN_IDS).size === CLEAN_IDS.length, 'CLEAN_IDS contains duplicates')
  for (const required of [0, 1, 0xffffffff]) {
    check(CLEAN_IDS.includes(required), `CLEAN_IDS must include 0x${hex(required, 8)}`)
  }

  for (const [i, v] of CORRUPTED.entries()) {
    check(isU32(v.originalId), `corrupted[${i}] originalId outside [0, 2^32 - 1]`)
    check(
      v.flipped.every((f) => Number.isInteger(f) && f >= 0 && f < MESSAGE_BITS),
      `corrupted[${i}] has a flipped index outside [0, 63]`,
    )
    check(new Set(v.flipped).size === v.flipped.length, `corrupted[${i}] flipped indices are not distinct`)
  }

  const fourFlip = CORRUPTED.filter((v) => v.expected !== null && v.expected.correctedBits === 4)
  check(fourFlip.length === 1, 'expected exactly one corrupted vector with correctedBits 4')
  const [ff] = fourFlip
  check(ff.flipped.length === 4, 'the 4-flip vector must have exactly 4 flipped indices')
  check(ff.flipped.every((f) => f <= 62), 'the 4-flip vector must flip only Codeword indices [0, 62]')
  check(ff.expected.id === ff.originalId, 'the 4-flip vector must expect its original ID')

  check(CORRUPTED.filter((v) => v.expected === null).length === 1, 'expected exactly one corrupted vector with expected null')
}

// ---------------------------------------------------------------------------
// Vector computation
// ---------------------------------------------------------------------------

export function hex(value, digits) {
  return value.toString(16).toUpperCase().padStart(digits, '0')
}

/**
 * Run all self-checks, then compute every Test_Vector. Throws SelfCheckError
 * (exit code 3) on any failure, before any document is touched.
 */
export function computeVectors() {
  runSelfChecks()
  validateConstants()

  const clean = CLEAN_IDS.map((id) => {
    const { bits, crc, parity } = encodeVector(id)
    return {
      id,
      idHex: `0x${hex(id, 8)}`,
      crcHex: `0x${hex(crc, 2)}`,
      parityHex: `0x${hex(parity, 6)}`,
      bits: bits.join(''),
      expected: { id, correctedBits: 0 },
    }
  })

  const corrupted = CORRUPTED.map((v) => {
    const bits = encodeVector(v.originalId).bits
    for (const f of v.flipped) bits[f] ^= 1
    return {
      originalId: v.originalId,
      originalIdHex: `0x${hex(v.originalId, 8)}`,
      flipped: v.flipped.slice(),
      bits: bits.join(''),
      expected: v.expected === null ? null : { id: v.expected.id, correctedBits: v.expected.correctedBits },
    }
  })

  return { formatVersion: 1, clean, corrupted }
}

// ---------------------------------------------------------------------------
// Entry point (output, write mode and --check are added in task 4.3)
// ---------------------------------------------------------------------------

function main() {
  try {
    computeVectors()
  } catch (err) {
    if (err instanceof SelfCheckError) {
      process.stderr.write(`${err.message}\n`)
      process.exit(err.exitCode)
    }
    throw err
  }
  process.exit(0)
}

const invokedDirectly =
  process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)

if (invokedDirectly) main()
