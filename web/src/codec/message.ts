/**
 * Tilemark 64-bit Message codec, as defined in /docs/message-format.md.
 *
 * Layout: ID_Field 0-31 (MSB first), CRC_Field 32-38 (CRC-7, MSB first),
 * Parity_Field 39-62 (systematic BCH(63,39) over 0-38), Spare_Field 63
 * (written 0, ignored on decode: Spare_Bit_Option_B).
 *
 * Pure and stateless. Operates on plain Uint8Array data only.
 */

import { bchCorrect, bchParity } from './bch'
import { crc7 } from './crc'

export interface DecodeOptions {
  /** Integer 0..4. Default 4. A read needing more corrections is rejected before the CRC check. */
  maxCorrectableErrors?: number
}

export type DecodeResult = {
  /** Decoded Product_ID, unsigned, 0..2^32-1. */
  id: number
  /** Codeword positions (Message indices 0..62) the BCH decoder located and flipped. Never counts the Spare_Field. */
  correctedBits: number
} | null

export const MESSAGE_BITS = 64
export const ID_START = 0
export const ID_BITS = 32
export const CRC_START = 32
export const CRC_BITS = 7
export const PARITY_START = 39
export const PARITY_BITS = 24
export const SPARE_INDEX = 63
export const CODEWORD_BITS = 63
export const DEFAULT_MAX_CORRECTABLE_ERRORS = 4

/** Returns a new Uint8Array(64) of 0/1 values. Throws RangeError for an invalid id. */
export function encodeMessage(id: number): Uint8Array {
  if (typeof id !== 'number' || !Number.isInteger(id) || id < 0 || id > 0xffffffff) {
    throw new RangeError('encodeMessage: id must be an integer in [0, 4294967295]')
  }
  const v = id >>> 0 // -0 becomes 0
  const m = new Uint8Array(MESSAGE_BITS)
  for (let i = 0; i < ID_BITS; i++) m[ID_START + i] = (v >>> (31 - i)) & 1
  const c = crc7(v)
  for (let i = 0; i < CRC_BITS; i++) m[CRC_START + i] = (c >>> (6 - i)) & 1
  bchParity(m)
  // m[SPARE_INDEX] stays 0.
  return m
}

function describe(v: unknown): string {
  if (v === null) return 'null'
  if (typeof v === 'object') return (v as object).constructor?.name ?? 'object'
  return typeof v
}

function readLimit(options: unknown): number {
  if (options === undefined) return DEFAULT_MAX_CORRECTABLE_ERRORS
  if (typeof options !== 'object' || options === null) {
    throw new RangeError(`decodeMessage: options must be an object or undefined, got ${describe(options)}`)
  }
  const v: unknown = (options as DecodeOptions).maxCorrectableErrors
  if (v === undefined) return DEFAULT_MAX_CORRECTABLE_ERRORS
  if (typeof v !== 'number' || !Number.isInteger(v) || v < 0 || v > 4) {
    throw new RangeError(
      `decodeMessage: maxCorrectableErrors must be an integer in [0, 4], got ${String(v)}`,
    )
  }
  return v === 0 ? 0 : v // -0 counts as 0
}

/**
 * Decodes a 64-bit Message. Returns `{ id, correctedBits }`, or `null` if the
 * read is uncorrectable, needs more than `maxCorrectableErrors` corrections,
 * or fails the CRC. Throws RangeError only for invalid arguments.
 */
export function decodeMessage(bits: Uint8Array, options?: DecodeOptions): DecodeResult {
  if (!(bits instanceof Uint8Array)) {
    throw new RangeError(`decodeMessage: bits must be a Uint8Array, got ${describe(bits)}`)
  }
  if (bits.length !== MESSAGE_BITS) {
    throw new RangeError(`decodeMessage: bits must have length 64, got ${bits.length}`)
  }
  for (let i = 0; i < MESSAGE_BITS; i++) {
    const b = bits[i]
    if (b !== 0 && b !== 1) {
      throw new RangeError(`decodeMessage: bits[${i}] must be 0 or 1, got ${b}`)
    }
  }
  const limit = readLimit(options)

  // Codeword copy (indices 0..62). The Spare_Field (index 63) is never read.
  const word = bits.slice(0, CODEWORD_BITS)
  const n = bchCorrect(word)
  if (n < 0 || n > limit) return null

  let id = 0
  for (let i = 0; i < ID_BITS; i++) id = (id << 1) | word[ID_START + i]
  id >>>= 0
  let c = 0
  for (let i = 0; i < CRC_BITS; i++) c = (c << 1) | word[CRC_START + i]

  return c === crc7(id) ? { id, correctedBits: n } : null
}
