import { describe, expect, it } from 'vitest'
import {
  CODEWORD_BITS,
  CRC_BITS,
  CRC_START,
  ID_BITS,
  ID_START,
  MESSAGE_BITS,
  PARITY_BITS,
  PARITY_START,
  SPARE_INDEX,
  decodeMessage,
  encodeMessage,
  type DecodeOptions,
} from '../message'
import { withFlips } from './message.helpers'

const EDGE_IDS = [0, 1, 2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1]
const OPTION_FORMS: (DecodeOptions | undefined)[] = [undefined, ...[0, 1, 2, 3, 4].map((m) => ({ maxCorrectableErrors: m }))]

describe('message edge cases', () => {
  it('round trips the edge IDs with every option form (Requirement 5.4)', () => {
    for (const id of EDGE_IDS) {
      for (const opts of OPTION_FORMS) {
        const r = opts === undefined ? decodeMessage(encodeMessage(id)) : decodeMessage(encodeMessage(id), opts)
        expect(r).not.toBeNull()
        expect(r!.id).toBe(id)
        expect(r!.id).toBeGreaterThanOrEqual(0)
        expect(r!.correctedBits).toBe(0)
      }
    }
  })

  for (const id of [0, 0xffffffff]) {
    it(`corrects every single and pair flip for id ${id} (Requirements 6.6, 6.7)`, () => {
      const m = encodeMessage(id)
      let singles = 0
      let pairs = 0
      for (let a = 0; a < 63; a++) {
        const r1 = decodeMessage(withFlips(m, [a]))
        expect(r1).toEqual({ id, correctedBits: 1 })
        singles++
        for (let b = a + 1; b < 63; b++) {
          const r2 = decodeMessage(withFlips(m, [a, b]))
          expect(r2).toEqual({ id, correctedBits: 2 })
          pairs++
        }
      }
      expect(singles).toBe(63)
      expect(pairs).toBe(1953)
    })

    it(`rejects every single flip at m=0 and every pair at m=1 for id ${id}`, () => {
      const m = encodeMessage(id)
      for (let a = 0; a < 63; a++) {
        expect(decodeMessage(withFlips(m, [a]), { maxCorrectableErrors: 0 })).toBeNull()
        for (let b = a + 1; b < 63; b++) {
          expect(decodeMessage(withFlips(m, [a, b]), { maxCorrectableErrors: 1 })).toBeNull()
        }
      }
    })
  }

  it('layout fields are contiguous and cover indices 0..63 exactly once (Requirement 2.6)', () => {
    const fields: [number, number][] = [
      [ID_START, ID_BITS],
      [CRC_START, CRC_BITS],
      [PARITY_START, PARITY_BITS],
      [SPARE_INDEX, 1],
    ]
    let next = 0
    for (const [start, width] of fields) {
      expect(start).toBe(next)
      next += width
    }
    expect(next).toBe(MESSAGE_BITS)
    expect(CODEWORD_BITS).toBe(63)
    expect(ID_BITS).toBe(32)
    expect(MESSAGE_BITS).toBe(64)
  })

  it('all-zero and all-ones Messages decode to null (blank and fully dark reads)', () => {
    expect(decodeMessage(new Uint8Array(64))).toBeNull()
    expect(decodeMessage(new Uint8Array(64).fill(1))).toBeNull()
    const ones63 = new Uint8Array(64).fill(1)
    ones63[63] = 0
    expect(decodeMessage(ones63)).toBeNull()
  })
})
