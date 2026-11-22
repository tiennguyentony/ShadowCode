import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('../crc', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../crc')>()
  return { crc7: vi.fn(actual.crc7) }
})

import { crc7 } from '../crc'
import { decodeMessage, encodeMessage } from '../message'
import { withFlips } from './message.helpers'

const valid = () => encodeMessage(0x12345678)

describe('encodeMessage validation (Requirement 1.4)', () => {
  const bad: unknown[] = [NaN, Infinity, -Infinity, 0.5, -1, 2 ** 32, '1', 1n, null, undefined]
  for (const v of bad) {
    it(`throws RangeError for ${String(v)}`, () => {
      expect(() => encodeMessage(v as number)).toThrow(RangeError)
    })
  }
  it('accepts 0 and -0, both encoding as 0', () => {
    expect(Array.from(encodeMessage(-0))).toEqual(Array.from(encodeMessage(0)))
  })
})

describe('decodeMessage bits validation (Requirements 3.7, 3.8)', () => {
  const badBits: [string, unknown][] = [
    ['null', null],
    ['undefined', undefined],
    ['plain array', Array.from(valid())],
    ['Int8Array', new Int8Array(64)],
    ['Uint8ClampedArray', new Uint8ClampedArray(64)],
    ['length 0', new Uint8Array(0)],
    ['length 63', new Uint8Array(63)],
    ['length 65', new Uint8Array(65)],
  ]
  for (const [name, v] of badBits) {
    it(`throws RangeError for ${name}`, () => {
      expect(() => decodeMessage(v as Uint8Array)).toThrow(RangeError)
    })
  }
  for (const value of [2, 255]) {
    it(`throws RangeError naming index and value for element ${value}`, () => {
      const b = valid()
      b[17] = value
      expect(() => decodeMessage(b)).toThrow(new RegExp(`bits\\[17\\].*${value}`))
    })
  }
})

describe('decodeMessage options validation (Requirements 4.1-4.4)', () => {
  const badOptions: [string, unknown][] = [
    ['null', null],
    ['number', 4],
    ['string', 'x'],
    ['boolean', true],
  ]
  for (const [name, v] of badOptions) {
    it(`throws RangeError for options ${name}`, () => {
      expect(() => decodeMessage(valid(), v as object)).toThrow(RangeError)
    })
  }
  const badLimits: unknown[] = ['4', 4n, true, null, NaN, Infinity, -Infinity, 0.5, 3.9, -1, 5]
  for (const v of badLimits) {
    it(`throws RangeError for maxCorrectableErrors ${String(v)}`, () => {
      expect(() => decodeMessage(valid(), { maxCorrectableErrors: v as number })).toThrow(RangeError)
    })
  }
  it('accepts -0 as 0', () => {
    expect(decodeMessage(valid(), { maxCorrectableErrors: -0 })).toEqual({ id: 0x12345678, correctedBits: 0 })
  })
})

describe('CRC is not computed after a BCH rejection (Requirements 3.4, 4.5)', () => {
  beforeEach(() => vi.mocked(crc7).mockClear())

  it('is not called when bchCorrect fails', () => {
    const r = withFlips(valid(), [0, 9, 21, 33, 44, 52, 60])
    vi.mocked(crc7).mockClear()
    expect(decodeMessage(r)).toBeNull()
    expect(crc7).not.toHaveBeenCalled()
  })

  it('is not called when corrections exceed the limit', () => {
    const r = withFlips(valid(), [3, 17, 35])
    vi.mocked(crc7).mockClear()
    expect(decodeMessage(r, { maxCorrectableErrors: 2 })).toBeNull()
    expect(crc7).not.toHaveBeenCalled()
  })

  it('is called on an accepted read', () => {
    const r = valid()
    vi.mocked(crc7).mockClear()
    expect(decodeMessage(r)).not.toBeNull()
    expect(crc7).toHaveBeenCalledTimes(1)
  })
})
