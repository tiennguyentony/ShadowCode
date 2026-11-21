// Feature: message-codec, Property 16: Format_Document vector conformance
import { describe, expect, it } from 'vitest'
import { decodeMessage, encodeMessage } from '../message'
import { bitString, hex, loadFormatVectors, readField } from './message.helpers'

const { blockCount, fenceCount, data } = loadFormatVectors()

describe('Format_Document Vector_Block shape', () => {
  it('has exactly one Vector_Block and one json fence', () => {
    expect(blockCount).toBe(1)
    expect(fenceCount).toBe(1)
    expect(data.formatVersion).toBe(1)
  })

  it('has well-formed clean vectors including 0, 1 and 4294967295', () => {
    const ids = data.clean.map((v) => v.id)
    expect(new Set(ids).size).toBeGreaterThanOrEqual(5)
    for (const required of [0, 1, 4294967295]) expect(ids).toContain(required)
    for (const v of data.clean) {
      expect(v.bits).toMatch(/^[01]{64}$/)
      expect(v.idHex).toBe(hex(v.id, 8))
      expect(v.crcHex).toBe(hex(readField(Array.from(v.bits, Number), 32, 7), 2))
      expect(v.parityHex).toBe(hex(readField(Array.from(v.bits, Number), 39, 24), 6))
    }
  })

  it('has one 4-flip corrupted vector and one null vector', () => {
    for (const v of data.corrupted) {
      expect(v.bits).toMatch(/^[01]{64}$/)
      expect(v.originalIdHex).toBe(hex(v.originalId, 8))
    }
    const four = data.corrupted.filter((v) => v.expected !== null && v.expected.correctedBits === 4)
    expect(four).toHaveLength(1)
    expect(four[0].flipped).toHaveLength(4)
    expect(four[0].flipped.every((f) => f >= 0 && f <= 62)).toBe(true)
    expect(data.corrupted.filter((v) => v.expected === null)).toHaveLength(1)
  })
})

describe('Property 16: Format_Document vector conformance', () => {
  for (const v of data.clean) {
    it(`clean ${v.idHex}: encodes to the listed bits and decodes back`, () => {
      expect(bitString(encodeMessage(v.id))).toBe(v.bits)
      expect(decodeMessage(Uint8Array.from(v.bits, Number))).toEqual({ id: v.id, correctedBits: 0 })
      expect(v.expected).toEqual({ id: v.id, correctedBits: 0 })
    })
  }

  for (const v of data.corrupted) {
    it(`corrupted ${v.originalIdHex} flips [${v.flipped.join(', ')}]: decodes to the listed result`, () => {
      const m = encodeMessage(v.originalId)
      for (const f of v.flipped) m[f] ^= 1
      expect(bitString(m)).toBe(v.bits)
      expect(decodeMessage(Uint8Array.from(v.bits, Number))).toEqual(v.expected)
    })
  }
})
