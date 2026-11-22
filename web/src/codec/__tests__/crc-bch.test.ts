import { describe, expect, it } from 'vitest'
import { crc7 } from '../crc'
import { bchCorrect, bchParity } from '../bch'
import { refModG } from './message.helpers'

describe('crc7', () => {
  it('matches the worked values in the Format_Document', () => {
    expect(crc7(0)).toBe(0x7f)
    expect(crc7(1)).toBe(0x76)
    expect(crc7(2)).toBe(0x6d)
    expect(crc7(0xffffffff)).toBe(0x6a)
  })

  it('core (without final XOR) over ASCII "123456789" equals the CRC-7/MMC check 0x75', () => {
    let reg = 0
    for (const ch of '123456789') {
      const byte = ch.charCodeAt(0)
      for (let b = 7; b >= 0; b--) {
        const fb = ((reg >>> 6) & 1) ^ ((byte >>> b) & 1)
        reg = (reg << 1) & 0x7f
        if (fb) reg ^= 0x09
      }
    }
    expect(reg).toBe(0x75)
  })
})

describe('bch', () => {
  const sample = (): Uint8Array => {
    const m = new Uint8Array(63)
    for (let i = 0; i < 39; i++) m[i] = (i * 7 + 3) % 5 < 2 ? 1 : 0
    bchParity(m)
    return m
  }

  it('bchParity output is divisible by g(x)', () => {
    expect(refModG(sample()).every((b) => b === 0)).toBe(true)
  })

  it('bchCorrect returns 0 on a codeword and k for k <= 4 flips', () => {
    const c = sample()
    expect(bchCorrect(c.slice())).toBe(0)
    const flips = [0, 20, 38, 62]
    for (let k = 1; k <= 4; k++) {
      const r = c.slice()
      for (const f of flips.slice(0, k)) r[f] ^= 1
      expect(bchCorrect(r)).toBe(k)
      expect(Array.from(r)).toEqual(Array.from(c))
    }
  })
})
