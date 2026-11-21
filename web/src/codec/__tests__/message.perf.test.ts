// Requirements 13.5-13.7: mean decode time under 1 ms.
import { describe, expect, it } from 'vitest'
import { decodeMessage, encodeMessage } from '../message'
import { mulberry32 } from './message.helpers'

const WARMUP = 1_000
const TIMED = 10_000

function build(kind: 'flips0to4' | 'random', count: number, seed: number): Uint8Array[] {
  const next = mulberry32(seed)
  const out: Uint8Array[] = []
  for (let t = 0; t < count; t++) {
    if (kind === 'flips0to4') {
      const m = encodeMessage(next())
      const k = next() % 5
      const chosen = new Set<number>()
      while (chosen.size < k) chosen.add(next() % 63)
      for (const i of chosen) m[i] ^= 1
      out.push(m)
    } else {
      const m = new Uint8Array(64)
      for (let i = 0; i < 64; i++) m[i] = next() & 1
      out.push(m)
    }
  }
  return out
}

describe('decode performance', () => {
  for (const kind of ['flips0to4', 'random'] as const) {
    it(`timing kind=${kind}`, () => {
      const warm = build(kind, WARMUP, 1)
      const timed = build(kind, TIMED, 2)
      for (const m of warm) decodeMessage(m)
      let sink = 0
      const t0 = performance.now()
      for (const m of timed) if (decodeMessage(m) !== null) sink++
      const meanMs = (performance.now() - t0) / TIMED
      console.log(`[message-codec] timing kind=${kind} calls=${TIMED} meanMs=${meanMs.toFixed(5)} (accepted ${sink})`)
      expect(meanMs).toBeLessThan(1)
    })
  }
})
