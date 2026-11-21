// Requirement 15: spare-bit analysis and Genuine_Read measurements. Reported only, no bounds.
// Spare_Bit_Option_B (ignore index 63) is what the codec implements. Option A
// (reject if index 63 is 1) is derived here from each trial's single decode.
import { describe, expect, it } from 'vitest'
import { decodeMessage, encodeMessage, type DecodeResult } from '../message'
import {
  GENUINE_SEED,
  K_VALUES,
  MEASUREMENT_SEED,
  RANDOM_TRIALS,
  addCounts,
  bitString,
  emptyCounts,
  fmtRate,
  forEachHeavyTrial,
  forEachRandomMessage,
  hex,
  mulberry32,
  rate,
  type Counts,
} from './message.helpers'

const optionA = (r: DecodeResult, spare: number): DecodeResult => (spare === 1 ? null : r)

function classify(c: Counts, r: DecodeResult, id: number) {
  c.n++
  if (r === null) c.null++
  else if (r.id === id) c.correct++
  else c.falseAccept++
}

const row = (c: Counts) => `${c.correct} / ${c.null} / ${c.falseAccept}`

describe('spare-bit analysis', () => {
  it('spare-bit analysis m=4', () => {
    const perK = { A: new Map<number, Counts>(), B: new Map<number, Counts>() }
    for (const k of K_VALUES) {
      perK.A.set(k, emptyCounts())
      perK.B.set(k, emptyCounts())
    }
    forEachHeavyTrial(({ k, id, flips, bits }) => {
      let r: DecodeResult
      try {
        r = decodeMessage(bits, { maxCorrectableErrors: 4 })
      } catch (e) {
        console.error(`[message-codec] throw spare m=4 id=${hex(id, 8)} flips=${flips.join(',')}`)
        throw e
      }
      classify(perK.B.get(k)!, r, id)
      classify(perK.A.get(k)!, optionA(r, bits[63]), id)
    })

    const random = { A: { nul: 0, non: 0 }, B: { nul: 0, non: 0 } }
    forEachRandomMessage((bits) => {
      let r: DecodeResult
      try {
        r = decodeMessage(bits, { maxCorrectableErrors: 4 })
      } catch (e) {
        console.error(`[message-codec] throw spare m=4 input=${bitString(bits)}`)
        throw e
      }
      if (r === null) random.B.nul++
      else random.B.non++
      if (optionA(r, bits[63]) === null) random.A.nul++
      else random.A.non++
    })

    const lines = [`[message-codec] spareBit m=4 seed=${hex(MEASUREMENT_SEED, 8)}`]
    lines.push('| k | trials | Option A: correct / null / FA | Option A FA rate | Option B: correct / null / FA | Option B FA rate |')
    lines.push('|---|---|---|---|---|---|')
    const sum = (map: Map<number, Counts>, ks: readonly number[]) =>
      ks.map((k) => map.get(k)!).reduce(addCounts, emptyCounts())
    const rows: [string, Counts, Counts][] = [
      ...K_VALUES.map((k) => [String(k), perK.A.get(k)!, perK.B.get(k)!] as [string, Counts, Counts]),
      ['k ≥ 8', sum(perK.A, [8, 12, 16, 32]), sum(perK.B, [8, 12, 16, 32])],
      ['all', sum(perK.A, K_VALUES), sum(perK.B, K_VALUES)],
    ]
    for (const [k, a, b] of rows) {
      lines.push(`| ${k} | ${a.n} | ${row(a)} | ${fmtRate(rate(a))} | ${row(b)} | ${fmtRate(rate(b))} |`)
      expect(a.correct + a.null + a.falseAccept).toBe(a.n)
      expect(b.correct + b.null + b.falseAccept).toBe(b.n)
    }
    lines.push('| Option | trials | null | non-null | Rejection_Rate |', '|---|---|---|---|---|')
    for (const opt of ['A', 'B'] as const) {
      const { nul, non } = random[opt]
      expect(nul + non).toBe(RANDOM_TRIALS)
      lines.push(`| ${opt} | ${RANDOM_TRIALS} | ${nul} | ${non} | ${((nul / RANDOM_TRIALS) * 100).toFixed(3)}% |`)
    }
    console.log(lines.join('\n'))
  }, 60_000)
})

describe('genuine reads', () => {
  const TRIALS = 100_000
  for (const p of [0.01, 0.03, 0.05]) {
    it(`genuine-read p=${p}`, () => {
      const next = mulberry32(GENUINE_SEED)
      const thr = Math.floor(p * 2 ** 32)
      const A = emptyCounts()
      const B = emptyCounts()
      let spareRejected = 0
      for (let t = 0; t < TRIALS; t++) {
        const id = next()
        const bits = encodeMessage(id)
        const flips: number[] = []
        for (let i = 0; i < 64; i++) {
          if (next() < thr) {
            bits[i] ^= 1
            flips.push(i)
          }
        }
        let r: DecodeResult
        try {
          r = decodeMessage(bits, { maxCorrectableErrors: 4 })
        } catch (e) {
          console.error(`[message-codec] throw genuine p=${p} id=${hex(id, 8)} flips=${flips.join(',')}`)
          throw e
        }
        classify(B, r, id)
        classify(A, optionA(r, bits[63]), id)
        if (r !== null && r.id === id && bits[63] === 1) spareRejected++
      }
      expect(A.n).toBe(TRIALS)
      expect(B.n).toBe(TRIALS)
      expect(A.correct + A.null + A.falseAccept).toBe(TRIALS)
      expect(B.correct + B.null + B.falseAccept).toBe(TRIALS)
      const share = B.correct === 0 ? 0 : spareRejected / B.correct
      console.log(
        `[message-codec] genuineRead p=${p} seed=${hex(GENUINE_SEED, 8)} trials=${TRIALS} ` +
          `optionA=${row(A)} fractionCorrectA=${(A.correct / TRIALS).toFixed(5)} ` +
          `optionB=${row(B)} fractionCorrectB=${(B.correct / TRIALS).toFixed(5)} ` +
          `spareRejectionShare=${spareRejected}/${B.correct}=${(share * 100).toFixed(2)}%`,
      )
    }, 60_000)
  }
})
