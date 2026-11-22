// Seeded measurements for Requirements 7 and 8 (Properties 14 and 15).
import { describe, expect, it } from 'vitest'
import { decodeMessage } from '../message'
import {
  K_VALUES,
  MEASUREMENT_SEED,
  RANDOM_TRIALS,
  TRIALS_PER_K,
  forEachHeavyTrial,
  forEachRandomMessage,
  addCounts,
  bitString,
  emptyCounts,
  formatFalseAccept,
  hex,
  rate,
  type Counts,
} from './message.helpers'

function runFalseAccept(m: number) {
  const perK = new Map<number, Counts>(K_VALUES.map((k) => [k, emptyCounts()]))
  forEachHeavyTrial(({ k, id, flips, bits }) => {
    let r
    try {
      r = decodeMessage(bits, { maxCorrectableErrors: m })
    } catch (e) {
      console.error(`[message-codec] throw m=${m} id=${hex(id, 8)} flips=${flips.join(',')}`)
      throw e
    }
    const c = perK.get(k)!
    c.n++
    if (r === null) c.null++
    else if (r.id === id) c.correct++
    else c.falseAccept++
  })
  const k8 = [8, 12, 16, 32].map((k) => perK.get(k)!).reduce(addCounts, emptyCounts())
  const total = [...perK.values()].reduce(addCounts, emptyCounts())
  console.log(formatFalseAccept(`falseAccept m=${m}`, MEASUREMENT_SEED, perK, k8, total))
  for (const c of perK.values()) {
    expect(c.n).toBe(TRIALS_PER_K)
    expect(c.correct + c.null + c.falseAccept).toBe(c.n)
  }
  expect(total.n).toBe(TRIALS_PER_K * K_VALUES.length)
  return { k8, total }
}

function runRandomInput(m: number) {
  let nulls = 0
  let nonNull = 0
  forEachRandomMessage((bits) => {
    let r
    try {
      r = decodeMessage(bits, { maxCorrectableErrors: m })
    } catch (e) {
      console.error(`[message-codec] throw m=${m} input=${bitString(bits)}`)
      throw e
    }
    if (r === null) nulls++
    else nonNull++
  })
  const rejectionRate = nulls / RANDOM_TRIALS
  console.log(
    `[message-codec] randomInput m=${m} seed=${hex(MEASUREMENT_SEED, 8)} trials=${RANDOM_TRIALS} null=${nulls} nonNull=${nonNull} rejectionRate=${(rejectionRate * 100).toFixed(3)}%`,
  )
  expect(nulls + nonNull).toBe(RANDOM_TRIALS)
  return rejectionRate
}

describe('seeded measurements', () => {
  // Feature: message-codec, Property 14: Bounded false accepts under heavy corruption (measured)
  it('false-accept m=4', () => {
    const { k8 } = runFalseAccept(4)
    expect(rate(k8)).toBeLessThanOrEqual(1e-3)
  }, 60_000)

  it('false-accept m=3', () => {
    runFalseAccept(3)
  }, 60_000)

  // Feature: message-codec, Property 15: Random input is rejected (measured)
  it('random-input m=4', () => {
    expect(runRandomInput(4)).toBeGreaterThanOrEqual(0.999)
  }, 60_000)

  it('random-input m=3', () => {
    runRandomInput(3)
  }, 60_000)
})
