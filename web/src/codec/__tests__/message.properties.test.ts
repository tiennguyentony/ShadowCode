import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { decodeMessage, encodeMessage, type DecodeOptions, type DecodeResult } from '../message'
import { FC_SEED, hex, readField, refCrc7, refMessage, refModG, withFlips } from './message.helpers'

const RUNS = 1_000
const RUNS_HEAVY = 10_000

const params = (numRuns: number, examples?: unknown[][]) =>
  ({ seed: FC_SEED, numRuns, ...(examples ? { examples } : {}) }) as fc.Parameters<never>

const idArb = fc.noBias(fc.integer({ min: 0, max: 0xffffffff }))
const limitArb = fc.noBias(fc.integer({ min: 0, max: 4 }))
const flipsArb = (k: number, max = 62) =>
  fc.noBias(fc.uniqueArray(fc.noBias(fc.integer({ min: 0, max })), { minLength: k, maxLength: k }))
const randomMessageArb = fc
  .array(fc.noBias(fc.integer({ min: 0, max: 1 })), { minLength: 64, maxLength: 64 })
  .map((a) => Uint8Array.from(a))
/** Encoded Messages with 0..maxFlips random flips anywhere in 0..63. */
const encodedFlippedArb = (maxFlips: number) =>
  fc
    .tuple(idArb, fc.noBias(fc.integer({ min: 0, max: maxFlips })).chain((k) => flipsArb(k, 63)))
    .map(([id, flips]) => withFlips(encodeMessage(id), flips))
const anyMessageArb = (maxFlips: number) => fc.oneof(randomMessageArb, encodedFlippedArb(maxFlips))
const optionsArb: fc.Arbitrary<DecodeOptions | undefined> = fc.oneof(
  fc.constant(undefined),
  limitArb.map((m) => ({ maxCorrectableErrors: m })),
)
const limitOf = (o: DecodeOptions | undefined) => o?.maxCorrectableErrors ?? 4

const same = (a: DecodeResult, b: DecodeResult) =>
  a === null || b === null ? a === b : a.id === b.id && a.correctedBits === b.correctedBits

const ID_EXAMPLES = [[0], [1], [0xffffffff]]

const logSeed = (n: number) => console.log(`[message-codec] property ${n} fast-check seed=${hex(FC_SEED, 8)}`)

describe('message codec properties', () => {
  // Feature: message-codec, Property 1: Encoder output structure and layout
  it('Property 1: encoder output structure and layout', () => {
    logSeed(1)
    fc.assert(
      fc.property(idArb, (id) => {
        const m = encodeMessage(id)
        expect(m).toBeInstanceOf(Uint8Array)
        expect(m.length).toBe(64)
        expect(m.every((b) => b === 0 || b === 1)).toBe(true)
        expect(readField(m, 0, 32)).toBe(id)
        expect(readField(m, 32, 7)).toBe(refCrc7(id))
        expect(refModG(m.subarray(0, 63)).every((b) => b === 0)).toBe(true)
        expect(m[63]).toBe(0)
      }),
      params(RUNS, ID_EXAMPLES),
    )
  })

  // Feature: message-codec, Property 2: Encoding is deterministic and isolated from caller mutation
  it('Property 2: encoding is deterministic and isolated from caller mutation', () => {
    logSeed(2)
    const writes = fc.array(fc.tuple(fc.integer({ min: 0, max: 63 }), fc.integer({ min: 0, max: 255 })))
    fc.assert(
      fc.property(idArb, writes, (id, ws) => {
        const first = encodeMessage(id)
        const copy = first.slice()
        for (const [i, v] of ws) first[i] = v
        const second = encodeMessage(id)
        expect(second).not.toBe(first)
        expect(Array.from(second)).toEqual(Array.from(copy))
      }),
      params(RUNS),
    )
  })

  // Feature: message-codec, Property 3: Round trip
  it('Property 3: round trip', () => {
    logSeed(3)
    fc.assert(
      fc.property(idArb, optionsArb, (id, opts) => {
        const r = opts === undefined ? decodeMessage(encodeMessage(id)) : decodeMessage(encodeMessage(id), opts)
        if (r === null || r.id !== id || r.correctedBits !== 0) {
          throw new Error(`id=${id} options=${JSON.stringify(opts)} result=${JSON.stringify(r)}`)
        }
        expect(Number.isInteger(r.id) && r.id >= 0 && r.id <= 0xffffffff).toBe(true)
      }),
      params(RUNS_HEAVY, [[0, undefined], [1, undefined], [0xffffffff, undefined]]),
    )
  })

  // Feature: message-codec, Property 4: Errors within the limit are corrected and counted
  it('Property 4a: errors within the option limit are corrected and counted', () => {
    logSeed(4)
    const spec = limitArb.chain((m) =>
      fc.noBias(fc.integer({ min: 0, max: m })).chain((k) => flipsArb(k).map((flips) => ({ m, flips }))),
    )
    fc.assert(
      fc.property(idArb, spec, fc.boolean(), (id, { m, flips }, spare) => {
        const bits = withFlips(encodeMessage(id), flips)
        bits[63] = spare ? 1 : 0
        const r = decodeMessage(bits, { maxCorrectableErrors: m })
        expect(r, `id=${id} m=${m} flips=${flips} spare=${spare}`).toEqual({ id, correctedBits: flips.length })
      }),
      params(RUNS_HEAVY, [
        [0, { m: 4, flips: [0, 1, 2, 3] }, true],
        [1, { m: 4, flips: [59, 60, 61, 62] }, false],
        [0xffffffff, { m: 0, flips: [] }, true],
      ]),
    )
  })

  // Feature: message-codec, Property 4: Errors within the limit are corrected and counted
  it('Property 4b: up to 4 errors are corrected with options omitted', () => {
    logSeed(4)
    const flips = fc.noBias(fc.integer({ min: 0, max: 4 })).chain((k) => flipsArb(k))
    fc.assert(
      fc.property(idArb, flips, fc.boolean(), (id, f, spare) => {
        const bits = withFlips(encodeMessage(id), f)
        bits[63] = spare ? 1 : 0
        expect(decodeMessage(bits), `id=${id} flips=${f}`).toEqual({ id, correctedBits: f.length })
      }),
      params(RUNS_HEAVY, [[0, [0, 31, 32, 62], false], [1, [], true], [0xffffffff, [5, 38, 39], false]]),
    )
  })

  // Feature: message-codec, Property 5: Errors beyond the option limit are rejected
  it('Property 5: errors beyond the option limit are rejected', () => {
    logSeed(5)
    const spec = fc
      .noBias(fc.integer({ min: 0, max: 3 }))
      .chain((m) =>
        fc.noBias(fc.integer({ min: m + 1, max: 4 })).chain((k) => flipsArb(k).map((flips) => ({ m, flips }))),
      )
    fc.assert(
      fc.property(idArb, spec, fc.boolean(), (id, { m, flips }, spare) => {
        const bits = withFlips(encodeMessage(id), flips)
        bits[63] = spare ? 1 : 0
        expect(decodeMessage(bits, { maxCorrectableErrors: m }), `id=${id} m=${m} flips=${flips}`).toBeNull()
      }),
      params(RUNS_HEAVY, [[0, { m: 3, flips: [0, 1, 2, 3] }, false], [0xffffffff, { m: 0, flips: [62] }, true]]),
    )
  })

  // Feature: message-codec, Property 6: Monotonicity in the correction limit
  it('Property 6: monotonicity in the correction limit', () => {
    logSeed(6)
    fc.assert(
      fc.property(anyMessageArb(6), (bits) => {
        const results = [0, 1, 2, 3, 4].map((m) => decodeMessage(bits, { maxCorrectableErrors: m }))
        for (let m = 0; m < 4; m++) {
          for (let m2 = m + 1; m2 <= 4; m2++) {
            if (results[m] !== null) expect(same(results[m], results[m2])).toBe(true)
            if (results[m2] === null) expect(results[m]).toBeNull()
          }
        }
      }),
      params(RUNS),
    )
  })

  // Feature: message-codec, Property 7: Option forms that mean the same value give the same result
  it('Property 7: option forms that mean the same value give the same result', () => {
    logSeed(7)
    const extra = fc.dictionary(fc.string().filter((k) => k !== 'maxCorrectableErrors' && k !== '__proto__'), fc.anything())
    fc.assert(
      fc.property(anyMessageArb(6), limitArb, extra, (bits, m, junk) => {
        const ref4 = decodeMessage(bits, { maxCorrectableErrors: 4 })
        for (const r of [
          decodeMessage(bits),
          decodeMessage(bits, undefined),
          decodeMessage(bits, {}),
          decodeMessage(bits, { maxCorrectableErrors: undefined }),
        ]) {
          expect(same(r, ref4)).toBe(true)
        }
        expect(same(decodeMessage(bits, { maxCorrectableErrors: -0 }), decodeMessage(bits, { maxCorrectableErrors: 0 }))).toBe(true)
        const plain = decodeMessage(bits, { maxCorrectableErrors: m })
        expect(same(decodeMessage(bits, { ...junk, maxCorrectableErrors: m }), plain)).toBe(true)
      }),
      params(RUNS),
    )
  })

  // Feature: message-codec, Property 8: Spare-bit independence
  it('Property 8: spare-bit independence', () => {
    logSeed(8)
    fc.assert(
      fc.property(anyMessageArb(6), optionsArb, (bits, opts) => {
        const r1 = decodeMessage(bits, opts)
        const r2 = decodeMessage(withFlips(bits, [63]), opts)
        expect(same(r1, r2)).toBe(true)
      }),
      params(RUNS),
    )
  })

  // Feature: message-codec, Property 9: Decoding valid input is total
  it('Property 9: decoding valid input is total', () => {
    logSeed(9)
    fc.assert(
      fc.property(anyMessageArb(32), optionsArb, (bits, opts) => {
        const r = decodeMessage(bits, opts)
        if (r !== null) {
          expect(Number.isInteger(r.id) && r.id >= 0 && r.id <= 0xffffffff).toBe(true)
          expect(Number.isInteger(r.correctedBits) && r.correctedBits >= 0 && r.correctedBits <= limitOf(opts)).toBe(true)
        }
      }),
      params(RUNS),
    )
  })

  // Feature: message-codec, Property 10: Inputs are not modified
  it('Property 10: inputs are not modified', () => {
    logSeed(10)
    const bitsArb = fc.oneof(
      anyMessageArb(8),
      fc
        .tuple(anyMessageArb(4), fc.array(fc.tuple(fc.integer({ min: 0, max: 63 }), fc.integer({ min: 2, max: 255 })), { minLength: 1 }))
        .map(([b, ws]) => {
          const out = b.slice()
          for (const [i, v] of ws) out[i] = v
          return out
        }),
    )
    const optsArb = fc.oneof(
      fc.constant(undefined),
      fc.record({ maxCorrectableErrors: fc.oneof(limitArb, fc.double(), fc.integer({ min: -10, max: 10 })) }),
    )
    fc.assert(
      fc.property(bitsArb, optsArb, (bits, opts) => {
        const bitsCopy = bits.slice()
        const optsCopy = opts === undefined ? undefined : { ...opts }
        try {
          decodeMessage(bits, opts)
        } catch (e) {
          expect(e).toBeInstanceOf(RangeError)
        }
        expect(bits.length).toBe(64)
        expect(Array.from(bits)).toEqual(Array.from(bitsCopy))
        expect(opts).toEqual(optsCopy)
      }),
      params(RUNS),
    )
  })

  // Feature: message-codec, Property 11: Statelessness
  it('Property 11: statelessness', () => {
    logSeed(11)
    fc.assert(
      fc.property(
        anyMessageArb(6),
        optionsArb,
        fc.array(fc.tuple(anyMessageArb(6), optionsArb), { minLength: 0, maxLength: 20 }),
        (x, o, others) => {
          const before = decodeMessage(x, o)
          for (const [b, oo] of others) decodeMessage(b, oo)
          expect(same(before, decodeMessage(x, o))).toBe(true)
        },
      ),
      params(RUNS),
    )
  })

  // Feature: message-codec, Property 12: CRC mismatch is rejected
  it('Property 12: CRC mismatch is rejected', () => {
    logSeed(12)
    fc.assert(
      fc.property(idArb, fc.integer({ min: 1, max: 127 }), (id, delta) => {
        const wrong = refCrc7(id) ^ delta
        const bits = refMessage(id, wrong)
        expect(refModG(bits.subarray(0, 63)).every((b) => b === 0)).toBe(true)
        for (let m = 0; m <= 4; m++) expect(decodeMessage(bits, { maxCorrectableErrors: m })).toBeNull()
      }),
      params(RUNS),
    )
  })

  // Feature: message-codec, Property 13: Invalid arguments throw RangeError
  it('Property 13: invalid arguments throw RangeError', () => {
    logSeed(13)
    const badId = fc.oneof(
      fc.double().filter((v) => !Number.isInteger(v) || v < 0 || v > 0xffffffff),
      fc.integer({ min: -(2 ** 31), max: -1 }),
      fc.integer({ min: 2 ** 32, max: Number.MAX_SAFE_INTEGER }),
      fc.string(),
      fc.bigInt(),
      fc.constant(null),
      fc.constant(undefined),
      fc.boolean(),
    )
    const badBits = fc.oneof(
      fc
        .tuple(anyMessageArb(4), fc.integer({ min: 0, max: 63 }), fc.integer({ min: 2, max: 255 }))
        .map(([b, i, v]) => {
          const out = b.slice()
          out[i] = v
          return out
        }),
      fc.integer({ min: 0, max: 200 }).filter((n) => n !== 64).map((n) => new Uint8Array(n)),
    )
    const badLimit = fc.oneof(
      fc.double().filter((v) => !Number.isInteger(v) || v < 0 || v > 4),
      fc.integer().filter((v) => v < 0 || v > 4),
      fc.string(),
      fc.bigInt(),
      fc.constant(null),
      fc.boolean(),
    )
    const anyOpts = fc.oneof(optionsArb, badLimit.map((v) => ({ maxCorrectableErrors: v as number })))
    fc.assert(
      fc.property(badId, (v) => {
        expect(() => encodeMessage(v as number)).toThrow(RangeError)
      }),
      params(RUNS),
    )
    fc.assert(
      fc.property(badBits, anyOpts, (b, o) => {
        expect(() => decodeMessage(b, o)).toThrow(RangeError)
      }),
      params(RUNS),
    )
    fc.assert(
      fc.property(anyMessageArb(4), badLimit, (b, v) => {
        expect(() => decodeMessage(b, { maxCorrectableErrors: v as number })).toThrow(RangeError)
      }),
      params(RUNS),
    )
  })
})
