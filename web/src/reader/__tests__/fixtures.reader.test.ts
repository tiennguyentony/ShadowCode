// Requirements 10.6 to 10.8: the reader on onnxruntime-web wasm in Node against the Fixtures.
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import { decodeMessage } from '../../codec/message'
import { flatten, imageToInput } from '../flatten'
import { meanOf } from '../logitWindow'
import { bitsOf, bundleDir, countDiff, largestAbs, loadContract, loadProducts, maxAbsDiff, readPng } from './fixtures'
import { nodeRunReader } from './ortNode'

// Model runs and full-frame detection share the CPU with the other test files.
vi.setConfig({ testTimeout: 120_000 })
const contract = loadContract()
const frames = loadProducts().flatMap((p) => p.frames)
const tiles = frames.flatMap((f) => f.tiles.map((t) => [`${f.set}/${f.product}/${t.patch}`, f, t] as const))

describe('reader on each Whole_Tile (10.6)', () => {
  it.each(tiles)('%s', async (_name, f, t) => {
    const run = await nodeRunReader()
    const L = largestAbs(t.logits)
    // The tile's logits come from the unrounded flatten with the reference corners (design D12).
    const fromFrame = await run(flatten(readPng(f.frameFile), t.corners, contract.flatten))
    const dFrame = maxAbsDiff(fromFrame, t.logits)
    expect(dFrame, `unrounded flatten: ${dFrame} > L/1000 = ${L / 1000}`).toBeLessThanOrEqual(L / 1000)
    expect(bitsOf(fromFrame)).toEqual(t.bits)
    // The 8-bit patch file: rounding moves the logits by up to about 1% of L.
    const fromPatch = await run(imageToInput(readPng(join(f.dir, t.patch))))
    const dPatch = maxAbsDiff(fromPatch, t.logits)
    console.log(`${f.set}/${f.product}/${t.patch}: unrounded flatten ${(dFrame / L).toExponential(2)} L, patch file ${(dPatch / L).toExponential(2)} L`)
    expect(dPatch, `patch file: ${dPatch} > L/50 = ${L / 50}`).toBeLessThanOrEqual(L / 50)
    expect(bitsOf(fromPatch)).toEqual(t.bits)
  })
})

describe('reader on the bundle fixtures (10.7)', () => {
  const dir = join(bundleDir(), 'fixtures')
  const names = readdirSync(dir).filter((n) => /^\d+\.json$/.test(n)).map((n) => n.slice(0, -5)).sort()
  it('has bundle fixtures', () => expect(names.length).toBeGreaterThan(0))
  it.each(names)('fixtures/%s_patch.png', async (name) => {
    const j = JSON.parse(readFileSync(join(dir, `${name}.json`), 'utf8'))
    const run = await nodeRunReader()
    const logits = await run(imageToInput(readPng(join(dir, `${name}_patch.png`))))
    const L = largestAbs(j.logits)
    expect(maxAbsDiff(logits, j.logits)).toBeLessThanOrEqual(L / 1000)
    expect(bitsOf(logits)).toEqual(j.bits)
  })
})

describe('frame logits and product (10.8)', () => {
  it.each(frames.map((f) => [`${f.set}/${f.product}/${f.name}`, f] as const))('%s', (_name, f) => {
    const mean = meanOf(f.tiles.map((t) => t.logits))
    const L = largestAbs(f.logits)
    expect(maxAbsDiff(mean, f.logits)).toBeLessThanOrEqual(L / 1000)
    expect(bitsOf(mean)).toEqual(f.bits)
    const result = decodeMessage(Uint8Array.from(f.bits))
    expect(result?.id).toBe(f.productId)
    expect(String(f.productId)).toBe(f.product)
    // 10.11: a single tile may have a wrong bit; reported, not required.
    console.log(`${f.set}/${f.product}/${f.name}: tile bits wrong against expected_bits ${f.tiles.map((t) => countDiff(t.bits, f.expectedBits)).join(', ')}`)
  })
})
