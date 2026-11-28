// Requirements 10.9 to 10.12 with the real model; 3.3 to 3.5, 5.3 to 5.5 and 7.3 with a stand-in model,
// on the Fixture frames. The tests that need no Fixture are in pipeline.test.ts.
import { describe, expect, it, vi } from 'vitest'
import { encodeMessage } from '../../codec/message'
import { areaResize } from '../frame'
import { TAG_IDS } from '../layout'
import { createFrameReader, distanceHint } from '../pipeline'
import { detectTags } from '../tags'
import type { Point, RgbaFrame, RunReader } from '../types'
import { bitsOf, countDiff, loadContract, loadProducts, readPng } from './fixtures'
import { nodeRunReader } from './ortNode'

// Model runs and full-frame detection share the CPU with the other test files.
vi.setConfig({ testTimeout: 120_000 })
const contract = loadContract()
const products = loadProducts()

/** Paint a tag white: its corners pushed out by `grow` about its centre, filled. */
function coverTag(frame: RgbaFrame, corners: readonly Point[], grow = 1.4): RgbaFrame {
  const data = new Uint8ClampedArray(frame.data)
  const cx = corners.reduce((s, p) => s + p[0], 0) / 4
  const cy = corners.reduce((s, p) => s + p[1], 0) / 4
  const q = corners.map(([x, y]): Point => [cx + (x - cx) * grow, cy + (y - cy) * grow])
  const xs = q.map((p) => p[0])
  const ys = q.map((p) => p[1])
  const inside = (x: number, y: number) => {
    let sign = 0
    for (let k = 0; k < 4; k++) {
      const a = q[k]
      const b = q[(k + 1) % 4]
      const s = Math.sign((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]))
      if (s === 0) continue
      if (sign === 0) sign = s
      else if (s !== sign) return false
    }
    return true
  }
  for (let r = Math.max(0, Math.floor(Math.min(...ys))); r < Math.min(frame.height, Math.ceil(Math.max(...ys))); r++) {
    for (let c = Math.max(0, Math.floor(Math.min(...xs))); c < Math.min(frame.width, Math.ceil(Math.max(...xs))); c++) {
      if (inside(c + 0.5, r + 0.5)) data.set([255, 255, 255, 255], (r * frame.width + c) * 4)
    }
  }
  return { width: frame.width, height: frame.height, data }
}

function crop(frame: RgbaFrame, x0: number, y0: number, w: number, h: number): RgbaFrame {
  const data = new Uint8ClampedArray(w * h * 4)
  for (let r = 0; r < h; r++) data.set(frame.data.subarray(((y0 + r) * frame.width + x0) * 4, ((y0 + r) * frame.width + x0 + w) * 4), r * w * 4)
  return { width: w, height: h, data }
}

function white(width: number, height: number): RgbaFrame {
  return { width, height, data: new Uint8ClampedArray(width * height * 4).fill(255) }
}

/** Stand-in model: the logits of a clean Message for `id`, +-5. */
function standIn(id: number): RunReader & { calls: number } {
  const logits = Float32Array.from(encodeMessage(id), (b) => (b ? 5 : -5))
  const f = Object.assign(
    async () => {
      f.calls++
      return logits
    },
    { calls: 0 },
  )
  return f
}

describe('whole reader on each product, frames in order (10.9, 10.10)', () => {
  it.each(products.map((p) => [`${p.set}/${p.product}`, p] as const))('%s', async (_name, p) => {
    const read = createFrameReader({ contract, runReader: await nodeRunReader() })
    for (const [i, f] of p.frames.entries()) {
      const out = await read(readPng(f.frameFile))
      const d = out.detail!
      expect(out.missingTags).toBeUndefined()
      expect(d.framesAveraged).toBe(Math.min(i + 1, contract.framesAveraged))
      const frameDiff = countDiff(bitsOf(d.frameLogits!), f.bits)
      const windowDiff = countDiff(Array.from(d.bits!), f.bits)
      console.log(
        `${f.set}/${f.product}/${f.name}: own frame bits differ from the fixture's in ${frameDiff}, ` +
          `${d.framesAveraged}-frame average in ${windowDiff}; decoded ${out.result?.id}; ${d.pxPerTile!.toFixed(1)} px per tile; hint ${out.hint}`,
      )
      expect(out.result?.id).toBe(f.productId)
      if (f.set === 'synthetic') {
        expect(frameDiff).toBe(0)
        expect(windowDiff).toBe(0)
      }
      expect(out.hint).toBe(distanceHint(d.pxPerTile!, contract.trainedPxPerTile))
    }
  })
})

describe('missing tags (10.12, 3.3 to 3.5)', () => {
  const first = products[0].frames[0]

  it('10.12: one tag covered gives no read, the hint, and that tag as missing', async () => {
    const frame = readPng(first.frameFile)
    const tags = detectTags(frame, { ids: TAG_IDS })
    const covered = coverTag(frame, tags.get(2)!.corners)
    const read = createFrameReader({ contract, runReader: await nodeRunReader() })
    const out = await read(covered)
    expect(out.result).toBeNull()
    expect(out.hint).toBe('show all four corners')
    expect(out.missingTags).toEqual([2])
  })

  it('3.5: one to three tags found leaves the window unchanged; 3.4: none found empties it', async () => {
    const frame = readPng(first.frameFile)
    const tags = detectTags(frame, { ids: TAG_IDS })
    const model = standIn(first.productId)
    const read = createFrameReader({ contract, runReader: model })
    expect((await read(frame)).detail!.framesAveraged).toBe(1)
    expect((await read(frame)).detail!.framesAveraged).toBe(2)
    const partly = coverTag(coverTag(frame, tags.get(0)!.corners), tags.get(3)!.corners)
    const p = await read(partly)
    expect(p).toMatchObject({ result: null, hint: 'show all four corners', missingTags: [0, 3] })
    expect(p.detail!.framesAveraged).toBe(2)
    const next = await read(frame)
    expect(next.result?.id).toBe(first.productId)
    expect(next.detail!.framesAveraged).toBe(3)
    const none = await read(white(1080, 1920))
    expect(none).toMatchObject({ result: null, hint: 'show all four corners', missingTags: [0, 1, 2, 3] })
    expect(none.detail!.framesAveraged).toBe(0)
    expect((await read(frame)).detail!.framesAveraged).toBe(1)
    expect(model.calls).toBe(8) // two tiles per Read, four Reads
  })

  it('7.4: a new reader instance starts with an empty window', async () => {
    const frame = readPng(first.frameFile)
    const model = standIn(first.productId)
    const a = createFrameReader({ contract, runReader: model })
    await a(frame)
    await a(frame)
    const b = createFrameReader({ contract, runReader: model })
    expect((await b(frame)).detail!.framesAveraged).toBe(1)
  })
})

describe('distance hints (5.3 to 5.5)', () => {

  it('5.3: a frame shrunk to 0.9 (about 218 px per tile) says closer and still reads', async () => {
    const f = products[0].frames[0]
    const frame = readPng(f.frameFile)
    const small = areaResize(frame, Math.round(frame.width * 0.9), Math.round(frame.height * 0.9))
    const read = createFrameReader({ contract, runReader: standIn(f.productId) })
    const out = await read(small)
    expect(out.detail!.pxPerTile!).toBeLessThan(contract.trainedPxPerTile[0])
    expect(out.hint).toBe('closer')
    expect(out.result?.id).toBe(f.productId)
  })

  it('5.4: the centre of a 330 px frame enlarged by 1.1 (about 360 px per tile) says farther and still reads', async () => {
    const f = products[0].frames[2]
    const frame = readPng(f.frameFile)
    const w = Math.round(frame.width / 1.1)
    const h = Math.round(frame.height / 1.1)
    const big = areaResize(crop(frame, (frame.width - w) >> 1, (frame.height - h) >> 1, w, h), frame.width, frame.height)
    const read = createFrameReader({ contract, runReader: standIn(f.productId) })
    const out = await read(big)
    expect(out.detail!.pxPerTile!).toBeGreaterThan(contract.trainedPxPerTile[1])
    expect(out.hint).toBe('farther')
    expect(out.result?.id).toBe(f.productId)
  })

  it('5.5 and 2.2: a 1440 x 2560 frame is shrunk to 1080 x 1920 first and reads with no hint', async () => {
    const f = products[0].frames[1]
    const frame = readPng(f.frameFile)
    const big = areaResize(frame, 1440, 2560)
    const read = createFrameReader({ contract, runReader: standIn(f.productId) })
    const out = await read(big)
    expect(out.detail!.frameSize).toEqual([1080, 1920])
    expect(out.hint).toBeNull()
    expect(out.result?.id).toBe(f.productId)
  })
})

describe('model output check', () => {

  it('rejects a model that returns the wrong number of logits, naming MESSAGE_BITS', async () => {
    const f = products[0].frames[0]
    const read = createFrameReader({ contract, runReader: async () => new Float32Array(32) })
    await expect(read(readPng(f.frameFile))).rejects.toThrow(/MESSAGE_BITS/)
  })
})
