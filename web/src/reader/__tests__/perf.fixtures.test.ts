// Requirement 11: time per Processed_Frame on this machine, Node, onnxruntime-web wasm.
import os from 'node:os'
import * as ort from 'onnxruntime-web'
import { describe, expect, it } from 'vitest'
import { createFrameReader } from '../pipeline'
import { loadContract, loadProducts, readPng } from './fixtures'
import { nodeRunReader } from './ortNode'

const ROUNDS = 3

const median = (v: number[]) => {
  const s = [...v].sort((a, b) => a - b)
  return s.length % 2 ? s[(s.length - 1) >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2
}

describe('time per frame (11.1)', () => {
  it('tag detection, two flattens and two reader runs', async () => {
    const contract = loadContract()
    const run = await nodeRunReader()
    // One reader per product, so that the window never mixes products.
    const products = loadProducts()
      .filter((p) => p.set === 'synthetic')
      .map((p) => p.frames.map((f) => readPng(f.frameFile)))
    const frames = products.flat()
    // Warm-up: JIT and the wasm session.
    const warm = createFrameReader({ contract, runReader: run })
    for (const f of products[0]) await warm(f)

    const detect: number[] = []
    const flat: number[] = []
    const reader: number[] = []
    const total: number[] = []
    for (let r = 0; r < ROUNDS; r++) {
      for (const productFrames of products) {
        const read = createFrameReader({ contract, runReader: run })
        for (const f of productFrames) {
          const t0 = performance.now()
          const out = await read(f)
          total.push(performance.now() - t0)
          const t = out.detail!.timing
          detect.push(t.detectMs)
          flat.push(t.flattenMs)
          reader.push(t.readerMs)
          expect(out.result).not.toBeNull()
        }
      }
    }
    const fmt = (v: number[]) => `median ${median(v).toFixed(1)} ms, max ${Math.max(...v).toFixed(1)} ms`
    console.log(
      [
        `timing over ${frames.length} synthetic frames x ${ROUNDS} rounds, ${os.cpus()[0].model}, ${os.cpus().length} logical CPUs, Node ${process.version}, onnxruntime-web wasm numThreads ${ort.env.wasm.numThreads}`,
        `  tag detection:     ${fmt(detect)}`,
        `  two flattens:      ${fmt(flat)}`,
        `  two reader runs:   ${fmt(reader)}`,
        `  whole frame:       ${fmt(total)}`,
      ].join('\n'),
    )
  }, 120_000)
})
