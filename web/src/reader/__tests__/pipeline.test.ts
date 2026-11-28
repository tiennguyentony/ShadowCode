// Frame pipeline checks that need no Fixture (Requirements 2.1, 3.3, 3.4, 5.3 to 5.5).
// The contract is the committed web/public/models/20261126-ba0ab4d/contract.json.
import { describe, expect, it } from 'vitest'
import { encodeMessage } from '../../codec/message'
import { createFrameReader, distanceHint } from '../pipeline'
import type { RgbaFrame } from '../types'
import { loadContract } from './fixtures'

const contract = loadContract()

function standIn(id: number) {
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

function white(width: number, height: number): RgbaFrame {
  return { width, height, data: new Uint8ClampedArray(width * height * 4).fill(255) }
}

describe('distance hints (5.3 to 5.5)', () => {
  it('limits come from camera.trained_px_per_tile', () => {
    const [low, high] = contract.trainedPxPerTile
    expect(distanceHint(low - 0.01, contract.trainedPxPerTile)).toBe('closer')
    expect(distanceHint(low, contract.trainedPxPerTile)).toBeNull()
    expect(distanceHint(high, contract.trainedPxPerTile)).toBeNull()
    expect(distanceHint(high + 0.01, contract.trainedPxPerTile)).toBe('farther')
  })
})

describe('frames without a read', () => {
  it('2.1: rejects a wrong data length', async () => {
    const read = createFrameReader({ contract, runReader: standIn(1001) })
    await expect(read({ width: 4, height: 4, data: new Uint8ClampedArray(63) })).rejects.toThrow(/expected 64/)
  })

  it('3.3, 3.4: a frame with no tag gives null, the hint, all four IDs, an empty window and no model run', async () => {
    const model = standIn(1001)
    const read = createFrameReader({ contract, runReader: model })
    const out = await read(white(1080, 1920))
    expect(out).toMatchObject({ result: null, hint: 'show all four corners', missingTags: [0, 1, 2, 3] })
    expect(out.detail!.framesAveraged).toBe(0)
    expect(model.calls).toBe(0)
  })

  it('2.2: a 1440 x 2560 frame is read at 1080 x 1920', async () => {
    const read = createFrameReader({ contract, runReader: standIn(1001) })
    const out = await read(white(1440, 2560))
    expect(out.detail!.frameSize).toEqual([1080, 1920])
  })
})
