// Requirements 10.4 and 10.5: Flatten_Corners and patch values against every Fixture.
import { join } from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import { flatten } from '../flatten'
import { shrinkToShortSide } from '../frame'
import { LABEL_LAYOUT, TAG_IDS } from '../layout'
import { locateTiles } from '../locate'
import { detectTags } from '../tags'
import { loadContract, loadProducts, readPng } from './fixtures'

// Model runs and full-frame detection share the CPU with the other test files.
vi.setConfig({ testTimeout: 120_000 })
const contract = loadContract()
const frames = loadProducts().flatMap((p) => p.frames)

describe('Flatten_Corners within 1 px of the reference (10.4)', () => {
  it.each(frames.map((f) => [`${f.set}/${f.product}/${f.name}`, f] as const))('%s', (_name, f) => {
    const frame = shrinkToShortSide(readPng(f.frameFile), contract.streamShortSide)
    expect([frame.width, frame.height]).toEqual([...f.frameSize])
    const tags = detectTags(frame, { ids: TAG_IDS })
    const located = locateTiles(new Map([...tags].map(([id, t]) => [id, t.centre])), LABEL_LAYOUT)
    expect(located.ok, `missing tags ${!located.ok && located.missingTags}`).toBe(true)
    if (!located.ok) return
    expect(located.tileCorners).toHaveLength(f.tiles.length)
    let worst = 0
    f.tiles.forEach((tile, i) => {
      expect([tile.row, tile.col]).toEqual([LABEL_LAYOUT.wholeTiles[i].row, LABEL_LAYOUT.wholeTiles[i].col])
      tile.corners.forEach((ref, k) => {
        const [x, y] = located.tileCorners[i][k]
        worst = Math.max(worst, Math.abs(x - ref[0]), Math.abs(y - ref[1]))
        expect(Math.abs(x - ref[0]), `tile r${tile.row} corner ${k} x`).toBeLessThanOrEqual(1)
        expect(Math.abs(y - ref[1]), `tile r${tile.row} corner ${k} y`).toBeLessThanOrEqual(1)
      })
    })
    console.log(`${f.set}/${f.product}/${f.name}: largest corner difference ${worst.toFixed(3)} px, ${located.pxPerTile.toFixed(1)} px per tile`)
  })
})

describe('flattened patch within 1 level of the patch file (10.5)', () => {
  const tiles = frames.flatMap((f) => f.tiles.map((t) => [`${f.set}/${f.product}/${t.patch}`, f, t] as const))
  it.each(tiles)('%s', (_name, f, t) => {
    const frame = readPng(f.frameFile)
    const patch = readPng(join(f.dir, t.patch))
    const input = flatten(frame, t.corners, contract.flatten, true)
    const plane = contract.tilePx * contract.tilePx
    let worst = 0
    let off = 0
    for (let c = 0; c < 3; c++) {
      for (let p = 0; p < plane; p++) {
        const d = Math.abs(Math.round(input[c * plane + p] * 255) - patch.data[p * 4 + c])
        worst = Math.max(worst, d)
        if (d > 0) off++
      }
    }
    console.log(`${f.set}/${f.product}/${t.patch}: ${off} of ${3 * plane} values off, at most ${worst} level`)
    expect(worst).toBeLessThanOrEqual(1)
  })
})
