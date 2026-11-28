// Requirement 1: the committed bundle files (contract.json, reader.onnx) and the Contract.
// Runs on a clean checkout; the comparison with the source bundle is in bundle.fixtures.test.ts.
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { parseContract } from '../contract'
import { BUNDLE_ID, bundleDir, loadContractJson } from './fixtures'

const sha256 = (buf: Buffer) => createHash('sha256').update(buf).digest('hex')

describe('model bundle 20261126-ba0ab4d (Requirement 1)', () => {
  it('1.4: reader.onnx has the bytes and sha256 of models.reader', () => {
    const json = loadContractJson() as { models: { reader: { file: string; bytes: number; sha256: string } } }
    const onnx = readFileSync(join(bundleDir(), json.models.reader.file))
    expect(onnx.length).toBe(json.models.reader.bytes)
    expect(sha256(onnx)).toBe(json.models.reader.sha256)
  })

  it('1.4: the contract says trained, flattened, TILE_PX 384, MESSAGE_BITS 64', () => {
    const json = loadContractJson()
    expect(json.bundle_id).toBe(BUNDLE_ID)
    expect(json.trained).toBe(true)
    expect(json.reader_input).toBe('flattened')
    expect(json.TILE_PX).toBe(384)
    expect(json.MESSAGE_BITS).toBe(64)
  })

  it('1.2, 1.5: parseContract reads the model numbers and the bundle_id', () => {
    const c = parseContract(loadContractJson())
    expect(c.bundleId).toBe(BUNDLE_ID)
    expect(c.tilePx).toBe(384)
    expect(c.messageBits).toBe(64)
    expect(c.trainedPxPerTile).toEqual([231, 340])
    expect(c.streamShortSide).toBe(1080)
    expect(c.framesAveraged).toBe(3)
    expect(c.flatten.target_size).toEqual([384, 384])
    expect(c.reader.input).toEqual({ name: 'image', shape: [1, 3, 384, 384] })
    expect(c.reader.output).toEqual({ name: 'logits', shape: [1, 64] })
  })

  const altered: [string, (j: Record<string, any>) => void, RegExp][] = [
    ['trained false', (j) => (j.trained = false), /trained/],
    ['reader_input direct', (j) => (j.reader_input = 'direct'), /reader_input/],
    ['MESSAGE_BITS 63', (j) => (j.MESSAGE_BITS = 63), /MESSAGE_BITS/],
    ['target_size 256', (j) => (j.flatten.target_size = [256, 256]), /flatten\.target_size/],
    ['TILE_PX missing', (j) => delete j.TILE_PX, /TILE_PX/],
    ['frames_averaged missing', (j) => delete j.reading.frames_averaged, /reading\.frames_averaged/],
    ['trained_px_per_tile reversed', (j) => (j.camera.trained_px_per_tile = [340, 231]), /camera\.trained_px_per_tile/],
    ['output shape [1, 32]', (j) => (j.models.reader.output.shape = [1, 32]), /models\.reader\.output\.shape/],
  ]
  it.each(altered)('1.3: rejects a contract with %s, naming the field', (_name, change, field) => {
    const j = structuredClone(loadContractJson()) as Record<string, any>
    change(j)
    expect(() => parseContract(j)).toThrow(field)
  })

  it('1.3: rejects a non-object', () => {
    expect(() => parseContract(null)).toThrow(/contract\.json/)
    expect(() => parseContract([])).toThrow(/contract\.json/)
  })
})
