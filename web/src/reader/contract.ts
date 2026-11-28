/**
 * The bundle's contract.json, checked once and typed (tile-reader Requirement 1, design D7).
 * Every model number the reader uses comes from here.
 */
import { MESSAGE_BITS } from '../codec/message'
import type { FlattenSpec } from './flatten'
import type { Point } from './types'

export interface TensorSpec {
  readonly name: string
  readonly shape: readonly number[]
}

export interface ReaderContract {
  readonly bundleId: string
  /** TILE_PX */
  readonly tilePx: number
  /** MESSAGE_BITS */
  readonly messageBits: number
  /** camera.trained_px_per_tile */
  readonly trainedPxPerTile: readonly [number, number]
  /** The short side of camera.stream_px. */
  readonly streamShortSide: number
  /** reading.frames_averaged */
  readonly framesAveraged: number
  readonly flatten: FlattenSpec
  readonly reader: {
    readonly file: string
    readonly bytes: number
    readonly sha256: string
    readonly input: TensorSpec
    readonly output: TensorSpec
  }
}

type Json = Record<string, unknown>

function fail(field: string, message: string): never {
  throw new Error(`contract.json: ${field} ${message}`)
}

function obj(v: unknown, field: string): Json {
  if (typeof v !== 'object' || v === null || Array.isArray(v)) fail(field, 'must be an object')
  return v as Json
}

function num(v: unknown, field: string): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) fail(field, `must be a number, got ${JSON.stringify(v)}`)
  return v
}

function posInt(v: unknown, field: string): number {
  const n = num(v, field)
  if (!Number.isInteger(n) || n <= 0) fail(field, `must be a positive integer, got ${n}`)
  return n
}

function str(v: unknown, field: string): string {
  if (typeof v !== 'string' || v.length === 0) fail(field, `must be a non-empty string, got ${JSON.stringify(v)}`)
  return v
}

function numArray(v: unknown, field: string, length?: number): number[] {
  if (!Array.isArray(v) || (length !== undefined && v.length !== length)) {
    fail(field, `must be an array${length === undefined ? '' : ` of ${length} numbers`}, got ${JSON.stringify(v)}`)
  }
  return v.map((x, i) => num(x, `${field}[${i}]`))
}

function tensor(v: unknown, field: string): TensorSpec {
  const o = obj(v, field)
  return Object.freeze({ name: str(o.name, `${field}.name`), shape: Object.freeze(numArray(o.shape, `${field}.shape`)) })
}

/** Check a parsed contract.json and return the fields the reader uses. Throws an Error naming the field. */
export function parseContract(json: unknown): ReaderContract {
  const c = obj(json, '(root)')
  if (c.trained !== true) fail('trained', `must be true (the weights must be trained), got ${JSON.stringify(c.trained)}`)
  if (c.reader_input !== 'flattened') fail('reader_input', `must be "flattened", got ${JSON.stringify(c.reader_input)}`)
  const tilePx = posInt(c.TILE_PX, 'TILE_PX')
  const messageBits = posInt(c.MESSAGE_BITS, 'MESSAGE_BITS')
  if (messageBits !== MESSAGE_BITS) fail('MESSAGE_BITS', `must be ${MESSAGE_BITS}, the Message size of the codec, got ${messageBits}`)

  const models = obj(c.models, 'models')
  const r = obj(models.reader, 'models.reader')
  const input = tensor(r.input, 'models.reader.input')
  const output = tensor(r.output, 'models.reader.output')
  if (input.shape.join() !== [1, 3, tilePx, tilePx].join()) {
    fail('models.reader.input.shape', `must be [1, 3, TILE_PX, TILE_PX], got ${JSON.stringify(input.shape)}`)
  }
  if (output.shape.join() !== [1, messageBits].join()) {
    fail('models.reader.output.shape', `must be [1, MESSAGE_BITS], got ${JSON.stringify(output.shape)}`)
  }

  const f = obj(c.flatten, 'flatten')
  const size = numArray(f.target_size, 'flatten.target_size', 2)
  if (size[0] !== tilePx || size[1] !== tilePx) fail('flatten.target_size', `must be [TILE_PX, TILE_PX], got ${JSON.stringify(size)}`)
  if (!Array.isArray(f.target_corners) || f.target_corners.length !== 4) fail('flatten.target_corners', 'must hold 4 corners')
  const targetCorners = f.target_corners.map((p, i) => numArray(p, `flatten.target_corners[${i}]`, 2) as unknown as Point)

  const camera = obj(c.camera, 'camera')
  const range = numArray(camera.trained_px_per_tile, 'camera.trained_px_per_tile', 2)
  if (!(range[0] > 0 && range[0] < range[1])) fail('camera.trained_px_per_tile', `must be [low, high] with 0 < low < high, got ${JSON.stringify(range)}`)
  const stream = numArray(camera.stream_px, 'camera.stream_px', 2)
  const reading = obj(c.reading, 'reading')

  return Object.freeze({
    bundleId: str(c.bundle_id, 'bundle_id'),
    tilePx,
    messageBits,
    trainedPxPerTile: Object.freeze([range[0], range[1]] as const),
    streamShortSide: posInt(Math.min(stream[0], stream[1]), 'camera.stream_px'),
    framesAveraged: posInt(reading.frames_averaged, 'reading.frames_averaged'),
    flatten: Object.freeze({ target_size: Object.freeze([size[0], size[1]] as const), target_corners: Object.freeze(targetCorners) }),
    reader: Object.freeze({
      file: str(r.file, 'models.reader.file'),
      bytes: posInt(r.bytes, 'models.reader.bytes'),
      sha256: str(r.sha256, 'models.reader.sha256'),
      input,
      output,
    }),
  })
}
