/**
 * Tile_Reader_Interface for the race (race spec, Requirement 5; design D6, D14).
 *
 * The Tile_Reader (spec tile-reader, web/src/reader/) turns one camera frame
 * into a Decode_Result and a Reader_Hint. The source the Race_Screen uses,
 * TILE_READER_SOURCE, is in ./readerSource.ts (race task 8), because it loads
 * ONNX Runtime and this file stays pure. The Stand_In_Reader below is kept for
 * tests and as the shape of a source that reads nothing.
 */
import type { DecodeResult } from '../codec/message'

/** One camera frame: 4 bytes per pixel (R, G, B, A), row by row from the top-left. */
export interface RgbaFrame {
  readonly width: number
  readonly height: number
  readonly data: Uint8ClampedArray
}

export type ReaderHint = 'show all four corners' | 'closer' | 'farther'

export interface TileReadout {
  readonly result: DecodeResult
  readonly hint: ReaderHint | null
  /** AprilTag IDs (0 to 3) not found, given with the hint 'show all four corners'. */
  readonly missingTags?: readonly number[]
}

/** One Tile_Reader: frame in, readout out. Holds the 3-frame logit window of one Run. */
export type ReadFrame = (frame: RgbaFrame) => Promise<TileReadout>

export type ExecutionProvider = 'webgpu' | 'wasm'

export interface TileReaderSource {
  readonly bundleId: string
  /** True for the Stand_In_Reader, which never finds a product. */
  readonly standIn: boolean
  /** The ONNX Runtime execution provider in use once the model is loaded; null before, and for the stand-in. */
  readonly executionProvider: ExecutionProvider | null
  /** Starts loading the model; resolves when it is ready. Safe to call more than once. */
  preload(): Promise<void>
  /** A fresh reader with an empty frame window; called once per Tilemark Run. */
  createReader(): Promise<ReadFrame>
}

export const STAND_IN_BUNDLE_ID = '20261126-ba0ab4d'

function checkFrame(frame: RgbaFrame): void {
  const { width, height, data } = frame
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError(`RgbaFrame: width and height must be positive integers, got ${width} x ${height}`)
  }
  if (data.length !== width * height * 4) {
    throw new RangeError(`RgbaFrame: data has ${data.length} bytes, expected ${width * height * 4} for ${width} x ${height}`)
  }
}

/** Stand_In_Reader: checks the frame, reads nothing. Never invents a product. */
export const standInReaderSource: TileReaderSource = Object.freeze({
  bundleId: STAND_IN_BUNDLE_ID,
  standIn: true,
  executionProvider: null,
  async preload(): Promise<void> {},
  async createReader(): Promise<ReadFrame> {
    return async (frame: RgbaFrame): Promise<TileReadout> => {
      checkFrame(frame)
      return { result: null, hint: null }
    }
  },
})
