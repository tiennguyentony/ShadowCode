/**
 * Tile_Reader_Interface for the race (race spec, Requirement 5; design D6, D14).
 *
 * The Tile_Reader (spec tile-reader, web/src/reader/) turns one camera frame
 * into a Decode_Result and a Reader_Hint. It is not built yet, so the race
 * uses the Stand_In_Reader below. Task 8 of the race spec sets
 * TILE_READER_SOURCE to the real reader, with bundleId read from the bundle's
 * contract.json.
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

export interface TileReaderSource {
  readonly bundleId: string
  /** True for the Stand_In_Reader, which never finds a product. */
  readonly standIn: boolean
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
  async createReader(): Promise<ReadFrame> {
    return async (frame: RgbaFrame): Promise<TileReadout> => {
      checkFrame(frame)
      return { result: null, hint: null }
    }
  },
})

/** The source the race screen uses. Task 8: replace with the real Tile_Reader. */
export const TILE_READER_SOURCE: TileReaderSource = standInReaderSource
