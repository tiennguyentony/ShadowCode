/**
 * Shared types of the tile reader (spec tile-reader).
 *
 * RgbaFrame, ReaderHint, TileReadout and ReadFrame have the same shape as the
 * Tile_Reader_Interface of the race spec (web/src/race/tileReader.ts on branch
 * web/race), so the race can use the reader without an adapter.
 */
import type { DecodeResult } from '../codec/message'

/** One image: 4 bytes per pixel (R, G, B, A), row by row from the top-left. */
export interface RgbaFrame {
  readonly width: number
  readonly height: number
  readonly data: Uint8ClampedArray | Uint8Array
}

/** [x, y] in frame pixels; the centre of the pixel in column j, row i is (j + 0.5, i + 0.5). */
export type Point = readonly [number, number]

/** Four corners: top-left, top-right, bottom-right, bottom-left, as the label is printed. */
export type Quad = readonly [Point, Point, Point, Point]

export type ReaderHint = 'show all four corners' | 'closer' | 'farther'

/** What the reader saw in one frame, for tests and a debug view. */
export interface ReadDetail {
  /** Size of the frame the reader worked on, after any shrink. */
  readonly frameSize: readonly [number, number]
  /** Tag centres found, by tag ID. */
  readonly tagCentres: ReadonlyMap<number, Point>
  /** Flatten corners per whole tile, in the label's whole_tiles order. Empty without a read. */
  readonly tileCorners: readonly Quad[]
  /** Mean side of the tile quadrilaterals in frame pixels, or null without a read. */
  readonly pxPerTile: number | null
  /** Logits per tile, the frame's logits (their mean) and the window's mean. Null without a read. */
  readonly tileLogits: readonly Float32Array[] | null
  readonly frameLogits: Float32Array | null
  readonly windowLogits: Float32Array | null
  /** Bits of windowLogits (logit > 0), or null without a read. */
  readonly bits: Uint8Array | null
  /** Frames in the logit window after this frame. */
  readonly framesAveraged: number
  /** Milliseconds spent per step in this frame. */
  readonly timing: { readonly detectMs: number; readonly flattenMs: number; readonly readerMs: number }
}

export interface TileReadout {
  readonly result: DecodeResult
  readonly hint: ReaderHint | null
  /** AprilTag IDs (0 to 3) not found, given with the hint 'show all four corners'. */
  readonly missingTags?: readonly number[]
  readonly detail?: ReadDetail
}

/** One Tile_Reader: frame in, readout out. Holds the logit window of one scan. */
export type ReadFrame = (frame: RgbaFrame) => Promise<TileReadout>

/** Runs reader.onnx on one reader input (NCHW, RGB, 0..1) and returns its logits. */
export type RunReader = (input: Float32Array) => Promise<Float32Array>
