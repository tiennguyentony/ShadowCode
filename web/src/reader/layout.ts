/**
 * The demo label layout (tile-reader Requirement 4), the same for every demo label.
 * Millimetres on the printed label, origin at its top-left corner, x right, y down.
 * A test checks these numbers against every fixture's label.json.
 */
import type { Point, Quad } from './types'

export interface WholeTile {
  readonly row: number
  readonly col: number
  /** Top-left corner of the tile, mm. */
  readonly x: number
  readonly y: number
}

export interface LabelLayout {
  readonly labelMm: readonly [number, number]
  readonly artworkMm: readonly [number, number]
  readonly artworkOriginMm: Point
  readonly marginMm: number
  readonly tagMm: number
  readonly gapMm: number
  readonly tileMm: number
  /** Tag centres by tag ID: 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left. */
  readonly tagCentresMm: ReadonlyMap<number, Point>
  /** The tiles that are read, in this order: row 0, then row 1. */
  readonly wholeTiles: readonly WholeTile[]
}

export const LABEL_LAYOUT: LabelLayout = Object.freeze({
  labelMm: [114, 149] as const,
  artworkMm: [70, 105] as const,
  artworkOriginMm: [22, 22] as const,
  marginMm: 22,
  tagMm: 14,
  gapMm: 4,
  tileMm: 50,
  tagCentresMm: new Map<number, Point>([
    [0, [11, 11]],
    [1, [103, 11]],
    [2, [103, 138]],
    [3, [11, 138]],
  ]),
  wholeTiles: Object.freeze([
    { row: 0, col: 0, x: 22, y: 22 },
    { row: 1, col: 0, x: 22, y: 72 },
  ]),
})

/** The tag IDs the reader needs, in order. */
export const TAG_IDS: readonly number[] = Object.freeze([...LABEL_LAYOUT.tagCentresMm.keys()])

/** Corners of a whole tile in label mm: top-left, top-right, bottom-right, bottom-left as printed. */
export function tileCornersMm(tile: WholeTile, tileMm: number): Quad {
  const { x, y } = tile
  return [
    [x, y],
    [x + tileMm, y],
    [x + tileMm, y + tileMm],
    [x, y + tileMm],
  ]
}
