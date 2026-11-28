/**
 * The frame pipeline of the tile reader (spec tile-reader, design "Overview").
 *
 * One Processed_Frame in, one TileReadout out: shrink, find Tags 0 to 3, Flatten_Corners
 * of the two Whole_Tiles, flatten, reader, mean of the two tiles, Logit_Window,
 * decodeMessage. One Decode_Result after every Processed_Frame (scanner design D8).
 * The Scanner applies the catalog check and the 2-in-a-row rule.
 *
 * Pure apart from the model, which is passed in as `runReader`.
 */
import { decodeMessage } from '../codec/message'
import type { ReaderContract } from './contract'
import { flatten } from './flatten'
import { checkFrame, shrinkToShortSide } from './frame'
import { LABEL_LAYOUT, type LabelLayout } from './layout'
import { locateTiles } from './locate'
import { detectTags } from './tags'
import type { Point, ReadDetail, ReaderHint, ReadFrame, RgbaFrame, RunReader, TileReadout } from './types'
import { LogitWindow, logitsToBits, meanOf } from './logitWindow'

export interface FrameReaderOptions {
  readonly contract: ReaderContract
  readonly runReader: RunReader
  /** Defaults to LABEL_LAYOUT. */
  readonly layout?: LabelLayout
}

const now = (): number => globalThis.performance.now()

/** Distance hint from Px_Per_Tile and the Trained_Range. */
export function distanceHint(pxPerTile: number, [low, high]: readonly [number, number]): ReaderHint | null {
  if (pxPerTile < low) return 'closer'
  if (pxPerTile > high) return 'farther'
  return null
}

/** A reader instance with its own, empty Logit_Window. */
export function createFrameReader(options: FrameReaderOptions): ReadFrame {
  const { contract, runReader } = options
  const layout = options.layout ?? LABEL_LAYOUT
  const tagIds = [...layout.tagCentresMm.keys()]
  const logitWindow = new LogitWindow(contract.framesAveraged)
  let busy = false

  async function read(input: RgbaFrame): Promise<TileReadout> {
    checkFrame(input)
    const frame = shrinkToShortSide(input, contract.streamShortSide)

    const t0 = now()
    const tags = detectTags(frame, { ids: tagIds })
    const detectMs = now() - t0
    const tagCentres = new Map<number, Point>([...tags].map(([id, t]) => [id, t.centre]))
    const located = locateTiles(tagCentres, layout)

    const base = {
      frameSize: [frame.width, frame.height] as const,
      tagCentres,
    }
    if (!located.ok) {
      if (located.missingTags.length === tagIds.length) logitWindow.clear()
      const detail: ReadDetail = {
        ...base,
        tileCorners: [],
        pxPerTile: null,
        tileLogits: null,
        frameLogits: null,
        windowLogits: null,
        bits: null,
        framesAveraged: logitWindow.length,
        timing: { detectMs, flattenMs: 0, readerMs: 0 },
      }
      return { result: null, hint: 'show all four corners', missingTags: located.missingTags, detail }
    }

    let flattenMs = 0
    let readerMs = 0
    const tileLogits: Float32Array[] = []
    for (const corners of located.tileCorners) {
      const t1 = now()
      const readerInput = flatten(frame, corners, contract.flatten)
      const t2 = now()
      const logits = await runReader(readerInput)
      readerMs += now() - t2
      flattenMs += t2 - t1
      if (logits.length !== contract.messageBits) {
        throw new Error(`tile reader: the model returned ${logits.length} logits, expected MESSAGE_BITS = ${contract.messageBits}`)
      }
      tileLogits.push(Float32Array.from(logits))
    }
    const frameLogits = meanOf(tileLogits)
    logitWindow.push(frameLogits)
    const windowLogits = logitWindow.mean() as Float32Array
    const bits = logitsToBits(windowLogits)
    const result = decodeMessage(bits)
    const detail: ReadDetail = {
      ...base,
      tileCorners: located.tileCorners,
      pxPerTile: located.pxPerTile,
      tileLogits,
      frameLogits,
      windowLogits,
      bits,
      framesAveraged: logitWindow.length,
      timing: { detectMs, flattenMs, readerMs },
    }
    return { result, hint: distanceHint(located.pxPerTile, contract.trainedPxPerTile), detail }
  }

  // Frames are read one at a time; a call while one is running would mix the logitWindow.
  return async (frame: RgbaFrame): Promise<TileReadout> => {
    if (busy) throw new Error('tile reader: a frame is still being read; pass the next frame after the previous call returns')
    busy = true
    try {
      return await read(frame)
    } finally {
      busy = false
    }
  }
}
