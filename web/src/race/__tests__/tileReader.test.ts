import { describe, expect, it } from 'vitest'
import { BUNDLE_ID, TILE_READER_SOURCE } from '../readerSource'
import { STAND_IN_BUNDLE_ID, standInReaderSource, type RgbaFrame } from '../tileReader'

const frame = (width: number, height: number, bytes = width * height * 4): RgbaFrame => ({
  width,
  height,
  data: new Uint8ClampedArray(bytes),
})

describe('Stand_In_Reader (5.7, 7.1)', () => {
  it('returns no result and no hint for a 1080 x 1920 frame', async () => {
    const read = await standInReaderSource.createReader()
    await expect(read(frame(1080, 1920))).resolves.toEqual({ result: null, hint: null })
  })

  it('rejects a frame whose data length does not match its size', async () => {
    const read = await standInReaderSource.createReader()
    await expect(read(frame(4, 4, 63))).rejects.toThrow(/expected 64/)
    await expect(read(frame(0, 4, 0))).rejects.toThrow(RangeError)
  })

  it('gives a fresh function per Run', async () => {
    const a = await standInReaderSource.createReader()
    const b = await standInReaderSource.createReader()
    expect(a).not.toBe(b)
  })

  it('the stand-in has no execution provider and preloads at once', async () => {
    expect(STAND_IN_BUNDLE_ID).toBe('20261126-ba0ab4d')
    expect(standInReaderSource.executionProvider).toBeNull()
    await expect(standInReaderSource.preload()).resolves.toBeUndefined()
  })
})

describe('TILE_READER_SOURCE (race task 8, 7.1)', () => {
  it('is the real reader on bundle 20261126-ba0ab4d, not the stand-in', () => {
    expect(BUNDLE_ID).toBe('20261126-ba0ab4d')
    expect(TILE_READER_SOURCE.bundleId).toBe('20261126-ba0ab4d')
    expect(TILE_READER_SOURCE.standIn).toBe(false)
  })

  it('has no execution provider before the model is loaded', () => {
    expect(TILE_READER_SOURCE.executionProvider).toBeNull()
  })
})
