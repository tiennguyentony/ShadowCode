/**
 * The Tile_Reader source the Race_Screen uses (race task 8): the real reader of
 * spec tile-reader, web/src/reader/, on bundle 20261126-ba0ab4d.
 *
 * The model loads once, on the first preload() or createReader(). bundleId is the
 * bundle folder's name until contract.json is loaded, then contract.json's bundle_id;
 * the loader rejects a contract whose bundle_id differs from the folder's.
 */
import { loadTileReader, type TileReaderSource as LoadedSource } from '../reader/browser'
import type { ExecutionProvider, ReadFrame, TileReaderSource } from './tileReader'

export const BUNDLE_ID = '20261126-ba0ab4d'

/** The bundle folder as served from public/: <site>/models/<bundle_id>/, the race page being <site>/race/. */
function bundleUrl(): string {
  return new URL(`../models/${BUNDLE_ID}/`, globalThis.location.href).href
}

let loaded: LoadedSource | null = null
let loading: Promise<LoadedSource> | null = null

function load(): Promise<LoadedSource> {
  loading ??= loadTileReader(bundleUrl(), { bundleId: BUNDLE_ID }).then(
    (source) => (loaded = source),
    (e) => {
      loading = null // a later Start may try again
      throw e
    },
  )
  return loading
}

export const TILE_READER_SOURCE: TileReaderSource = Object.freeze({
  get bundleId(): string {
    return loaded?.bundleId ?? BUNDLE_ID
  },
  standIn: false,
  get executionProvider(): ExecutionProvider | null {
    return loaded?.executionProvider ?? null
  },
  async preload(): Promise<void> {
    await load()
  },
  async createReader(): Promise<ReadFrame> {
    return (await load()).createReader()
  },
})
