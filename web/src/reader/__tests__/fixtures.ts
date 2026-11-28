// Test helpers: the Fixture_Sets in training/export/demo-fixtures and the bundle copy in
// web/public/models/. Read only. Missing folders throw with the path (Requirement 10.2).
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { PNG } from 'pngjs'
import { parseContract, type ReaderContract } from '../contract'
import type { Point, RgbaFrame } from '../types'

export const BUNDLE_ID = '20261126-ba0ab4d'
export const WEB_DIR = resolve(fileURLToPath(new URL('../../../', import.meta.url)))
export const FIXTURE_ROOT = resolve(process.env.TILEMARK_DEMO_FIXTURES ?? join(WEB_DIR, '..', 'training', 'export', 'demo-fixtures'))
export const BUNDLE_DIR = join(WEB_DIR, 'public', 'models', BUNDLE_ID)

export function requireDir(path: string, what: string, hint: string): string {
  if (!existsSync(path)) throw new Error(`${what} not found at ${path}. ${hint}`)
  return path
}

const FIXTURE_HINT =
  'Set TILEMARK_DEMO_FIXTURES to the demo-fixtures folder (it exists only in the main checkout, D:\\Coding\\ShadowCode\\training\\export\\demo-fixtures).'
const BUNDLE_HINT = 'Run `npm run copy-bundle` in /web (set TILEMARK_BUNDLE_SRC outside the main checkout).'

export interface TileFixture {
  readonly row: number
  readonly col: number
  readonly corners: readonly Point[]
  readonly patch: string
  readonly logits: readonly number[]
  readonly bits: readonly number[]
}

export interface FrameFixture {
  readonly set: string
  readonly product: string
  readonly name: string
  readonly dir: string
  readonly frameFile: string
  readonly frameSize: readonly [number, number]
  readonly tiles: readonly TileFixture[]
  readonly logits: readonly number[]
  readonly bits: readonly number[]
  readonly expectedBits: readonly number[]
  readonly productId: number
}

export interface ProductFixtures {
  readonly set: string
  readonly product: string
  readonly dir: string
  readonly label: Record<string, unknown>
  readonly frames: readonly FrameFixture[]
}

/** The Fixture_Sets present: always `synthetic` (throws if missing), plus `photos` once it exists. */
export function fixtureSets(): string[] {
  requireDir(FIXTURE_ROOT, 'Fixture_Root', FIXTURE_HINT)
  requireDir(join(FIXTURE_ROOT, 'synthetic'), 'synthetic Fixture_Set', FIXTURE_HINT)
  return ['synthetic', 'photos'].filter((s) => existsSync(join(FIXTURE_ROOT, s)))
}

export function loadProducts(): ProductFixtures[] {
  const out: ProductFixtures[] = []
  for (const set of fixtureSets()) {
    const setDir = join(FIXTURE_ROOT, set)
    const products = readdirSync(setDir, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name).sort()
    if (products.length === 0) throw new Error(`no product folders in ${setDir}`)
    for (const product of products) {
      const dir = join(setDir, product)
      const label = JSON.parse(readFileSync(join(dir, 'label.json'), 'utf8'))
      const names = readdirSync(dir).filter((f) => /^\d+\.json$/.test(f)).map((f) => f.slice(0, -5)).sort()
      if (names.length === 0) throw new Error(`no NN.json fixtures in ${dir}`)
      const frames = names.map((name): FrameFixture => {
        const j = JSON.parse(readFileSync(join(dir, `${name}.json`), 'utf8'))
        return {
          set,
          product,
          name,
          dir,
          frameFile: join(dir, j.frame ?? `${name}_frame.png`),
          frameSize: j.frame_size,
          tiles: j.tiles,
          logits: j.logits,
          bits: j.bits,
          expectedBits: j.expected_bits,
          productId: j.product_id,
        }
      })
      out.push({ set, product, dir, label, frames })
    }
  }
  return out
}

export function readPng(path: string): RgbaFrame {
  if (!existsSync(path)) throw new Error(`PNG not found at ${path}`)
  const png = PNG.sync.read(readFileSync(path))
  return { width: png.width, height: png.height, data: new Uint8ClampedArray(png.data.buffer, png.data.byteOffset, png.data.length) }
}

export function bundleDir(): string {
  return requireDir(BUNDLE_DIR, 'Model_Bundle copy', BUNDLE_HINT)
}

export function loadContractJson(): Record<string, unknown> {
  return JSON.parse(readFileSync(join(bundleDir(), 'contract.json'), 'utf8'))
}

export function loadContract(): ReaderContract {
  return parseContract(loadContractJson())
}

/** L: the largest absolute value. */
export function largestAbs(values: ArrayLike<number>): number {
  let m = 0
  for (let i = 0; i < values.length; i++) m = Math.max(m, Math.abs(values[i]))
  return m
}

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new Error(`length ${a.length} != ${b.length}`)
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]))
  return m
}

export function bitsOf(logits: ArrayLike<number>): number[] {
  return Array.from(logits as ArrayLike<number>, (v) => (v > 0 ? 1 : 0))
}

export function countDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let n = 0
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++
  return n
}
