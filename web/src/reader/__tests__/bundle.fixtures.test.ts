// Requirement 1.1: the copy in web/public/models/ equals the source bundle in training/export/.
// Needs the git-ignored source bundle and the copy's fixtures/ (npm run copy-bundle).
import { createHash } from 'node:crypto'
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { describe, expect, it } from 'vitest'
import { BUNDLE_ID, bundleDir, requireDir, WEB_DIR } from './fixtures'

const sha256 = (buf: Buffer) => createHash('sha256').update(buf).digest('hex')

function* files(dir: string): Generator<string> {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) yield* files(p)
    else yield p
  }
}

describe('bundle copy (1.1)', () => {
  it('every file of the copy is byte-identical to the source bundle', () => {
    const src = requireDir(
      process.env.TILEMARK_BUNDLE_SRC ?? join(WEB_DIR, '..', 'training', 'export', BUNDLE_ID),
      'source bundle',
      'Set TILEMARK_BUNDLE_SRC to the bundle folder (it exists only in the main checkout, D:\\Coding\\ShadowCode\\training\\export\\20261126-ba0ab4d).',
    )
    requireDir(join(bundleDir(), 'fixtures'), "the copy's fixtures/ folder", 'Run `npm run copy-bundle` in /web.')
    const copied = [...files(bundleDir())].map((f) => relative(bundleDir(), f)).sort()
    const source = [...files(src)].map((f) => relative(src, f)).sort()
    expect(copied).toEqual(source)
    for (const rel of source) {
      expect(sha256(readFileSync(join(bundleDir(), rel))), rel).toBe(sha256(readFileSync(join(src, rel))))
    }
    expect(existsSync(join(bundleDir(), 'reader.onnx'))).toBe(true)
  })
})
