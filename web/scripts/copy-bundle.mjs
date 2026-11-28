// Copy the model bundle unchanged into web/public/models/<bundle_id>/ (tile-reader design D13).
// Source: TILEMARK_BUNDLE_SRC, default ../training/export/20261126-ba0ab4d from /web.
// Reads the source only. Checks the sha256 of reader.onnx against contract.json.
import { createHash } from 'node:crypto'
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const BUNDLE_ID = '20261126-ba0ab4d'
const webDir = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const src = resolve(process.env.TILEMARK_BUNDLE_SRC ?? join(webDir, '..', 'training', 'export', BUNDLE_ID))
const dst = join(webDir, 'public', 'models', BUNDLE_ID)

if (!existsSync(join(src, 'contract.json'))) {
  console.error(`copy-bundle: no contract.json in ${src}. Set TILEMARK_BUNDLE_SRC to the bundle folder.`)
  process.exit(1)
}
const contract = JSON.parse(readFileSync(join(src, 'contract.json'), 'utf8'))
if (contract.bundle_id !== BUNDLE_ID) {
  console.error(`copy-bundle: ${src} holds bundle ${contract.bundle_id}, expected ${BUNDLE_ID}`)
  process.exit(1)
}

function* files(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) yield* files(p)
    else yield p
  }
}

let count = 0
for (const file of files(src)) {
  const target = join(dst, relative(src, file))
  mkdirSync(dirname(target), { recursive: true })
  copyFileSync(file, target)
  if (!readFileSync(file).equals(readFileSync(target))) {
    console.error(`copy-bundle: ${target} differs from ${file} after copying`)
    process.exit(1)
  }
  count++
}

const onnx = readFileSync(join(dst, contract.models.reader.file))
const sha = createHash('sha256').update(onnx).digest('hex')
if (sha !== contract.models.reader.sha256 || onnx.length !== contract.models.reader.bytes) {
  console.error(`copy-bundle: reader.onnx sha256 ${sha} (${onnx.length} bytes) does not match contract.json`)
  process.exit(1)
}
console.log(`copy-bundle: ${count} files from ${src} to ${dst}; reader.onnx sha256 ${sha}`)
