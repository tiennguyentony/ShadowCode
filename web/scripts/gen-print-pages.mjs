// Print_Generator (race spec, Requirement 8.9; design D10).
//
// Writes print/qr-codes.html and print/products.js from web/src/config.ts.
// Imports the TypeScript modules directly (Node 22.18+ / 24 type stripping).
//
// Usage (from /web): node scripts/gen-print-pages.mjs [--check]
//   --check: write nothing; exit 1 if a file is missing or differs.
import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'
import { PRODUCTS } from '../src/config.ts'
import { productsScript, qrSheetHtml } from '../src/print/qrSheet.ts'

const printDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../print')
const files = {
  'qr-codes.html': qrSheetHtml(PRODUCTS),
  'products.js': productsScript(PRODUCTS),
}
const normalize = (text) => text.replace(/\r\n/g, '\n')

if (process.argv.includes('--check')) {
  const stale = Object.entries(files).filter(([name, text]) => {
    const file = path.join(printDir, name)
    return !fs.existsSync(file) || normalize(fs.readFileSync(file, 'utf8')) !== text
  })
  for (const [name] of stale) console.error(`out of date: print/${name}`)
  if (stale.length) {
    console.error('Run `npm run print:gen` in /web.')
    process.exit(1)
  }
  console.log('print pages are up to date')
} else {
  fs.mkdirSync(printDir, { recursive: true })
  for (const [name, text] of Object.entries(files)) {
    fs.writeFileSync(path.join(printDir, name), text, 'utf8')
    console.log(`wrote print/${name}`)
  }
}
