// Test helper: reader.onnx of the bundle copy on onnxruntime-web wasm in Node.
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import * as ort from 'onnxruntime-web'
import { createOnnxRunReader } from '../onnx'
import type { RunReader } from '../types'
import { bundleDir, loadContract } from './fixtures'

let cached: Promise<RunReader> | null = null

export function nodeRunReader(): Promise<RunReader> {
  cached ??= (async () => {
    const contract = loadContract()
    const model = readFileSync(join(bundleDir(), contract.reader.file))
    const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] })
    return createOnnxRunReader(ort, session, contract)
  })()
  return cached
}
