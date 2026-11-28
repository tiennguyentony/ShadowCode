/**
 * reader.onnx through ONNX Runtime Web (tile-reader Requirement 8).
 * Feeds and reads by the names and shapes in the Contract. Works with the Node build
 * (tests) and the browser build (browser.ts) of onnxruntime-web.
 */
import type { InferenceSession, Tensor } from 'onnxruntime-web'
import type { ReaderContract } from './contract'
import type { RunReader } from './types'

/** The part of the `ort` namespace this file needs. */
export interface OrtLike {
  readonly Tensor: new (type: 'float32', data: Float32Array, dims: readonly number[]) => Tensor
}

/** Execution providers in order of preference. WebGL is never used (docs/model-contract.md). */
export const EXECUTION_PROVIDERS: readonly string[] = Object.freeze(['webgpu', 'wasm'])

export function createOnnxRunReader(ort: OrtLike, session: InferenceSession, contract: ReaderContract): RunReader {
  const { input, output } = contract.reader
  const size = input.shape.reduce((a, b) => a * b, 1)
  return async (data: Float32Array): Promise<Float32Array> => {
    if (data.length !== size) throw new RangeError(`reader input: expected ${size} values for shape [${input.shape.join(', ')}], got ${data.length}`)
    const result = await session.run({ [input.name]: new ort.Tensor('float32', data, input.shape) })
    const out = result[output.name]
    if (!out) throw new Error(`reader output "${output.name}" missing from the model's result`)
    return Float32Array.from(out.data as Float32Array)
  }
}
