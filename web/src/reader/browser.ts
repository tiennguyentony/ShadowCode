/**
 * Browser loader of the tile reader (tile-reader Requirement 12.2, design D11).
 *
 * Fetches contract.json and reader.onnx from the bundle folder, checks them, and
 * returns a source with the shape of the race spec's TileReaderSource. ONNX Runtime
 * Web runs on WebGPU where available, else wasm; never WebGL. The providers are tried
 * one at a time, so the one in use is known (executionProvider, for the debug line).
 */
import * as ort from 'onnxruntime-web'
import jsepWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url'
import { parseContract, type ReaderContract } from './contract'
import { createOnnxRunReader, EXECUTION_PROVIDERS } from './onnx'
import { createFrameReader } from './pipeline'
import type { ReadFrame } from './types'

export type ExecutionProvider = 'webgpu' | 'wasm'

export interface TileReaderSource {
  readonly bundleId: string
  readonly standIn: false
  readonly contract: ReaderContract
  /** The execution provider the session runs on. */
  readonly executionProvider: ExecutionProvider
  /** A fresh reader with an empty Logit_Window; call once per scan. */
  createReader(): Promise<ReadFrame>
}

async function fetchOk(url: URL): Promise<Response> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`tile reader: GET ${url.href} failed with ${res.status} ${res.statusText}`)
  return res
}

function hex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), (b) => b.toString(16).padStart(2, '0')).join('')
}

export interface LoadOptions {
  /** Default ['webgpu', 'wasm']: WebGPU where the browser has it, else wasm. Never WebGL. */
  readonly executionProviders?: readonly ExecutionProvider[]
  /** If set, contract.json must have this bundle_id. */
  readonly bundleId?: string
}

/**
 * baseUrl: absolute URL of the bundle folder, for example
 * new URL('models/20261126-ba0ab4d/', document.baseURI).href.
 */
export async function loadTileReader(baseUrl: string, options: LoadOptions = {}): Promise<TileReaderSource> {
  const providers = options.executionProviders ?? (EXECUTION_PROVIDERS as readonly ExecutionProvider[])
  if (providers.length === 0 || providers.some((p) => p !== 'webgpu' && p !== 'wasm')) {
    throw new RangeError(`tile reader: execution providers must be webgpu or wasm, got ${JSON.stringify(providers)}`)
  }
  const base = new URL(baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`) // throws for a relative URL
  // wasmPaths must be absolute: a relative path is resolved against the ort module, not the page.
  ort.env.wasm.wasmPaths = { wasm: new URL(jsepWasmUrl, import.meta.url).href }

  const contract = parseContract(await (await fetchOk(new URL('contract.json', base))).json())
  if (options.bundleId !== undefined && contract.bundleId !== options.bundleId) {
    throw new Error(`tile reader: ${base.href}contract.json has bundle_id ${contract.bundleId}, expected ${options.bundleId}`)
  }
  const model = await (await fetchOk(new URL(contract.reader.file, base))).arrayBuffer()
  if (model.byteLength !== contract.reader.bytes) {
    throw new Error(`tile reader: ${contract.reader.file} has ${model.byteLength} bytes, contract.json says ${contract.reader.bytes}`)
  }
  if (globalThis.crypto?.subtle) {
    const sha = hex(await globalThis.crypto.subtle.digest('SHA-256', model))
    if (sha !== contract.reader.sha256) throw new Error(`tile reader: ${contract.reader.file} sha256 ${sha} does not match contract.json`)
  }
  // One provider at a time, so the one in use is known. One run on a blank input before
  // returning, so WebGPU compiles its shaders here and not on the first timed frame.
  const errors: string[] = []
  let chosen: { provider: ExecutionProvider; runReader: ReturnType<typeof createOnnxRunReader> } | null = null
  for (const provider of providers) {
    try {
      const session = await ort.InferenceSession.create(new Uint8Array(model), { executionProviders: [provider] })
      const runReader = createOnnxRunReader(ort, session, contract)
      await runReader(new Float32Array(contract.reader.input.shape.reduce((a, b) => a * b, 1)))
      chosen = { provider, runReader }
      break
    } catch (e) {
      errors.push(`${provider}: ${(e as Error).message}`)
    }
  }
  if (!chosen) throw new Error(`tile reader: no execution provider could run ${contract.reader.file} (${errors.join('; ')})`)
  const { provider, runReader } = chosen
  return Object.freeze({
    bundleId: contract.bundleId,
    standIn: false as const,
    contract,
    executionProvider: provider,
    async createReader(): Promise<ReadFrame> {
      return createFrameReader({ contract, runReader })
    },
  })
}
