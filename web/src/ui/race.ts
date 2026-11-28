// Race_Screen: QR mode versus Tilemark mode on one rear Camera_Stream.
// Spec: .kiro/specs/race/requirements.md and design.md. The timing and results
// rules are in ../race/race.ts; this file does camera, frame loops and display.

import type jsQRFunction from 'jsqr'
import { PRODUCTS, SCANNER_CONFIGURATION, productName } from '../config'
import { createFrameRateCounter } from '../race/frameRate'
import { createQrTextParser } from '../race/qrText'
import {
  MODES,
  createRace,
  elapsedMs,
  formatSeconds,
  isRunning,
  productFound,
  reset,
  results,
  runningMode,
  start,
  type Mode,
  type RaceState,
} from '../race/race'
import { TILE_READER_SOURCE } from '../race/readerSource'
import type { ReadFrame, RgbaFrame, TileReadout } from '../race/tileReader'
import { createScanner, type Scanner } from '../scanner/scanner'

/** jsQR runs on a copy of the frame with this short side (design D9). */
const JSQR_SHORT_SIDE = 540

// BarcodeDetector is not in TypeScript's DOM library; this is the part used here.
interface DetectedBarcode {
  readonly rawValue: string
}
interface BarcodeDetectorLike {
  detect(source: CanvasImageSource): Promise<DetectedBarcode[]>
}
interface BarcodeDetectorClass {
  new (options: { formats: string[] }): BarcodeDetectorLike
  getSupportedFormats(): Promise<string[]>
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

const video = $<HTMLVideoElement>('video')
const startBtn: Record<Mode, HTMLButtonElement> = { qr: $('start-qr'), tilemark: $('start-tilemark') }
const resetBtn = $<HTMLButtonElement>('reset')
const timerBox: Record<Mode, HTMLElement> = { qr: $('timer-qr'), tilemark: $('timer-tilemark') }
const timeOut: Record<Mode, HTMLElement> = { qr: $('time-qr'), tilemark: $('time-tilemark') }
const totalCell: Record<Mode, HTMLElement> = { qr: $('total-qr'), tilemark: $('total-tilemark') }
const rowsEl = $('rows')
const hintEl = $('hint')
const noteEl = $('note')
const errorEl = $('error')
const differenceEl = $('difference')
const debugEl = $('debug')

const MODE_NAME: Record<Mode, string> = { qr: 'QR', tilemark: 'Tilemark' }
const parseQrText = createQrTextParser(PRODUCTS.map((p) => p.id))
const frameRate = createFrameRateCounter()

let race: RaceState = createRace(PRODUCTS.map((p) => p.id))
/** Incremented by Start and Reset; a frame loop stops when its token is stale. */
let token = 0
let cameraReady = false
let qrDetector: BarcodeDetectorLike | null = null
let qrReaderName = 'checking'
/** Loaded only when BarcodeDetector cannot read QR codes, so most phones skip its download. */
let jsQR: typeof jsQRFunction | null = null

const frameCanvas = document.createElement('canvas')
const frameCtx = frameCanvas.getContext('2d', { willReadFrequently: true })!
const smallCanvas = document.createElement('canvas')
const smallCtx = smallCanvas.getContext('2d', { willReadFrequently: true })!

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

function showError(message: string) {
  errorEl.textContent = message
}

function showHint(out: TileReadout | null) {
  if (!out || out.hint === null) {
    hintEl.textContent = ''
    return
  }
  const missing = out.missingTags?.length ? ` (missing tag ${out.missingTags.join(', ')})` : ''
  hintEl.textContent = out.hint + missing
}

function renderRows() {
  const r = results(race)
  rowsEl.replaceChildren(
    ...r.rows.map((row) => {
      const tr = document.createElement('tr')
      const name = document.createElement('td')
      name.textContent = productName(row.id) ?? String(row.id)
      tr.append(name)
      for (const mode of MODES) {
        const td = document.createElement('td')
        td.textContent = formatSeconds(row.ms[mode])
        if (row.ms[mode] !== null) td.className = 'found'
        tr.append(td)
      }
      return tr
    }),
  )
  for (const mode of MODES) totalCell[mode].textContent = formatSeconds(r.totals[mode])
  if (r.differenceMs === null) differenceEl.textContent = ''
  else if (r.faster === 'tie') differenceEl.textContent = 'Tie'
  else differenceEl.textContent = `${MODE_NAME[r.faster!]} faster by ${formatSeconds(Math.abs(r.differenceMs))}`
}

function renderButtons() {
  const running = runningMode(race)
  for (const mode of MODES) {
    startBtn[mode].disabled = !cameraReady || (running !== null && running !== mode)
    startBtn[mode].textContent = `${race.runs[mode] ? 'Restart' : 'Start'} ${MODE_NAME[mode]}`
  }
}

function setRace(next: RaceState) {
  if (next === race) return
  race = next
  renderRows()
  renderButtons()
}

function renderDebug(now: number) {
  const tilemarkOn = isRunning(race.runs.tilemark)
  const fps = tilemarkOn ? frameRate.perSecond(now).toFixed(1) : '–'
  const stream = video.videoWidth ? `${video.videoWidth}×${video.videoHeight}` : '–'
  const reader = TILE_READER_SOURCE.standIn ? ' (stand-in reader)' : ''
  const provider = TILE_READER_SOURCE.standIn ? '' : ` · reader on ${TILE_READER_SOURCE.executionProvider ?? 'loading'}`
  debugEl.textContent =
    `bundle ${TILE_READER_SOURCE.bundleId}${reader}${provider} · ${fps} frames/s · stream ${stream} · QR reader: ${qrReaderName}`
}

// Timers and the debug line, every animation frame (Requirements 3.10, 7.2).
function tick() {
  const now = performance.now()
  for (const mode of MODES) {
    const run = race.runs[mode]
    timeOut[mode].textContent = formatSeconds(elapsedMs(run, now))
    timerBox[mode].classList.toggle('running', isRunning(run))
    timerBox[mode].classList.toggle('done', run?.totalMs != null)
  }
  renderDebug(now)
  requestAnimationFrame(tick)
}

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

function nextVideoFrame(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof video.requestVideoFrameCallback === 'function') video.requestVideoFrameCallback(() => resolve())
    else requestAnimationFrame(() => resolve())
  })
}

/** The current video frame at its delivered size (Requirement 5.1). */
function captureFrame(): RgbaFrame | null {
  const w = video.videoWidth
  const h = video.videoHeight
  if (!w || !h) return null
  if (frameCanvas.width !== w) frameCanvas.width = w
  if (frameCanvas.height !== h) frameCanvas.height = h
  frameCtx.drawImage(video, 0, 0, w, h)
  const image = frameCtx.getImageData(0, 0, w, h)
  return { width: w, height: h, data: image.data }
}

async function readQrTexts(): Promise<string[]> {
  if (qrDetector) {
    const codes = await qrDetector.detect(video)
    return codes.map((c) => c.rawValue)
  }
  jsQR ??= (await import('jsqr')).default
  const w = video.videoWidth
  const h = video.videoHeight
  if (!w || !h) return []
  const scale = Math.min(1, JSQR_SHORT_SIDE / Math.min(w, h))
  const sw = Math.round(w * scale)
  const sh = Math.round(h * scale)
  if (smallCanvas.width !== sw) smallCanvas.width = sw
  if (smallCanvas.height !== sh) smallCanvas.height = sh
  smallCtx.drawImage(video, 0, 0, sw, sh)
  const code = jsQR(smallCtx.getImageData(0, 0, sw, sh).data, sw, sh, { inversionAttempts: 'dontInvert' })
  return code ? [code.data] : []
}

async function qrLoop(myToken: number) {
  while (myToken === token && isRunning(race.runs.qr)) {
    await nextVideoFrame()
    if (myToken !== token) return
    let texts: string[] = []
    try {
      texts = await readQrTexts()
    } catch (e) {
      if (qrDetector) {
        showError(`BarcodeDetector failed (${(e as Error).message}); using jsQR from now on.`)
        qrDetector = null
        qrReaderName = 'jsQR'
      } else {
        showError(`jsQR failed: ${(e as Error).message}`)
      }
    }
    if (myToken !== token) return
    const now = performance.now()
    for (const text of texts) {
      const id = parseQrText(text)
      if (id !== null) setRace(productFound(race, 'qr', id, now))
    }
  }
}

async function tilemarkLoop(myToken: number, read: ReadFrame, scanner: Scanner) {
  while (myToken === token && isRunning(race.runs.tilemark)) {
    await nextVideoFrame()
    if (myToken !== token) return
    const frame = captureFrame()
    if (!frame) continue
    let out: TileReadout
    try {
      out = await read(frame)
    } catch (e) {
      showError(`Tile reader: ${(e as Error).message}`)
      out = { result: null, hint: null }
    }
    if (myToken !== token) return
    const now = performance.now()
    frameRate.record(now)
    showHint(out)
    const shown = scanner.push(out.result)
    if (shown !== null) setRace(productFound(race, 'tilemark', shown, now))
  }
  if (myToken === token) showHint(null)
}

// ---------------------------------------------------------------------------
// Buttons
// ---------------------------------------------------------------------------

async function onStart(mode: Mode) {
  if (!cameraReady) return
  const running = runningMode(race)
  if (running !== null && running !== mode) return
  showError('')
  showHint(null)
  // Reader and Scanner are made before the clock starts, so loading them is not timed.
  let read: ReadFrame | null = null
  let scanner: Scanner | null = null
  if (mode === 'tilemark') {
    try {
      read = await TILE_READER_SOURCE.createReader()
    } catch (e) {
      showError(`Tile reader could not start: ${(e as Error).message}`)
      return
    }
    scanner = createScanner(SCANNER_CONFIGURATION)
  }
  const next = start(race, mode, performance.now())
  if (next === race) return
  const myToken = ++token
  frameRate.reset()
  setRace(next)
  if (mode === 'qr') void qrLoop(myToken)
  else void tilemarkLoop(myToken, read!, scanner!)
}

function onReset() {
  token++
  frameRate.reset()
  showHint(null)
  showError('')
  setRace(reset(race))
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

async function pickQrReader() {
  const BarcodeDetector = (globalThis as { BarcodeDetector?: BarcodeDetectorClass }).BarcodeDetector
  if (BarcodeDetector) {
    try {
      const formats = await BarcodeDetector.getSupportedFormats()
      if (formats.includes('qr_code')) {
        qrDetector = new BarcodeDetector({ formats: ['qr_code'] })
        qrReaderName = 'BarcodeDetector'
        return
      }
    } catch {
      // Fall through to jsQR.
    }
  }
  qrReaderName = 'jsQR'
  try {
    jsQR = (await import('jsqr')).default
  } catch (e) {
    showError(`jsQR could not be loaded: ${(e as Error).message}`)
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showError('getUserMedia is not available. The page must be served over HTTPS or from localhost.')
    return
  }
  let stream: MediaStream
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'environment', width: { ideal: 1080 }, height: { ideal: 1920 } },
    })
  } catch (e) {
    showError(`getUserMedia failed: ${(e as Error).name}: ${(e as Error).message}`)
    return
  }
  stream.getVideoTracks()[0]?.addEventListener('ended', () => {
    cameraReady = false
    onReset()
    showError('The camera stream ended. Reload the page.')
    renderButtons()
  })
  video.srcObject = stream
  await video.play().catch(() => {})
}

video.addEventListener('playing', () => {
  cameraReady = true
  renderButtons()
})
for (const mode of MODES) startBtn[mode].addEventListener('click', () => void onStart(mode))
resetBtn.addEventListener('click', onReset)

if (TILE_READER_SOURCE.standIn) {
  noteEl.textContent = 'Stand-in tile reader: Tilemark mode cannot find products until the tile-reader spec is built.'
}
// Load the model while the page starts, so the first Tilemark Run does not wait for it.
TILE_READER_SOURCE.preload().catch((e) => showError(`Tile reader could not load: ${(e as Error).message}`))
renderRows()
renderButtons()
requestAnimationFrame(tick)
void pickQrReader()
void startCamera()
