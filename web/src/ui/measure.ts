// Camera calibration page: freeze a frame, tap two points 1 cm apart on a ruler,
// read the distance in stream pixels. Save frame downloads the frozen frame as PNG.

import { frameFileName } from './frameName'

const MAX_MEASUREMENTS = 5

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

const cameraSelect = $<HTMLSelectElement>('camera')
const freezeBtn = $<HTMLButtonElement>('freeze')
const saveBtn = $<HTMLButtonElement>('save')
const clearBtn = $<HTMLButtonElement>('clear')
const resEl = $('res')
const labelEl = $('label')
const gpuEl = $('gpu')
const errorEl = $('error')
const hintEl = $('hint')
const video = $<HTMLVideoElement>('video')
const canvas = $<HTMLCanvasElement>('frame')
const listEl = $('list')
const avgEl = $('avg')

const ctx = canvas.getContext('2d')!
// Untouched copy of the frozen frame, used to redraw under the markers.
const frozen = document.createElement('canvas')
const frozenCtx = frozen.getContext('2d')!

type Point = { x: number; y: number }

let stream: MediaStream | null = null
let isFrozen = false
let points: Point[] = []
const measurements: number[] = []

function showError(msg: string) {
  errorEl.textContent = msg
}

function updateInfo() {
  const track = stream?.getVideoTracks()[0]
  if (!track) return
  const s = track.getSettings()
  // videoWidth/videoHeight are the decoded frame size, i.e. what Freeze captures.
  resEl.textContent = `${video.videoWidth}x${video.videoHeight} (settings ${s.width}x${s.height}${s.frameRate ? ` @ ${Math.round(s.frameRate)} fps` : ''})`
  labelEl.textContent = track.label || '(no label)'
}

async function fillCameraList(activeId: string | undefined) {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput')
  cameraSelect.replaceChildren(
    ...devices.map((d, i) => {
      const opt = document.createElement('option')
      opt.value = d.deviceId
      opt.textContent = d.label || `Camera ${i + 1}`
      opt.selected = d.deviceId === activeId
      return opt
    }),
  )
}

async function startCamera(deviceId?: string) {
  showError('')
  stream?.getTracks().forEach((t) => t.stop())
  stream = null
  freezeBtn.disabled = true
  unfreeze()

  const video_: MediaTrackConstraints = {
    width: { ideal: 1920 },
    height: { ideal: 1080 },
    ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'environment' }),
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: video_, audio: false })
  } catch (e) {
    showError(`getUserMedia failed: ${(e as Error).name}: ${(e as Error).message}`)
    return
  }
  video.srcObject = stream
  await video.play().catch(() => {})
  const track = stream.getVideoTracks()[0]
  // Labels are only filled in after permission is granted, so list devices now.
  await fillCameraList(track.getSettings().deviceId)
  freezeBtn.disabled = false
}

video.addEventListener('loadedmetadata', updateInfo)
video.addEventListener('resize', updateInfo)

function freeze() {
  const w = video.videoWidth
  const h = video.videoHeight
  if (!w || !h) return
  frozen.width = canvas.width = w
  frozen.height = canvas.height = h
  frozenCtx.drawImage(video, 0, 0, w, h)
  points = []
  redraw()
  isFrozen = true
  video.hidden = true
  canvas.hidden = false
  saveBtn.disabled = false
  freezeBtn.textContent = 'Live'
  hintEl.textContent = 'Tap two points 1 cm apart on the ruler.'
}

function unfreeze() {
  isFrozen = false
  points = []
  video.hidden = false
  canvas.hidden = true
  saveBtn.disabled = true
  freezeBtn.textContent = 'Freeze'
  hintEl.textContent = ''
}

function redraw() {
  ctx.drawImage(frozen, 0, 0)
  // Marker size scales with the frame so it stays visible at any resolution.
  const r = Math.max(4, canvas.width / 200)
  ctx.lineWidth = Math.max(1, r / 3)
  ctx.strokeStyle = '#ff0040'
  for (const p of points) {
    ctx.beginPath()
    ctx.moveTo(p.x - r * 2, p.y)
    ctx.lineTo(p.x + r * 2, p.y)
    ctx.moveTo(p.x, p.y - r * 2)
    ctx.lineTo(p.x, p.y + r * 2)
    ctx.stroke()
  }
  if (points.length === 2) {
    ctx.strokeStyle = '#00e0ff'
    ctx.beginPath()
    ctx.moveTo(points[0].x, points[0].y)
    ctx.lineTo(points[1].x, points[1].y)
    ctx.stroke()
  }
}

function renderMeasurements() {
  listEl.replaceChildren(
    ...measurements.map((m) => {
      const li = document.createElement('li')
      li.textContent = m.toFixed(1)
      return li
    }),
  )
  avgEl.textContent = measurements.length
    ? (measurements.reduce((a, b) => a + b, 0) / measurements.length).toFixed(2)
    : '-'
}

canvas.addEventListener('pointerdown', (ev) => {
  if (!isFrozen) return
  // Convert screen (CSS) pixels to stream pixels.
  const rect = canvas.getBoundingClientRect()
  const p = {
    x: ((ev.clientX - rect.left) * canvas.width) / rect.width,
    y: ((ev.clientY - rect.top) * canvas.height) / rect.height,
  }
  if (points.length === 2) points = []
  points.push(p)
  if (points.length === 2) {
    const d = Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y)
    measurements.push(d)
    if (measurements.length > MAX_MEASUREMENTS) measurements.shift()
    renderMeasurements()
    hintEl.textContent = `Last: ${d.toFixed(1)} px. Tap again for another pair, or press Live.`
  }
  redraw()
})

// The untouched frozen copy (the stream's own pixels), not the canvas with the markers on it.
function saveFrame() {
  if (!isFrozen) return
  const name = frameFileName(frozen.width, frozen.height, new Date())
  frozen.toBlob((blob) => {
    if (!blob) {
      showError('Save frame failed: the frame could not be encoded as PNG.')
      return
    }
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = name
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }, 'image/png')
}

freezeBtn.addEventListener('click', () => (isFrozen ? unfreeze() : freeze()))
saveBtn.addEventListener('click', saveFrame)
clearBtn.addEventListener('click', () => {
  measurements.length = 0
  renderMeasurements()
})
cameraSelect.addEventListener('change', () => startCamera(cameraSelect.value))

async function checkWebGpu() {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu
  if (!gpu) {
    gpuEl.textContent = 'no (navigator.gpu missing)'
    return
  }
  try {
    const adapter = await gpu.requestAdapter()
    gpuEl.textContent = adapter ? 'yes (adapter found)' : 'navigator.gpu present, but no adapter'
  } catch (e) {
    gpuEl.textContent = `navigator.gpu present, requestAdapter failed: ${(e as Error).message}`
  }
}

checkWebGpu()
renderMeasurements()
if (!navigator.mediaDevices?.getUserMedia) {
  showError('getUserMedia is not available. The page must be served over HTTPS or from localhost.')
} else {
  startCamera()
}
