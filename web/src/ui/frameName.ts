// File name for a frame saved from the measure page: its size and when it was taken (UTC),
// e.g. frame-1080x1920-20261125-143000.png.

const pad = (n: number) => String(n).padStart(2, '0')

export function frameFileName(width: number, height: number, when: Date): string {
  const date = `${when.getUTCFullYear()}${pad(when.getUTCMonth() + 1)}${pad(when.getUTCDate())}`
  const time = `${pad(when.getUTCHours())}${pad(when.getUTCMinutes())}${pad(when.getUTCSeconds())}`
  return `frame-${width}x${height}-${date}-${time}.png`
}
