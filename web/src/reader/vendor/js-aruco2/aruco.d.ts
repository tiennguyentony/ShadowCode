// Types for the parts of the vendored js-aruco2 2.0.0 (aruco.js) that the tile reader uses.
export interface ArucoPoint {
  x: number
  y: number
}
export interface ArucoMarker {
  id: number
  /** Contour pixel indices (column, row) of the four outer corners. */
  corners: ArucoPoint[]
  hammingDistance: number
}
export interface ArucoImage {
  width: number
  height: number
  data: ArrayLike<number>
}
export declare const AR: {
  Detector: new (config?: { dictionaryName?: string; maxHammingDistance?: number }) => { detect(image: ArucoImage): ArucoMarker[] }
  DICTIONARIES: Record<string, { nBits: number; tau?: number; codeList: unknown[] }>
}
