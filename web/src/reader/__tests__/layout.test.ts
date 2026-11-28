// Requirement 4.2: the one copy of the label layout in code equals every label.json.
import { describe, expect, it } from 'vitest'
import { LABEL_LAYOUT } from '../layout'
import { loadProducts } from './fixtures'

describe('LABEL_LAYOUT against label.json (4.2)', () => {
  const products = loadProducts()
  it.each(products.map((p) => [`${p.set}/${p.product}`, p] as const))('%s', (_name, p) => {
    const l = p.label as Record<string, any>
    expect(l.label_mm).toEqual([...LABEL_LAYOUT.labelMm])
    expect(l.artwork_mm).toEqual([...LABEL_LAYOUT.artworkMm])
    expect(l.artwork_origin_mm).toEqual([...LABEL_LAYOUT.artworkOriginMm])
    expect(l.margin_mm).toBe(LABEL_LAYOUT.marginMm)
    expect(l.tag_mm).toBe(LABEL_LAYOUT.tagMm)
    expect(l.gap_mm).toBe(LABEL_LAYOUT.gapMm)
    expect(l.tile_mm).toBe(LABEL_LAYOUT.tileMm)
    const centres = Object.fromEntries([...LABEL_LAYOUT.tagCentresMm].map(([id, c]) => [String(id), [...c]]))
    expect(l.tag_centres_mm).toEqual(centres)
    expect(l.whole_tiles).toEqual(LABEL_LAYOUT.wholeTiles.map((t) => [t.row, t.col, t.x, t.y]))
  })
})
