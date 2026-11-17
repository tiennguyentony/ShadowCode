import { describe, it, expect } from 'vitest'
import * as fc from 'fast-check'

describe('tilemark placeholder', () => {
  it('passes a trivial property test', () => {
    fc.assert(
      fc.property(fc.string(), (s) => {
        expect(typeof s).toBe('string')
      })
    )
  })
})
