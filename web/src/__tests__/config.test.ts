import { describe, expect, it } from 'vitest'
import { PRODUCTS, productName, SCANNER_CONFIGURATION } from '../config'
import { createScanner } from '../scanner/scanner'

describe('app configuration', () => {
  it('the Product_Catalog holds exactly 1001, 1002 and 1003', () => {
    expect([...SCANNER_CONFIGURATION.productCatalog].sort((a, b) => a - b)).toEqual([1001, 1002, 1003])
    expect(PRODUCTS.map((p) => p.id)).toEqual([1001, 1002, 1003])
  })

  it('display names are "Product 1001" to "Product 1003"', () => {
    expect(PRODUCTS.map((p) => p.name)).toEqual(['Product 1001', 'Product 1002', 'Product 1003'])
    expect(productName(1002)).toBe('Product 1002')
    expect(productName(1004)).toBeUndefined()
    expect(productName(0)).toBeUndefined()
  })

  it('4.2 the app supplies Confirmation_Count 2', () => {
    expect(SCANNER_CONFIGURATION.confirmationCount).toBe(2)
  })

  it('the app configuration is accepted by the Scanner', () => {
    const s = createScanner(SCANNER_CONFIGURATION)
    expect(s.push({ id: 1001, correctedBits: 0 })).toBeNull()
    expect(s.push({ id: 1001, correctedBits: 1 })).toBe(1001)
  })

  it('PRODUCTS is frozen', () => {
    expect(Object.isFrozen(PRODUCTS)).toBe(true)
    expect(PRODUCTS.every((p) => Object.isFrozen(p))).toBe(true)
    expect(Object.isFrozen(SCANNER_CONFIGURATION)).toBe(true)
  })
})
