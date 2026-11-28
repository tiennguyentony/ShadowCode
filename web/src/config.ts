/**
 * App configuration: the Product_Catalog and the Scanner_Configuration.
 * The race screen and the code that creates the Scanner both import this
 * module, so product names live here and nowhere else. Tony replaces the
 * display names in PRODUCTS.
 */
import type { ScannerConfiguration } from './scanner/scanner'

export interface Product {
  readonly id: number
  readonly name: string
}

export const PRODUCTS: readonly Product[] = Object.freeze([
  Object.freeze({ id: 1001, name: 'Product 1001' }),
  Object.freeze({ id: 1002, name: 'Product 1002' }),
  Object.freeze({ id: 1003, name: 'Product 1003' }),
])

/** Requirement 4.2: the app supplies Confirmation_Count 2. */
export const SCANNER_CONFIGURATION: ScannerConfiguration = Object.freeze({
  productCatalog: new Set(PRODUCTS.map((p) => p.id)) as ReadonlySet<number>,
  confirmationCount: 2,
})

export function productName(id: number): string | undefined {
  return PRODUCTS.find((p) => p.id === id)?.name
}
