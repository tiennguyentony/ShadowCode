/**
 * QR_Text_Parser (race spec, Requirement 4.4 and 4.5).
 *
 * A QR code on a box holds its Product_ID as decimal text, for example "1001".
 * A text counts only if it is exactly String(id) for a Catalog_ID: no spaces,
 * sign, leading zeros or other characters. Product_ID 0 is reserved and never
 * accepted.
 */
export function createQrTextParser(catalog: Iterable<number>): (text: string) => number | null {
  const byText = new Map<string, number>()
  for (const id of catalog) {
    if (Number.isInteger(id) && id > 0) byText.set(String(id), id)
  }
  return (text) => (typeof text === 'string' ? (byText.get(text) ?? null) : null)
}
