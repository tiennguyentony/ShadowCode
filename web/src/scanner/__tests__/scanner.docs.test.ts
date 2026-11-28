// Requirements 1.6, 2.12, 2.13, 3.5, 3.6 and 4.3: the documentation lives in the
// comment block at the top of scanner.ts.
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const src = readFileSync(new URL('../scanner.ts', import.meta.url), 'utf8')
const block = src.match(/^\/\*\*[\s\S]*?\*\//)?.[0] ?? ''
const text = block.replace(/^\s*\*\s?/gm, '').replace(/\s+/g, ' ')

describe('scanner documentation block', () => {
  it('exists at the top of scanner.ts', () => {
    expect(block.length).toBeGreaterThan(0)
  })
  it('1.6 cites the Residual risk section with both measured rates for the catalog check', () => {
    expect(text).toMatch(/Why the catalog check \(Requirement 1\.6\)/)
    expect(text).toContain('"Residual risk" section of /docs/message-format.md')
    expect(text).toContain('2.9 × 10⁻⁴')
    expect(text).toContain('0.031%')
  })
  it('2.12 gives the reason for confirmation with the measured False_Accept_Rate', () => {
    expect(text).toMatch(/Why consecutive-match confirmation \(Requirement 2\.12\)\. A false accept at the 2\.9 × 10⁻⁴ rate/)
  })
  it('2.13 states at least N Decode_Attempts count against the 500 ms target', () => {
    expect(text).toContain('at least N Decode_Attempts')
    expect(text).toContain('under 500 ms')
    expect(text).toContain('input for choosing N')
  })
  it('3.5 states the reason for the range [2, 3]', () => {
    expect(text).toContain('[2, 3]')
    expect(text).toMatch(/N = 1 would remove the consecutive-match confirmation/)
  })
  it('3.6 and 4.3 state the chosen value 2 with both rates', () => {
    expect(text).toMatch(/The user chose a Confirmation_Count of 2/)
    expect(text).toMatch(/Chosen value \(Requirements 3\.6 and 4\.3\)[\s\S]*2\.9 × 10⁻⁴[\s\S]*0\.031%/)
  })
})
