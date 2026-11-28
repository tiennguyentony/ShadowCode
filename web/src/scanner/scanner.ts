/**
 * Scanner: turns one Decode_Result per Decode_Attempt into the Shown_Product.
 * Spec: .kiro/specs/scanner/requirements.md and design.md.
 *
 * Pure and stateful per instance. No camera, DOM or timer input. It sees only
 * the sequence of Decode_Results, so "consecutive" means consecutive
 * Decode_Attempts, however many camera frames were not processed in between.
 *
 * Why the catalog check (Requirement 1.6). The Message_Codec accepts some wrong
 * reads. The "Residual risk" section of /docs/message-format.md gives the
 * measured single-decode rates at maxCorrectableErrors = 4: a False_Accept_Rate
 * of 2.9 × 10⁻⁴ under heavy corruption (29 in 100,000 trials with 8, 12, 16 or
 * 32 flipped bits, seed 0x2545F491) and a Random_Input_Acceptance_Rate of 0.031%
 * (31 non-null results in 100,000 random 64-bit inputs, same seed). A falsely
 * accepted ID is an arbitrary 32-bit value, and the catalog uses a tiny part of
 * the 2^32 IDs, so discarding IDs outside the Product_Catalog removes almost all
 * of them. Product_ID 0 is reserved and is never shown, whatever the catalog holds.
 *
 * Why consecutive-match confirmation (Requirement 2.12). A false accept at the
 * 2.9 × 10⁻⁴ rate comes from one bad read. Requiring the same Catalog_ID on N
 * consecutive Decode_Attempts means that bad read must repeat with the same
 * wrong catalog ID N times in a row. How independent those N reads are depends
 * on how the tile reader schedules its 3-frame logit averages; see design.md,
 * "Frame windows and the independence argument".
 *
 * Timing (Requirement 2.13). Every successful read takes at least N
 * Decode_Attempts, where N is the Confirmation_Count. All of them count against
 * the successful-read target of under 500 ms in .kiro/steering/tech.md. This
 * relation is the input for choosing N.
 *
 * Range of N (Requirement 3.5). Only 2 and 3 are accepted: [2, 3]. N = 1 would
 * remove the consecutive-match confirmation that message-codec Requirement 12
 * assigns to the Scanner, and 2 and 3 are the only candidate values. There is
 * no default; a missing value is an error.
 *
 * Chosen value (Requirements 3.6 and 4.3). The user chose a Confirmation_Count
 * of 2 (Requirement 4, 2026-11-21), because of the measured False_Accept_Rate
 * of 2.9 × 10⁻⁴ and Random_Input_Acceptance_Rate of 0.031% above: an ID is
 * shown only if it is in the Product_Catalog and was decoded on 2 consecutive
 * Decode_Attempts. The app supplies 2 in /web/src/config.ts.
 */
import type { DecodeResult } from '../codec/message'

export const CONFIRMATION_COUNT_MIN = 2
export const CONFIRMATION_COUNT_MAX = 3

const RESERVED_ID = 0

export interface ScannerConfiguration {
  /** Product_IDs that exist as products. The Scanner only tests membership. */
  readonly productCatalog: ReadonlySet<number>
  /** N: consecutive Decode_Attempts with the same Catalog_ID before it is shown. Only 2 or 3. No default. */
  readonly confirmationCount: 2 | 3
}

export interface ScannerState {
  /** The held Catalog_ID, or null. */
  readonly candidateId: number | null
  /** Consecutive Decode_Attempts, ending with the latest, that returned candidateId. 0 when none is held. */
  readonly matchRun: number
}

export interface Scanner {
  /** Process one Decode_Attempt. Returns the Shown_Product, or null. */
  push(result: DecodeResult): number | null
  /** Current Candidate_ID and Match_Run, for tests and the debug view. */
  state(): ScannerState
}

function describeValue(value: unknown): string {
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'bigint') return `${value}n`
  if (typeof value === 'symbol') return value.toString()
  if (typeof value === 'function') return 'a function'
  if (typeof value === 'object' && value !== null) return 'an object'
  return String(value)
}

function readConfirmationCount(config: ScannerConfiguration): 2 | 3 {
  const value: unknown = config.confirmationCount
  if (value === undefined) {
    throw new RangeError(
      `Scanner_Configuration: Confirmation_Count is missing; supply an integer in [${CONFIRMATION_COUNT_MIN}, ${CONFIRMATION_COUNT_MAX}] (the app uses 2)`,
    )
  }
  if (
    typeof value !== 'number' ||
    !Number.isInteger(value) ||
    value < CONFIRMATION_COUNT_MIN ||
    value > CONFIRMATION_COUNT_MAX
  ) {
    throw new RangeError(
      `Scanner_Configuration: Confirmation_Count must be an integer in [${CONFIRMATION_COUNT_MIN}, ${CONFIRMATION_COUNT_MAX}], got ${describeValue(value)}`,
    )
  }
  return value as 2 | 3
}

export function createScanner(config: ScannerConfiguration): Scanner {
  if (typeof config !== 'object' || config === null) {
    throw new TypeError('Scanner_Configuration is missing')
  }
  const n = readConfirmationCount(config)
  const catalog = config.productCatalog

  let candidateId: number | null = null
  let matchRun = 0

  const clear = (): null => {
    candidateId = null
    matchRun = 0
    return null
  }

  return {
    push(result: DecodeResult): number | null {
      if (result === null) return clear()
      const id = result.id
      if (id === RESERVED_ID || !catalog.has(id)) return clear()
      if (id === candidateId) {
        matchRun += 1
      } else {
        candidateId = id
        matchRun = 1
      }
      return matchRun >= n ? id : null
    },
    state(): ScannerState {
      return { candidateId, matchRun }
    },
  }
}
