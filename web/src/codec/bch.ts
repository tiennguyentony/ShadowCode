/**
 * BCH(63,39) code, t = 4, over GF(2^6), as defined in /docs/message-format.md.
 *
 * Field: primitive polynomial p(x) = x^6 + x + 1 (0x43).
 * Generator: g(x) = m1 * m3 * m5 * m7 = 0x1DB2777 (degree 24).
 * Message index i (0..62) holds the coefficient of x^(62 - i).
 *
 * Internal helper of message.ts. Module-level values are immutable tables only;
 * every scratch array is allocated per call.
 */

const N = 63
const T = 4

/** g(x) without its x^24 term, used as the LFSR feedback mask. */
const G_FEEDBACK = 0xdb2777

/** EXP[i] = alpha^i for i in [0, 125] (doubled to avoid a modulo). */
const EXP: Uint8Array = new Uint8Array(2 * N)
/** LOG[x] = i with alpha^i = x, for x in [1, 63]. LOG[0] is unused. */
const LOG: Uint8Array = new Uint8Array(N + 1)

{
  let x = 1
  for (let i = 0; i < N; i++) {
    EXP[i] = x
    LOG[x] = i
    x <<= 1
    if (x & 0x40) x ^= 0x43
  }
  for (let i = N; i < 2 * N; i++) EXP[i] = EXP[i - N]
}

/** SYN_ROWS[r][i] = alpha^((j * (62 - i)) mod 63) for j = 1, 3, 5, 7 (r = 0..3). */
const SYN_ROWS: readonly Uint8Array[] = [1, 3, 5, 7].map((j) => {
  const row = new Uint8Array(N)
  for (let i = 0; i < N; i++) row[i] = EXP[(j * (62 - i)) % N]
  return row
})

function gfMul(a: number, b: number): number {
  return a === 0 || b === 0 ? 0 : EXP[LOG[a] + LOG[b]]
}

function gfDiv(a: number, b: number): number {
  // b is never 0 at the call site.
  return a === 0 ? 0 : EXP[LOG[a] + N - LOG[b]]
}

/** Odd syndromes S1, S3, S5, S7 of a 63-element word, written into out[0..3]. Returns true if all are zero. */
function oddSyndromes(r: Uint8Array, out: Uint8Array): boolean {
  let s1 = 0
  let s3 = 0
  let s5 = 0
  let s7 = 0
  const [r1, r3, r5, r7] = SYN_ROWS
  for (let i = 0; i < N; i++) {
    if (r[i]) {
      s1 ^= r1[i]
      s3 ^= r3[i]
      s5 ^= r5[i]
      s7 ^= r7[i]
    }
  }
  out[0] = s1
  out[1] = s3
  out[2] = s5
  out[3] = s7
  return (s1 | s3 | s5 | s7) === 0
}

/**
 * Reads m[0..38] and writes the 24 systematic parity bits into m[39..62]:
 * index 39 receives the x^23 coefficient and index 62 the x^0 coefficient.
 */
export function bchParity(m: Uint8Array): void {
  let reg = 0
  for (let i = 0; i < 39; i++) {
    const fb = (m[i] ^ (reg >>> 23)) & 1
    reg = (reg << 1) & 0xffffff
    if (fb) reg ^= G_FEEDBACK
  }
  for (let j = 0; j < 24; j++) m[39 + j] = (reg >>> (23 - j)) & 1
}

/**
 * Corrects a 63-element 0/1 array in place.
 *
 * @returns the number of positions located and flipped (0..4), or -1 if the
 * word is uncorrectable (the array is then left unchanged).
 */
export function bchCorrect(r: Uint8Array): number {
  // 1. Syndromes. S[k] holds S_k for k = 1..8.
  const odd = new Uint8Array(4)
  if (oddSyndromes(r, odd)) return 0
  const S = new Uint8Array(2 * T + 1)
  S[1] = odd[0]
  S[3] = odd[1]
  S[5] = odd[2]
  S[7] = odd[3]
  S[2] = gfMul(S[1], S[1])
  S[4] = gfMul(S[2], S[2])
  S[6] = gfMul(S[3], S[3])
  S[8] = gfMul(S[4], S[4])

  // 2. Berlekamp-Massey, 2t = 8 iterations. deg C <= L <= 8, so 9 coefficients suffice.
  const C = new Uint8Array(2 * T + 1)
  const B = new Uint8Array(2 * T + 1)
  const tmp = new Uint8Array(2 * T + 1)
  C[0] = 1
  B[0] = 1
  let L = 0
  let shift = 1
  let b = 1
  for (let n = 0; n < 2 * T; n++) {
    let d = S[n + 1]
    for (let i = 1; i <= L; i++) d ^= gfMul(C[i], S[n + 1 - i])
    if (d === 0) {
      shift++
      continue
    }
    const coef = gfDiv(d, b)
    if (2 * L <= n) {
      tmp.set(C)
      for (let i = 0; i + shift <= 2 * T; i++) C[i + shift] ^= gfMul(coef, B[i])
      L = n + 1 - L
      B.set(tmp)
      b = d
      shift = 1
    } else {
      for (let i = 0; i + shift <= 2 * T; i++) C[i + shift] ^= gfMul(coef, B[i])
      shift++
    }
  }

  // 3. Failure checks: too many errors, or deg Lambda != L.
  if (L > T) return -1
  if (C[L] === 0) return -1
  for (let i = L + 1; i <= 2 * T; i++) if (C[i] !== 0) return -1

  // 4. Chien search: index i (degree p = 62 - i) is an error if Lambda(alpha^-p) = 0.
  const roots = new Int8Array(T)
  let count = 0
  for (let i = 0; i < N; i++) {
    const e = (N - (62 - i)) % N // exponent of alpha^-p
    let v = C[0]
    for (let j = 1; j <= L; j++) {
      if (C[j]) v ^= EXP[(LOG[C[j]] + e * j) % N]
    }
    if (v === 0) {
      if (count === T) return -1
      roots[count++] = i
    }
  }
  if (count !== L) return -1

  // 5. Flip, then 6. re-check syndromes; undo on failure.
  for (let k = 0; k < count; k++) r[roots[k]] ^= 1
  if (!oddSyndromes(r, odd)) {
    for (let k = 0; k < count; k++) r[roots[k]] ^= 1
    return -1
  }
  return L
}
