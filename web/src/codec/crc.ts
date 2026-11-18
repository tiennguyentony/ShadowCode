/**
 * CRC-7 of a 32-bit unsigned Product_ID, as defined in /docs/message-format.md.
 *
 * Parameters: width 7, polynomial x^7 + x^3 + 1 (0x09), init 0x00,
 * no input or output reflection, final XOR 0x7F. The 32 ID bits are fed
 * most significant bit first (Message indices 0..31), with no padding.
 *
 * Internal helper of message.ts. The caller validates `id`; it must be an
 * integer in [0, 2^32 - 1].
 *
 * @returns the checksum, an integer in [0, 127].
 */
export function crc7(id: number): number {
  let reg = 0x00
  for (let b = 31; b >= 0; b--) {
    const fb = ((reg >>> 6) & 1) ^ ((id >>> b) & 1)
    reg = (reg << 1) & 0x7f
    if (fb) reg ^= 0x09
  }
  return reg ^ 0x7f
}
