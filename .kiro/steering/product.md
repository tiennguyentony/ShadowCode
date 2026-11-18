# Product

Tilemark replaces barcodes. A 32-bit product ID is hidden in artwork that repeats as a tile across the whole package surface. A camera reads it from any side and from a partial view — no need to flip the package to find a barcode.

## Demo

The demo is a timed race: scanning 3 packages by QR code (flip to find the code) versus by Tilemark (show any side). Tilemark must win on total time.

## Two parallel paths

- **Classical path (safe):** signal-processing codec — deterministic, no model weights, ships first.
- **Neural path (stretch):** learned encoder/decoder — higher robustness, exported to ONNX for in-browser inference.

Both paths share the same 32-bit ID contract and the same test suite.
