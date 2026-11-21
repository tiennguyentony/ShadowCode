# Model contract

This explains `contract.json`, the file that travels with every exported model bundle. It is written for the person wiring the models into the browser. Nothing in `/training` has to be read.

A bundle is made by `training/export_bundle.py` and lands in `training/export/<bundle_id>/` (the folder is git-ignored):

```
20261120-ea8f718/
  reader.onnx        the reader model, one self-contained file
  contract.json      everything below
  fixtures/
    01_frame.png     a camera-like frame
    01_patch.png     the patch the reader is given for that frame
    01.json          corners, expected logits, expected bits
    02_... to 05_...
```

`locator.onnx` will sit next to `reader.onnx` once the locator model exists. Until then `models.locator` in the contract is `null`.

## Read these four fields first

| Field | What to do with it |
|---|---|
| `trained` | `false` means the weights are random. The bundle is for wiring and for the fixture checks only; it cannot read a real mark, and its bits mean nothing. |
| `reader_input` | `flattened` or `direct`. It decides what you feed the reader, see "What the reader is given". The two need different browser code. |
| `bundle_id` | Date plus short git hash. Show it somewhere in a debug view, so a bug report says which models were running. |
| `git_dirty` | `true` means `/training` had uncommitted changes when the bundle was made, so `git_commit` is close to, but not exactly, the code that made it. |

`checkpoint_step` is the training step of the checkpoint the bundle came from, or `null` for random weights. `contract_version` goes up when the meaning of a field changes.

## The path of one frame

For `reader_input: flattened`:

1. Take a camera frame as RGB pixels.
2. Get four corners of one tile-sized square of the printed surface, in frame pixels. The locator will supply these. Until it exists, use the corners in the fixtures.
3. Flatten that quadrilateral to a 384 x 384 image (`flatten` in the contract).
4. Give it to the reader. Out come 64 numbers, the logits.
5. Bit i is 1 if logit i is greater than 0, otherwise 0.
6. Decode the 64 bits with the message codec (`/docs/message-format.md`). Error correction and the checksum are not part of the models.

`MESSAGE_BITS` (64) and `TILE_PX` (384) are fixed. Read them from the contract rather than typing the numbers into the code.

## The reader: `models.reader`

**Input** (`models.reader.input`)

| Field | Value | Meaning |
|---|---|---|
| `name` | `image` | The name to use in the feeds object. |
| `shape` | `[1, 3, 384, 384]` | One image. Batch, channel, row, column. The batch size is fixed at 1. |
| `dtype` | `float32` | |
| `channel_order` | `RGB` | Channel 0 is red. |
| `range` | `[0, 1]` | The 8-bit value divided by 255. No mean or standard deviation to subtract; the model does its own scaling. |
| `layout` | NCHW | The whole red plane first, then green, then blue. Each plane row by row from the top. Canvas pixel data is the other way round (RGBA per pixel), so it has to be rearranged. |

**Output** (`models.reader.output`)

| Field | Value | Meaning |
|---|---|---|
| `name` | `logits` | The name to read from the result. |
| `shape` | `[1, 64]` | One number per message bit. |
| `range` | any real number | Sign is the bit. Size is confidence: a value near 0 is a bit the reader is unsure of. |

Index i of the logits is Message index i of `/docs/message-format.md`: index 0 is the most significant bit of the Product_ID. The `bits` block of the contract repeats this.

**Other fields.** `file` is the file name, `bytes` and `sha256` let you check a download, `opset` and `operators` describe the ONNX file (see "ONNX Runtime Web" below).

## What the reader is given

### `reader_input: flattened`

The reader expects one square of the surface, exactly one tile period on a side, seen straight on and upright. It may start anywhere within the tile pattern: the tile repeats, so any tile-sized window holds the whole message, and the reader is trained on every offset. It is not trained on patches turned by 90 or 180 degrees, so the corner order matters.

The `flatten` block states the step exactly:

| Field | Meaning |
|---|---|
| `corner_order` | Top-left, top-right, bottom-right, bottom-left, as the surface is printed (not as the phone is held). |
| `corner_format` | Each corner is `[x, y]`. |
| `coordinates` | x to the right, y down, in frame pixels. The top-left corner of the frame is (0, 0). The centre of the pixel in column j, row i is (j + 0.5, i + 0.5). Corners are not whole numbers. |
| `target_size` | `[384, 384]`: the output image. |
| `target_corners` | The corners of the output image in the same order: (0, 0), (384, 0), (384, 384), (0, 384). |
| `mapping` | Find the homography that takes the four target corners to the four frame corners. For the output pixel in column j, row i, push (j + 0.5, i + 0.5) through it. That gives the place in the frame to read. |
| `interpolation` | Bilinear: blend the four frame pixels around that place. `interpolation_detail` gives the weights. No smoothing first. |
| `outside_frame` | A place outside the frame reads as black. |
| `colour` | Use the RGB values as they are. No colour-space conversion. |
| `output` | Keep the result as fractions; do not round to 8 bits in normal use. |

A homography is the 3 x 3 perspective transform between two views of a flat surface. Four point pairs fix it.

The code below does this step and builds the reader input. It was run against all five fixtures of bundle `20261120-ea8f718` on 2026-11-20 and reproduced the stored patches (at most 1 value in 442,368 off by one level).

```js
// Homography that takes from[k] to to[k] for k = 0..3. Nine numbers, row by row.
function homography(from, to) {
  const a = [];
  for (let k = 0; k < 4; k++) {
    const [x, y] = from[k];
    const [u, v] = to[k];
    a.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u]);
    a.push([0, 0, 0, x, y, 1, -v * x, -v * y, v]);
  }
  for (let col = 0; col < 8; col++) {
    let pivot = col;
    for (let r = col + 1; r < 8; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let r = 0; r < 8; r++) {
      if (r === col) continue;
      const f = a[r][col] / a[col][col];
      for (let c = col; c < 9; c++) a[r][c] -= f * a[col][c];
    }
  }
  return [...a.map((row, i) => row[8] / row[i]), 1];
}

// frame: { width, height, data } with RGBA bytes. corners: [[x, y] x 4] in frame pixels.
// spec: contract.flatten. Returns the reader input: Float32Array, NCHW, RGB, 0..1.
// roundToLevels: only for checking against a fixture.
function flatten(frame, corners, spec, roundToLevels = false) {
  const [tw, th] = spec.target_size;
  const h = homography(spec.target_corners, corners);
  const plane = tw * th;
  const input = new Float32Array(3 * plane);
  const at = (col, row, c) =>
    col < 0 || row < 0 || col >= frame.width || row >= frame.height ? 0 : frame.data[(row * frame.width + col) * 4 + c];
  for (let i = 0; i < th; i++) {
    for (let j = 0; j < tw; j++) {
      const sx = j + 0.5, sy = i + 0.5;
      const w = h[6] * sx + h[7] * sy + h[8];
      const fx = (h[0] * sx + h[1] * sy + h[2]) / w - 0.5;
      const fy = (h[3] * sx + h[4] * sy + h[5]) / w - 0.5;
      const x0 = Math.floor(fx), y0 = Math.floor(fy);
      const ax = fx - x0, ay = fy - y0;
      for (let c = 0; c < 3; c++) {
        let value =
          (1 - ay) * ((1 - ax) * at(x0, y0, c) + ax * at(x0 + 1, y0, c)) +
          ay * ((1 - ax) * at(x0, y0 + 1, c) + ax * at(x0 + 1, y0 + 1, c));
        if (roundToLevels) value = Math.round(value);
        input[c * plane + i * tw + j] = value / 255;
      }
    }
  }
  return input;
}
```

This is plain JavaScript on the CPU, written to be exact rather than fast. A WebGL or WebGPU version is fine as long as it passes the fixture checks below.

### `reader_input: direct`

No flatten. `flatten` is `null` and `center_crop` is set instead: the reader gets the 384 x 384 square from the middle of the frame, pixels copied as they are. `center_crop` also records the rotation and scale range the reader was trained on. This mode has only been run for a few training steps and is not shown to learn yet, so treat it as provisional.

To turn already-cropped RGBA pixels into the input:

```js
// pixels: { width, height, data } with RGBA bytes, as ImageData has them.
// Returns the reader input: Float32Array, NCHW, RGB, 0..1.
function imageToInput(pixels) {
  const plane = pixels.width * pixels.height;
  const input = new Float32Array(3 * plane);
  for (let p = 0; p < plane; p++) {
    for (let c = 0; c < 3; c++) input[c * plane + p] = pixels.data[p * 4 + c] / 255;
  }
  return input;
}
```

## Running the reader

```js
const session = await ort.InferenceSession.create(url + contract.models.reader.file, {
  executionProviders: ['webgpu', 'wasm'],
});
const { input, output } = contract.models.reader;
const result = await session.run({ [input.name]: new ort.Tensor('float32', data, input.shape) });
const logits = result[output.name].data;            // Float32Array, 64 numbers
const bits = Array.from(logits, (v) => (v > 0 ? 1 : 0));
```

## The fixtures

Each fixture is three files. `NN.json` holds:

| Field | Meaning |
|---|---|
| `frame`, `frame_size` | The frame PNG and its `[width, height]`. |
| `corners` | The four corners used for this frame, in the order and coordinates above. In direct mode they are the corners of the centre crop. |
| `patch` | The PNG of what the reader was given: the flattened patch (or the centre crop), rounded to 8 bits. |
| `logits` | The 64 numbers PyTorch produced from `patch`, read as 8-bit RGB divided by 255. |
| `bits` | Those logits turned into bits. |
| `embedded_bits` | The 64 bits that were put into the tile before it was "photographed". |
| `bits_equal_to_embedded` | How many of `bits` equal `embedded_bits`. About 32 of 64 for random weights. |
| `simulation_strength` | How hard the print-and-camera simulation was, 0 to 1. The five fixtures go from 0.2 to 1.0. |

The frames are generated images put through the training simulation (tilt, blur, colour shift, noise, JPEG-like damage). No camera and no dataset image is involved.

`embedded_bits` are random. They are not valid messages (`fixtures.embedded_bits_follow_message_format` is `false`), so do not expect the message codec to decode them. That changes once `/training` has its own copy of the codec.

### Two checks

Let L be the largest absolute value in a fixture's `logits`.

**Check 1, the model.** Decode `NN_patch.png`, build the input with `imageToInput`, run the reader. Every logit should be within L / 1000 of the expected one.

**Check 2, the flatten.** Decode `NN_frame.png`, flatten it with `corners` and `roundToLevels` set to true, run the reader. Same limit. The flattened pixels should equal `NN_patch.png`, apart from a handful of values that differ by one level.

Round only for this check. The expected logits come from the 8-bit patch, and the rounding alone moves the logits by up to about 0.5 percent of L.

Compare the logits, not only the bits. With random weights several wiring mistakes leave most bits unchanged.

When decoding a PNG through a canvas, switch off colour management and alpha premultiplication, or the pixel values shift:

```js
const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
```

### What was measured

On bundle `20261120-ea8f718` (random weights, L about 0.06) and on a throwaway 40-step checkpoint (L about 1.7), 2026-11-20. The two browser rows were run on the random-weight bundle only.

| Path | Largest logit difference, as a fraction of L |
|---|---|
| ONNX Runtime in Python against PyTorch | under 1 in a million |
| onnxruntime-web 1.31.1, wasm, in Node, check 1 | 2 in a million |
| onnxruntime-web 1.31.1, wasm, in Chromium 156, check 1 | 2 in a million |
| onnxruntime-web 1.31.1, WebGPU, in Chromium 156, check 1 | 2 in a million |
| Check 2 with the code above, wasm in Node | 5 in 100,000 at worst |

What typical mistakes do to the logits of the random-weight bundle, for comparison with the L / 1000 limit:

| Mistake | Logit difference, as a fraction of L | Bits changed, of 64 |
|---|---|---|
| Patch shifted by one pixel | about 1 to 3 percent | 0 to 1 |
| Upside down | 5 to 20 percent | 1 to 6 |
| BGR instead of RGB | 15 to 50 percent | 0 to 9 |
| Rows and columns swapped | 20 to 40 percent | 1 to 10 |
| Turned 90 degrees | 15 to 45 percent | 3 to 8 |
| Values 0 to 255 instead of 0 to 1 | tens of thousands of times L | 16 to 23 |

## ONNX Runtime Web

`reader.onnx` uses opset 20 and only standard operators (`models.reader.operators` lists them with counts): Add, AveragePool, Concat, Conv, Div, Gather, Gemm, MatMul, Mul, ReduceMean, Relu, Reshape, Sub, Transpose. Shapes are fixed. The weights are inside the file; there is no second data file to serve.

| Backend | Status | Basis |
|---|---|---|
| `wasm` | Works | Ran in Node and in Chromium 156 with onnxruntime-web 1.31.1; matches the fixtures. The package README states full ONNX operator coverage for this backend. |
| `webgpu` | Works | Ran in Chromium 156 on Windows with onnxruntime-web 1.31.1; matches the fixtures. Every operator is in the package's WebGPU operator table at opset 20. |
| `webgl` | Do not use | The file does not load (`Reduce op requires 1 input`). An opset 17 export loads but returns wrong numbers. The package marks WebGL as in maintenance mode. |
| `webnn` | Not tried | Every operator is in the package's WebNN table at opset 20; nothing was run. |

Not tried: Safari, Firefox, any phone.

One thing that cost time: `ort.env.wasm.wasmPaths` has to be an absolute URL. A relative path is resolved against the location of the ort module, not the page.

## The `verification` block

`export_bundle.py` runs each ONNX file in ONNX Runtime (Python, CPU) on the fixture inputs and on three random inputs, and compares with PyTorch. `verification.reader.max_abs_diff` is the largest difference found, in logit units; `tolerance` is the limit. If the limit is exceeded the script stops before it writes `contract.json`, so a bundle that has a `contract.json` passed.

`exported_with` lists the library versions used.

## Not in the bundle yet

- **Locator.** `models.locator` is `null`. When it exists it will take a frame and return four corners in the order and coordinates above, ready for the flatten step. Its export code is written but has never run.
- **Trained weights.** No long training run has been done.
- **Valid messages in the fixtures.** See `embedded_bits` above.
