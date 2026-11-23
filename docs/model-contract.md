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

The frame can be any shape. The fixture frames have been portrait (768 x 1366) since 2026-11-21, so a flatten that swaps width and height fails on them. The code above was run in Node on the five portrait fixtures of an export made that day, flatten only: at most 2 values in 442,368 off by one level.

### `reader_input: direct`

No flatten. `flatten` is `null` and `center_crop` is set instead: the reader gets the 384 x 384 square from the middle of the frame, pixels copied as they are. `center_crop` also records what the reader was trained on: `trained_rotation_deg` (any rotation) and `trained_px_per_tile`, the width of one tile in the crop. This mode has only been run for a few training steps and is not shown to learn yet, so treat it as provisional.

Since 2026-11-21 direct mode is matched to the phone: `trained_px_per_tile` is `[154, 340]`, the same range as `camera.trained_px_per_tile`, so the crop is taken from the camera frame as it is, with no scaling. The crop then holds between 1.1 and 2.5 tiles across. Bundles made before that date were trained at 307 to 480 pixels per tile (`trained_scale` 0.8 to 1.25) and do not fit a phone frame.

In direct mode the fixture frames are 384 x 384, the crop itself, so they check the model but not the cropping.

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

## The camera: `camera`

What the models were trained for. Nothing here changes how the reader is fed; it says when to expect a read to work. `null` in bundles made before 2026-11-21.

| Field | Value now | Meaning |
|---|---|---|
| `tile_mm` | `50` | Printed size of one tile. |
| `stream_px`, `orientation` | `[1080, 1920]`, `portrait` | The camera stream the demo uses, `[width, height]`. Phone only. |
| `trained_px_per_tile` | `[154, 340]` | Camera pixels across one printed tile, seen straight on. Training covers this range. Outside it the reader is untested: on the measured phone, closer than about 19 cm the tile is larger than 340 pixels, and further than about 42 cm it is smaller than 154. |
| `tiles_across_stream_width` | `[3.18, 7.01]` | 1080 divided by the two ends of the range. |
| `measured` | two entries | The measurements the range was built from: Galaxy S23 Ultra, 66.3 px/cm at about 20 cm (331.5 pixels per tile) and 37.2 px/cm at about 35 cm (186 pixels per tile). |
| `starting_band` | 16 to 51.2 cycles per tile | Where the code starts in training, and how many camera pixels its finest stripes get per cycle at the two ends of the range (3.01 and 6.64). The trained code may sit elsewhere. For information. |

To tell whether a tile is in range, measure the side of the located quadrilateral in frame pixels and compare it with `trained_px_per_tile`.

The flattened patch is always 384 x 384, whatever size the tile has in the frame. At 154 pixels per tile the flatten enlarges it about 2.5 times; that is expected and is what the reader was trained on.

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

The frames are generated images put through the training simulation (tilt, blur, colour shift, loss of resolution, noise, JPEG-like damage). No camera and no dataset image is involved.

A fixture frame is 768 x 1366: portrait, in the 9:16 shape of the phone stream, but not a 1080 x 1920 camera frame. It shows the surface at 384 pixels per tile, with the camera's lower resolution blurred into it. `fixtures.frame_size` in the contract repeats the size. Do not use the fixtures to judge speed or tile size on a real frame.

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

## The locator: design note

**Status: proposed 2026-11-22, not approved. Nothing below is built or trained. No locator training starts before this note is approved.**

The locator is stage 2. It takes the camera view and returns the four corners the flatten step needs. The embedder and the reader are not changed by it and are not trained in stage 2.

### What the app does with it

1. Take the central square of the camera stream, as wide as the short side: 1080 x 1080 of the 1080 x 1920 portrait stream. This is the part the user aims at the label.
2. Shrink it to 384 x 384 with smoothing (canvas `drawImage` scaling is fine). That is the locator input: RGB, NCHW, values 0 to 1, as for the reader.
3. Run the locator. Out come four corners, `[1, 4, 2]`, in locator-input pixels, same pixel convention as `flatten.coordinates`.
4. Map each corner back to stream pixels: multiply by 1080 / 384 = 2.8125, then add the crop's top-left (0, 420).
5. Flatten from the full-resolution stream with those corners and read, four times: once with the corners as returned and once with the list rolled by one, two and three places (rolling by r puts returned corner (k + r) mod 4 in place k). That is the same square turned by 0, 90, 180 and 270 degrees.
6. Keep the read whose message decodes and whose CRC passes. If none passes, there is no read for this frame.

**Why step 5.** The tile is square and repeats, so a view of the surface looks the same, as far as geometry goes, at all four right-angle turns. The locator is not asked to tell them apart, and it would have to read the mark to do so, which is the reader's job. The reader is not trained on turned patches, so only one of the four reads should pass. Cost: up to four reader runs per frame instead of one. Risk: each of the three wrong turns is another chance for a wrong message to get past BCH and the 7-bit CRC. That chance has not been measured; it should be, on held-out samples, before the app relies on it.

### What the corners are

The square the locator returns is one tile period on a side, on the surface, centred on the surface point seen at the middle of the locator input, with its sides along the surface's rows and columns. Any such square holds the whole message, because the reader is trained on every offset within the tile pattern. The order is top-left, top-right, bottom-right, bottom-left, up to the turn in step 5 (the true top is not known to the locator). Mirror images do not occur.

### Network

- Input 384 x 384 x 3. At the camera's 154 to 340 pixels per printed tile (`camera_px_per_tile`), one tile is 55 to 121 locator-input pixels, so the input shows about 3 to 7 tiles across.
- A plain CNN: six stride-2 stages, one residual block each, widths 32, 64, 96, 128, 160, 192 (BatchNorm, ReLU), giving a 6 x 6 map; a 1 x 1 convolution to 64 channels; flatten (2,304 values); a fully connected layer of 128; a fully connected layer of 8. The map is flattened, not pooled, so the output keeps where things are.
- The 8 outputs are the four corners as offsets from a reference square (centred on the input middle, 82 pixels on a side, the geometric middle of 55 and 121), in units of that side. The exported model adds the reference back, so `locator.onnx` returns plain corners in locator-input pixels.
- 2,637,768 parameters (counted for these layers in PyTorch), under the reader's 3 million target. Same ONNX opset and operator limits as the reader.

### Training targets

Taken from the homography the simulation already returns (surface pixels to frame pixels, exact), nothing else. With H that homography and m the middle of the frame:

- c = H⁻¹(m), the surface point seen at the middle;
- the corners on the surface are c + (−192, −192), c + (192, −192), c + (192, 192), c + (−192, 192);
- the targets are those four points carried through H.

The frame here is the locator input itself: the simulation renders it directly at 384 x 384, at 55 to 121 pixels per tile, instead of at the reader's 384 pixels per tile.

### Loss

For each sample, the corner error under turn r is the mean over the four corners of the distance between predicted corner k and target corner (k + r) mod 4. The loss takes the smallest over r = 0, 1, 2, 3 (step 5 above), with distances measured as smooth L1 on x and y, switching from squared to linear at 3 reader pixels.

**Reader pixels** are the unit for the loss and for every corner error reported: locator-input pixels times 384 / (tile side in locator-input pixels for that sample), that is, pixels at the reader's scale of 384 per tile. In those units `flatten.corner_jitter_px` = 3 is the corner error the reader is trained to tolerate (each corner up to ±3 in x and y). That is 0.43 to 0.95 locator-input pixels, a hard target for a first locator; the reader-accuracy comparison below shows what any shortfall costs.

### Simulation changes it needs

- **In-plane rotation** in `sample_homography`, from `simulation.perspective.rotation_deg` (uniform in [0, rotation_deg]). Default 0, so stage-1 configs train exactly as before; 360 for the locator. Because of the turn-free loss, only the angle modulo 90 degrees matters to the locator.
- **Tile size in the frame**, new `simulation.perspective.tile_px_in_frame: [low, high]` (log-uniform), replacing the fit-to-frame scale when set. Default null (stage 1 as before). For the locator, `auto`: `camera_px_per_tile` times 384 / `demo.stream_px` short side, which gives [55, 121].
- **Only part of the tiled surface in the frame.** My reading of the request: the printed, tiled area ends somewhere in the view, as a label does. New `simulation.surface_extent_tiles: [low, high]`: from the centre point c to each of the four edges of the printed area, uniform per edge, in tiles; [0.6, 4.0] for the locator. Outside it the frame shows background: another generated label from a training seed, not tiled, at a random scale, or a flat colour, half and half. The target square always lies inside the printed area and inside the frame; a sample where it would not is drawn again. Default null (endless tiling, as before).
- Tilt, blur, colour, noise and JPEG as in stage 1. Blur lengths are given at 384 pixels per tile and are scaled to the frame's tile size. The loss-of-resolution step does nothing here: the locator input is coarser than the camera at every tile size in the range. The surface is shrunk with smoothing before the warp, as the app's shrink in step 2 is.
- `smoke_test.py` gets a check for each of the first three.

### Training data

- Tiles from `data.source: generated_labels`, training seeds only (1,000,000 and up). Half the samples are the plain label (unmarked), half are marked by the frozen embedder of a stage-1 checkpoint with random messages, so the locator learns the geometry of the label and not one embedder's mark. The checkpoint is named in `docs/runs.md` when the run starts.
- Held-out evaluation: seeds 0 to 255 (the acceptance seeds; never drawn in training), one fixed frame per seed with a fixed geometry seed, so every report is on the same 256 samples. Training never sees them.
- Adam, learning rate 1e-3 falling to 1e-5 (cosine), batch 32, everything at full strength from step 0 (there is no mark to protect).

### What is measured

On the held-out samples: corner error in reader pixels, mean and worst 5 percent; how many samples needed a turn other than 0; and the reader's bit accuracy on patches flattened with the locator's corners next to the same reader with the true corners. For that comparison one camera-resolution frame is rendered per sample, the locator gets it shrunk as in step 2, and both patches are flattened from the same full-resolution frame, as the app will do.

## Not in the bundle yet

- **Locator.** `models.locator` is `null`. When it exists it will take a frame and return four corners in the order and coordinates above, ready for the flatten step. Its export code is written but has never run.
- **Trained weights.** No long training run has been done.
- **Valid messages in the fixtures.** See `embedded_bits` above.
