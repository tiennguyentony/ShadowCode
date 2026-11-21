# Third-party material used by /training

Every licence below was read on 2026-11-20 from the source named in the
"Checked against" column. Where a check could not be completed, the row says
so. Nothing in this file is assumed.

Add a row whenever code, weights or data from outside this repository are
brought in.

## Open points to settle before any commercial use

1. **LPIPS backbone weights (AlexNet, ImageNet-1K).** torchvision publishes
   them without a licence statement, and its README warns that pretrained
   weights may carry terms derived from the training dataset. The ImageNet
   terms of access could not be read (image-net.org refused the connection),
   so their wording is **not verified** here. Setting `loss.lpips_weight: 0`
   trains without these weights.
2. **The training images (Open Images V7, validation split).** The images are
   listed as CC BY 2.0, which has no non-commercial clause, but Google gives
   no warranty that each listing is right and no image was checked against
   its Flickr page here. See "Datasets". Two things follow from CC BY:
   any dataset image that is published (a marked sample in a demo, a
   TensorBoard sheet in a write-up) needs its author credited from
   `ATTRIBUTION.csv`; and whether trained weights count as derived from the
   images is not settled law, which this file does not try to decide.
3. **JPEG quantisation tables.** No authoritative statement on the copyright
   status of the standard's example tables was found.

## Borrowed code

No file was copied verbatim from another project. One design was ported and
one published formula is used. The embedder should be treated as adapted from
StegaStamp, whose MIT licence asks for its copyright notice to be kept with
substantial portions of the software.

| What | Where it is used | Origin | Licence | Checked against |
|---|---|---|---|---|
| Layer layout of the U-Net (five down stages, four up stages with skip connections, last stage also sees the raw input) and the idea of a dense layer turning the bits into a map that is concatenated with the image | `models/embedder.py` | StegaStamp, Tancik, Mildenhall, Ng (CVPR 2020), https://github.com/tancik/StegaStamp. Ported to PyTorch (the original is TensorFlow) and changed: circular padding, GroupNorm, tanh-bounded output, gratings init. | MIT License, "Copyright (c) 2019 Matthew Tancik, Ben Mildenhall, Ren Ng" | `LICENSE` in the repository, master branch |
| Rounding with a gradient, `round(x) + (x - round(x))^3`, in the approximate JPEG | `simulate.py`, `_soft_round` | Shin and Song, "JPEG-resistant adversarial images" (2017). A formula from the paper, also used by StegaStamp. No code taken. | Not applicable (a published formula) | Not applicable |

## Data embedded in the code

| What | Where it is used | Origin | Licence | Checked against |
|---|---|---|---|---|
| Luminance and chrominance quantisation tables, 64 numbers each | `simulate.py`, `_LUMA_TABLE` and `_CHROMA_TABLE` | JPEG standard, ITU-T T.81 (1992), Annex K, tables K.1 and K.2 | **Not verified.** The copy of T.81 at w3.org could not be read for a copyright notice, and the libjpeg-turbo licence, which ships the same tables, does not mention them. | https://www.w3.org/Graphics/JPEG/itu-t81.pdf (unreadable), libjpeg-turbo `LICENSE.md` (silent) |
| RGB to YCbCr coefficients and the quality-to-table scaling rule | `simulate.py`, `jpeg_approx` | JFIF / ITU-R BT.601 constants, and the scaling convention of libjpeg | Constants and a convention; no code taken | Not applicable |

## Pretrained weights

Only the LPIPS loss uses pretrained weights. The embedder and the reader are
trained from scratch.

| What | When it is used | Origin | Licence | Checked against |
|---|---|---|---|---|
| LPIPS linear-layer weights (`alex.pth`, `vgg.pth`, `squeeze.pth`, v0.1) | `loss.lpips_weight > 0`; shipped inside the `lpips` package | Zhang, Isola, Efros, Shechtman, Wang, https://github.com/richzhang/PerceptualSimilarity. Trained on the authors' BAPPS dataset. | The repository's licence: two-clause BSD text, "Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang". No separate licence is given for the weights or for BAPPS. | `LICENSE` in the installed `lpips-0.1.4` package and in the repository, master branch |
| AlexNet backbone, `AlexNet_Weights.IMAGENET1K_V1` (default, `loss.lpips_net: alex`) | Downloaded by torchvision from download.pytorch.org on first use, about 233 MB | torchvision model zoo; trained on ImageNet-1K | **No licence stated for the weights.** torchvision's README says pretrained models "may have their own licenses or terms and conditions derived from the dataset used for training". ImageNet terms: **not verified**, see open point 1. | torchvision `README.md` (main branch) and the AlexNet weights page at docs.pytorch.org |
| VGG16 or SqueezeNet backbone (`loss.lpips_net: vgg` or `squeeze`) | Only if selected in the config | torchvision model zoo; trained on ImageNet-1K | Same position as AlexNet. Not checked individually. | As above |

## Datasets

The training dataset was chosen and its licence read on 2026-11-21.

| What | When it is used | Licence |
|---|---|---|
| **Open Images V7, validation split** (Google). 41,620 JPEG photographs from Flickr, 12.88 GB. Default subset: the first 16,000 by image ID, 4.94 GB. Downloaded by `fetch_dataset.py` from `open-images-dataset.s3.amazonaws.com/validation/`; nothing from it is stored in this repository. | Training images, `data.train_dir` | Images: **CC BY 2.0**. Details below. |
| Smoke-test images and bundle fixtures | Generated by `data.py` (`procedural_image`): random colour fields, shapes and grain. Used by `smoke_test.py` and for the fixture frames written by `export_bundle.py`. | Generated by this repository's own code; nothing downloaded |
| ImageNet-1K, BAPPS | Not used directly. They are what the LPIPS weights above were trained on. | See "Pretrained weights" |

### Open Images V7, validation split

What was read, and where:

- **The dataset's licence statement**, at
  https://storage.googleapis.com/openimages/web/factsfigures_v7.html, section
  "Licenses": "The annotations are licensed by Google LLC under CC BY 4.0
  license. The images are listed as having a CC BY 2.0 license." followed by
  the note that Google "make[s] no representations or warranties regarding
  the license status of each image and you should verify the license for each
  image yourself."
- **The per-image licence column** of the split's image list
  (`validation-images-with-rotation.csv`, 15 MB, from
  `storage.googleapis.com/openimages/2018_04/validation/`). All 41,620 rows
  give `https://creativecommons.org/licenses/by/2.0/`. `fetch_dataset.py`
  stops if a row in the part it downloads says anything else.
- **The CC BY 2.0 legal code**, at
  https://creativecommons.org/licenses/by/2.0/legalcode. Section 3 grants the
  rights to reproduce the work and to create and reproduce derivative works,
  in all media and formats. The text has no non-commercial restriction.
  Section 4 requires that copies which are distributed or shown publicly keep
  the copyright notices and credit the author.
- **The sizes** (12.88 GB for the split, 4.94 GB for the default subset) are
  sums over the public bucket listing, not estimates.

What was **not** verified:

- No image's licence was checked at its Flickr page. An author can also have
  changed the licence on Flickr since the list was made (2018). Section 3
  calls the grant "perpetual (for the duration of the applicable
  copyright)"; that is the licence's own wording, not legal advice.
- The image list is annotation data under CC BY 4.0. That licence's text was
  not read through here. The list is downloaded next to the images and is
  not redistributed.

How the terms are met:

- **Attribution.** `fetch_dataset.py` writes `ATTRIBUTION.csv` beside the
  images: image ID, author, author profile URL, title, licence and source URL
  for every image fetched. Keep it with the dataset.
- **No redistribution.** The dataset folder is git-ignored and is never
  committed. Exported bundles use generated images for their fixtures, not
  dataset images.
- **Only the images are used.** No labels, boxes or other annotations.

## Libraries (installed with pip, not copied into this repository)

Licences as declared in each installed package's metadata and licence file.

| Package | Version checked | Licence | Checked against |
|---|---|---|---|
| torch | 2.14.1+cu126 | BSD-3-Clause for PyTorch itself; the wheel declares `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` for bundled components | package metadata and `LICENSE`; repository `LICENSE` (three-clause BSD text) |
| torchvision | 0.29.1+cu126 | BSD 3-Clause License, "Copyright (c) Soumith Chintala 2016" | package `LICENSE`; repository `LICENSE` |
| numpy | 2.5.4 | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | package metadata |
| pillow | 13.0.0 | `MIT-CMU` | package metadata |
| PyYAML | 6.0.3 | MIT | package metadata |
| lpips | 0.1.4 | Two-clause BSD text (metadata says only "BSD License") | package `LICENSE` |
| tensorboard | 2.21.0 | Apache License 2.0 | package metadata; repository `LICENSE` |
| onnx | 1.23.2 | `Apache-2.0` | package metadata; the package ships `LICENSE` and `NOTICE` |
| onnxruntime | 1.32.0 | MIT License | package metadata; the package ships `LICENSE` and `ThirdPartyNotices.txt` (not read through) |
| onnxscript | 0.7.2 | MIT License, "Copyright (c) Microsoft Corporation" | package metadata and `LICENSE` |

`onnx`, `onnxruntime` and `onnxscript` are used by `export_bundle.py` only.
An exported `reader.onnx` holds this project's own weights in the ONNX file
format; no third-party weights go into it.

`lpips`, `tensorboard` and the ONNX packages pull in further packages (among
them scipy, tqdm, protobuf, grpcio, absl-py, markdown, werkzeug, onnx_ir,
ml_dtypes, flatbuffers). Those were not checked one by one. `pip-licenses`
lists them all if a full audit is needed.

The CUDA build of torch bundles NVIDIA libraries under NVIDIA's own licence
terms; those were not reviewed.
