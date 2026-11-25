"""Export a bundle for the browser: ONNX models, contract.json and fixtures.

    python export_bundle.py checkpoints/stage1/step_00140000.pt
    python export_bundle.py --untrained
    python export_bundle.py --untrained --set train.reader_input=direct

Writes ``export/<bundle_id>/`` where bundle_id is the date plus the short git
hash, for example ``20261120-ea8f718``:

    reader.onnx      the reader, one file, for ONNX Runtime Web
    locator.onnx     only once the locator exists
    contract.json    names, shapes, value ranges, the flatten step, the camera
                     the models were trained for, checksums
    fixtures/        camera-like frames with corners, flattened patch and
                     expected logits and bits

``/docs/model-contract.md`` explains contract.json.

With a checkpoint, the models and every setting come from the checkpoint's
own saved config. With ``--untrained`` the weights are random (fixed seed) and
the config is ``--config`` plus any ``--set dotted.key=value`` overrides; such
a bundle is for wiring only and reads nothing.

Every ONNX file is run in ONNX Runtime on the fixture inputs and on random
inputs and compared with PyTorch. If the difference exceeds TOLERANCE the
script stops before contract.json is written, so a folder with a
contract.json is a folder that passed.

The locator part cannot be tried yet: no locator exists. It follows the
contract stated in models/locator.py.
"""

from __future__ import annotations

import argparse
import collections
import copy
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from camera import camera_summary, check_demo, frame_size
from config import load_config
from constants import MESSAGE_BITS, TILE_PX
from data import procedural_image
from geometry import center_crop, flatten_from_corners, flatten_random_patch
from models import Models, build_models
from simulate import direct_scale_range, simulate, wrap_surface

HERE = Path(__file__).resolve().parent
# 2: the `reading` block (the app averages the reader's logits over 3 frames of a scan).
CONTRACT_VERSION = 2
OPSET = 20
# Largest allowed difference between ONNX Runtime and PyTorch, in logit units.
TOLERANCE = 1e-4
READER_INPUTS = ("flattened", "direct")
CORNER_ORDER = ["top-left", "top-right", "bottom-right", "bottom-left"]
# Fixture frames: [width, height] in the 9:16 shape of the phone stream, and a
# surface large enough to fill them. Portrait on purpose: a flatten that mixes
# up width and height passes on a square frame and fails on this one.
FIXTURE_FRAME_PX = [768, 1366]
FIXTURE_SURFACE_REPEAT = 5
COORDINATES = (
    "x to the right, y down, in pixels of the image. The top-left corner of the image is (0, 0) "
    "and the centre of the pixel in column j, row i is (j + 0.5, i + 0.5)."
)


# -- small helpers ------------------------------------------------------------


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()


def git_state() -> dict:
    """Commit of the repository and whether /training differs from it."""
    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "short": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", ".")),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot read the git commit, which the bundle id needs: {exc}") from exc


def to_uint8(image: torch.Tensor) -> np.ndarray:
    """(3,H,W) in [0,1] -> (H,W,3) uint8, rounding to the nearest level."""
    return (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def from_uint8(pixels: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 -> (1,3,H,W) float32 in [0,1]. This is the browser's conversion too."""
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    """Indented JSON with lists of plain values kept on one line."""
    text = json.dumps(data, indent=2)
    text = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", text)
    path.write_text(text + "\n", encoding="utf-8")


# -- loading the models ---------------------------------------------------------


def load_models(args: argparse.Namespace) -> tuple[Models, dict, dict]:
    """Returns (models on the CPU in eval mode, config, facts about where they came from)."""
    if args.untrained:
        cfg = load_config(args.config, args.overrides)
        torch.manual_seed(args.seed)
        models = build_models(cfg)
        source = {"trained": False, "checkpoint_step": None}
    else:
        if args.overrides:
            raise SystemExit("overrides only apply with --untrained; a checkpoint brings its own config")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        cfg = state["config"]
        models = build_models(cfg)
        named = models.named()
        if set(state["models"]) != set(named):
            raise SystemExit(f"{args.checkpoint} holds {sorted(state['models'])}, its config builds {sorted(named)}")
        for name, module in named.items():
            module.load_state_dict(state["models"][name])
        source = {"trained": True, "checkpoint_step": int(state["step"])}
    if cfg["train"]["reader_input"] not in READER_INPUTS:
        raise SystemExit(f"train.reader_input must be one of {READER_INPUTS}")
    for module in models.named().values():
        module.cpu().float().eval()
    return models, cfg, source


# -- fixtures -------------------------------------------------------------------


@torch.no_grad()
def make_fixtures(models: Models, cfg: dict, directory: Path, count: int, seed: int) -> list[dict]:
    """Write the fixture images and return one record per fixture.

    Each fixture is made the way a training sample is: a generated tile is
    marked with random bits, repeated, and put through the print-and-camera
    simulation. Two things differ from training: the frame is portrait
    (FIXTURE_FRAME_PX) and the tile is repeated more often to fill it. The
    frame is then saved as an 8-bit PNG, and everything after
    that starts from the saved pixels, so the browser can reproduce the chain
    exactly: frame PNG -> flatten with the listed corners -> patch PNG ->
    reader -> logits.

    The flatten is done in float64 so the reference patch does not depend on
    float32 rounding.
    """
    directory.mkdir(parents=True)
    direct = cfg["train"]["reader_input"] == "direct"
    sim_cfg = copy.deepcopy(cfg)
    sim_cfg["simulation"]["enabled"] = True
    sim_cfg["simulation"]["frame_px"] = list(FIXTURE_FRAME_PX)
    repeat = max(int(cfg["surface"]["repeat"]), FIXTURE_SURFACE_REPEAT)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed + 1)
    strengths = np.linspace(0.2, 1.0, count) if count > 1 else np.array([1.0])
    records = []
    for index, strength in enumerate(strengths, start=1):
        tile = from_uint8(procedural_image(rng, TILE_PX))
        embedded = torch.randint(0, 2, (1, MESSAGE_BITS)).float()
        marked = (tile + models.embedder(tile, embedded)).clamp(0.0, 1.0)
        surface = wrap_surface(marked, repeat)
        sim = simulate(surface, sim_cfg, float(strength), direct=direct)

        frame_pixels = to_uint8(sim.frame[0])
        frame = from_uint8(frame_pixels)
        height, width = frame.shape[-2:]
        if direct:
            top, left = (height - TILE_PX) // 2, (width - TILE_PX) // 2
            corners = torch.tensor(
                [[left, top], [left + TILE_PX, top], [left + TILE_PX, top + TILE_PX], [left, top + TILE_PX]],
                dtype=torch.float64,
            )
            patch = center_crop(frame, TILE_PX)
        else:
            # True corners of one tile-sized square of the surface, no jitter.
            # Rounded first, so the numbers in the JSON are the numbers used.
            _, exact = flatten_random_patch(frame, sim.homography, TILE_PX, corner_jitter_px=0.0)
            corners = (exact[0].double() * 1000.0).round() / 1000.0
            patch = flatten_from_corners(frame.double(), corners.unsqueeze(0), TILE_PX)
        patch_pixels = to_uint8(patch[0])

        name = f"{index:02d}"
        Image.fromarray(frame_pixels).save(directory / f"{name}_frame.png")
        Image.fromarray(patch_pixels).save(directory / f"{name}_patch.png")
        records.append(
            {
                "name": name,
                "frame": frame,
                "reader_input": from_uint8(patch_pixels),
                "json": {
                    "frame": f"{name}_frame.png",
                    "frame_size": [width, height],
                    "corners": [[round(float(x), 3), round(float(y), 3)] for x, y in corners.tolist()],
                    "patch": f"{name}_patch.png",
                    "simulation_strength": round(float(strength), 3),
                    "embedded_bits": [int(b) for b in embedded[0].tolist()],
                },
            }
        )
    return records


# -- ONNX export and check ------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    module: torch.nn.Module
    input_name: str
    output_name: str
    example: torch.Tensor
    input_doc: dict
    output_doc: dict


def model_specs(models: Models, cfg: dict) -> list[ModelSpec]:
    if cfg["train"]["reader_input"] == "flattened":
        content = (
            "One flattened patch: a square of the surface exactly one tile period on a side, upright, "
            "starting anywhere within the tile pattern. Made by the flatten step below."
        )
    else:
        content = "The centre crop of the camera frame, not flattened. See center_crop below."
    specs = [
        ModelSpec(
            name="reader",
            module=models.reader,
            input_name="image",
            output_name="logits",
            example=torch.rand(1, 3, TILE_PX, TILE_PX),
            input_doc={"content": content},
            output_doc={
                "content": (
                    "One number per message bit. logits[0][i] belongs to Message index i of "
                    "/docs/message-format.md. Bit i is 1 when the number is greater than 0, else 0."
                ),
                "range": "any real number",
            },
        )
    ]
    if models.locator is not None:
        frame_w, frame_h = frame_size(cfg)
        specs.append(
            ModelSpec(
                name="locator",
                module=models.locator,
                input_name="frame",
                output_name="corners",
                example=torch.rand(1, 3, frame_h, frame_w),
                input_doc={"content": "One camera frame."},
                output_doc={
                    "content": (
                        "Four corners of one tile-sized square of the surface, as (x, y) in pixels of "
                        "the input frame. They go straight into the flatten step."
                    ),
                    "corner_order": CORNER_ORDER,
                    "coordinates": COORDINATES,
                    "range": "frame pixels; a corner may lie outside the frame",
                },
            )
        )
    return specs


def export_onnx(spec: ModelSpec, path: Path) -> dict:
    """Export one model to a single ONNX file and return what the file declares."""
    import onnx

    torch.onnx.export(
        spec.module,
        (spec.example,),
        str(path),
        input_names=[spec.input_name],
        output_names=[spec.output_name],
        opset_version=OPSET,
        dynamo=True,
        external_data=False,  # ONNX Runtime Web loads one file
        verbose=False,
    )
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)

    def describe(values) -> dict:
        (value,) = values
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in value.type.tensor_type.shape.dim]
        return {"name": value.name, "shape": dims, "dtype": onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type).name}

    declared_in, declared_out = describe(model.graph.input), describe(model.graph.output)
    if declared_in["name"] != spec.input_name or declared_out["name"] != spec.output_name:
        raise SystemExit(f"{path.name}: the exporter renamed an input or output: {declared_in}, {declared_out}")
    if any(not isinstance(d, int) for d in declared_in["shape"] + declared_out["shape"]):
        raise SystemExit(f"{path.name}: expected fixed shapes, got {declared_in}, {declared_out}")
    external = [t.name for t in model.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    if external or model.functions:
        raise SystemExit(f"{path.name}: not self-contained (external data: {external}, functions: {len(model.functions)})")
    domains = {n.domain or "ai.onnx" for n in model.graph.node}
    if domains != {"ai.onnx"}:
        raise SystemExit(f"{path.name}: uses operators outside the standard ONNX set: {sorted(domains)}")
    operators = collections.Counter(n.op_type for n in model.graph.node)
    opset = {o.domain or "ai.onnx": o.version for o in model.opset_import}
    return {
        "input": declared_in,
        "output": declared_out,
        "opset": opset["ai.onnx"],
        "operators": dict(sorted(operators.items())),
    }


@torch.no_grad()
def compare_with_pytorch(spec: ModelSpec, path: Path, inputs: dict[str, torch.Tensor]) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Run the ONNX file in ONNX Runtime and PyTorch on the same inputs."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rows, reference = [], {}
    for label, x in inputs.items():
        want = spec.module(x).numpy()
        got = session.run([spec.output_name], {spec.input_name: x.numpy()})[0]
        reference[label] = want
        row = {
            "input": label,
            "max_abs_diff": float(np.abs(got - want).max()),
            "largest_output": float(np.abs(want).max()),
        }
        if spec.name == "reader":
            row["bits_equal"] = int(((got > 0) == (want > 0)).sum())
        rows.append(row)
    return rows, reference


# -- contract ---------------------------------------------------------------------


def flatten_contract() -> dict:
    return {
        "input": "the camera frame as RGB, and four corners in frame pixels",
        "corner_order": CORNER_ORDER,
        "corner_format": "[x, y]",
        "coordinates": COORDINATES,
        "target_size": [TILE_PX, TILE_PX],
        "target_corners": [[0, 0], [TILE_PX, 0], [TILE_PX, TILE_PX], [0, TILE_PX]],
        "mapping": (
            "Find the homography H (3x3, perspective) that takes the four target_corners to the four "
            "frame corners, in the order listed. For the output pixel in column j, row i, compute "
            "(u, v, w) = H * (j + 0.5, i + 0.5, 1) and read the frame at (x, y) = (u / w, v / w)."
        ),
        "interpolation": "bilinear",
        "interpolation_detail": (
            "Blend the four frame pixels whose centres surround (x, y): with fx = x - 0.5 and "
            "fy = y - 0.5, the columns are floor(fx) and floor(fx) + 1 with weights 1 - frac(fx) and "
            "frac(fx), and the same for rows. No smoothing or mipmapping before sampling."
        ),
        "outside_frame": "pixels outside the frame count as 0 (black)",
        "colour": "no colour conversion; sample the 8-bit RGB values as they are",
        "output": (
            "float values; divide by 255 to get the reader input. In use, do not round. When checking against "
            "a fixture, round to whole 8-bit levels first: the expected logits come from the 8-bit patch."
        ),
        "reference_code": "training/geometry.py, flatten_from_corners",
    }


def crop_contract(cfg: dict) -> dict:
    d = cfg["simulation"]["direct"]
    low, high = direct_scale_range(cfg)
    return {
        "size": [TILE_PX, TILE_PX],
        "position": "centred: left = (frame width - size) // 2, top = (frame height - size) // 2",
        "resampling": "none; the crop copies frame pixels",
        "trained_rotation_deg": d["rotation_deg"],
        "trained_rotation_near_right_angles_deg": d.get("rotation_near_right_angles_deg"),
        "trained_px_per_tile": [round(low * TILE_PX), round(high * TILE_PX)],
        "trained_px_per_tile_means": "width of one printed tile in the crop, in pixels, seen straight on",
        "trained_scale": [round(low, 4), round(high, 4)],
        "simulated_frame_px": [TILE_PX, TILE_PX],
        "simulated_frame_px_order": "[width, height]",
        "simulated_frame_is": "the crop itself: training draws only the square the reader is given",
    }


def camera_contract(cfg: dict, models: Models) -> dict | None:
    """The camera the models were trained for, and the band the code started in."""
    camera = camera_summary(cfg)
    if camera is None:
        return None
    band = models.band
    camera["starting_band"] = {
        "cycles_per_tile": [round(band.low_cycles, 1), round(band.high_cycles, 1)],
        "grating_step": band.grating_step,
        "camera_px_per_cycle_of_highest": (
            None if band.px_per_cycle_low is None
            else [round(band.px_per_cycle_low, 2), round(band.px_per_cycle_high, 2)]
        ),
        "note": "where training starts; the trained code is free to sit elsewhere",
    }
    return camera


def reading_contract() -> dict:
    """How the reader's output is meant to be used: acceptance reads 3 correlated frames averaged."""
    return {
        "frames_averaged": 3,
        "frames_averaged_means": "average the reader's logits over the last 3 frames of one scan, then read the bits",
        "scan_distance_cm": [19, 28],
        "scan_distance_means": "phone to print; the trained camera range (camera.trained_px_per_tile) is this distance on the measured phone",
    }


def build_contract(
    bundle_id: str, repo: dict, source: dict, cfg: dict, models: Models, model_entries: dict,
    fixture_names: list[str], checks: dict,
) -> dict:
    import onnx
    import onnxruntime as ort

    flattened = cfg["train"]["reader_input"] == "flattened"
    return {
        "contract_version": CONTRACT_VERSION,
        "bundle_id": bundle_id,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": repo["commit"],
        "git_dirty": repo["dirty"],
        "trained": source["trained"],
        "checkpoint_step": source["checkpoint_step"],
        "reader_input": cfg["train"]["reader_input"],
        "MESSAGE_BITS": MESSAGE_BITS,
        "TILE_PX": TILE_PX,
        "models": model_entries,
        "flatten": flatten_contract() if flattened else None,
        "center_crop": None if flattened else crop_contract(cfg),
        "camera": camera_contract(cfg, models),
        "reading": reading_contract(),
        "bits": {
            "rule": "bit i = 1 if logits[i] > 0 else 0",
            "order": "index i is Message index i in /docs/message-format.md",
            "decoding": "error correction and checksum are not part of the models; see /docs/message-format.md",
        },
        "fixtures": {
            "directory": "fixtures",
            "names": fixture_names,
            "files_per_fixture": ["<name>_frame.png", "<name>_patch.png", "<name>.json"],
            "expected_values_from": "PyTorch, reading <name>_patch.png as 8-bit RGB divided by 255",
            "compare": "the logits, not only the bits; /docs/model-contract.md says how closely",
            "embedded_bits_follow_message_format": False,
            "frames_from": "generated images put through the training simulation; no camera, no dataset",
            "frame_size": list(FIXTURE_FRAME_PX) if flattened else [TILE_PX, TILE_PX],
            "frame_size_order": "[width, height]",
            "frame_is": (
                "a portrait window onto the surface at TILE_PX pixels per tile, in the 9:16 shape of the "
                "camera stream but not at its size; a real frame is camera.stream_px"
                if flattened
                else "the TILE_PX square the reader is given, in camera pixels; it stands for the centre of a real frame"
            ),
        },
        "verification": {
            "what": "each ONNX file run in ONNX Runtime (CPU) against PyTorch (CPU, float32) on the same inputs",
            "tolerance": TOLERANCE,
            **checks,
        },
        "exported_with": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    }


# -- main -------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("checkpoint", nargs="?", help="checkpoint file written by train.py")
    source.add_argument("--untrained", action="store_true", help="random weights; for wiring the browser only")
    parser.add_argument("--config", default=str(HERE / "configs" / "default.yaml"), help="config for --untrained")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="config override for --untrained, e.g. --set train.reader_input=direct")
    parser.add_argument("--out", default=str(HERE / "export"), help="folder the bundle folder is created in")
    parser.add_argument("--fixtures", type=int, default=5, help="number of fixture frames")
    parser.add_argument("--seed", type=int, default=0, help="seed for the fixtures and for --untrained weights")
    args = parser.parse_args()
    # The ONNX exporter prints characters a Windows console code page cannot encode.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    repo = git_state()
    bundle_id = f"{datetime.date.today():%Y%m%d}-{repo['short']}"
    bundle = Path(args.out) / bundle_id
    models, cfg, source = load_models(args)
    check_demo(cfg)
    if bundle.exists():
        print(f"replacing existing {bundle}")
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    fixtures = make_fixtures(models, cfg, bundle / "fixtures", args.fixtures, args.seed)
    generator = torch.Generator().manual_seed(args.seed + 2)

    model_entries: dict = {"reader": None, "locator": None}
    checks: dict = {}
    failed = False
    for spec in model_specs(models, cfg):
        path = bundle / f"{spec.name}.onnx"
        declared = export_onnx(spec, path)
        key = "reader_input" if spec.name == "reader" else "frame"
        inputs = {f"fixture {f['name']}": f[key] for f in fixtures}
        for n in range(3):
            inputs[f"random {n + 1}"] = torch.rand(spec.example.shape, generator=generator)
        rows, reference = compare_with_pytorch(spec, path, inputs)
        worst = max(row["max_abs_diff"] for row in rows)
        passed = worst <= TOLERANCE
        failed |= not passed
        checks[spec.name] = {"passed": passed, "max_abs_diff": worst, "inputs": rows}

        print(f"\n{path.name}: ONNX Runtime against PyTorch, tolerance {TOLERANCE:g}")
        for row in rows:
            bits = f"   bits equal {row['bits_equal']}/{MESSAGE_BITS}" if "bits_equal" in row else ""
            print(f"  {row['input']:12s} max |difference| {row['max_abs_diff']:.3e}   largest output {row['largest_output']:.3f}{bits}")
        print(f"  {'PASS' if passed else 'FAIL'}: worst difference {worst:.3e}")

        model_entries[spec.name] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "opset": declared["opset"],
            "operators": declared["operators"],
            "input": {
                **declared["input"],
                "layout": "NCHW: [batch, channel, row, column], rows top to bottom",
                "channel_order": "RGB",
                "range": [0.0, 1.0],
                "scale": "8-bit value divided by 255",
                **spec.input_doc,
            },
            "output": {**declared["output"], **spec.output_doc},
        }
        for fixture in fixtures:
            want = reference[f"fixture {fixture['name']}"][0]
            if spec.name == "reader":
                bits = [int(v > 0) for v in want]
                embedded = fixture["json"]["embedded_bits"]
                fixture["json"]["logits"] = [float(v) for v in want]
                fixture["json"]["bits"] = bits
                fixture["json"]["bits_equal_to_embedded"] = sum(int(a == b) for a, b in zip(bits, embedded))
            else:
                fixture["json"]["locator_corners"] = [[float(x), float(y)] for x, y in want]

    if failed:
        raise SystemExit(f"\nONNX and PyTorch disagree by more than {TOLERANCE:g}. No contract.json written; do not use {bundle}.")

    for fixture in fixtures:
        write_json(bundle / "fixtures" / f"{fixture['name']}.json", fixture["json"])
    contract = build_contract(bundle_id, repo, source, cfg, models, model_entries, [f["name"] for f in fixtures], checks)
    write_json(bundle / "contract.json", contract)

    print(f"\n{bundle}")
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        print(f"  {path.stat().st_size:>10,d}  {path.relative_to(bundle).as_posix()}")
    print(f"\nbundle_id {bundle_id}   trained {source['trained']}   reader_input {cfg['train']['reader_input']}")
    if repo["dirty"]:
        print("note: /training has uncommitted changes, so the commit in the bundle id is not the exact code that made it (git_dirty: true)")


if __name__ == "__main__":
    main()
