"""Procedural label art: clean digital packaging labels, one per seed.

Each seed gives one label, built from random choices: a background (flat,
two-colour gradient, stripes, dots or checks), 1 to 4 panels, a made-up
brand title, a smaller line, fine print, a logo made of shapes, and
sometimes a barcode-like box, a ruled table like a nutrition panel, a border
or small icons. About 1 label in 6 is mostly flat colour, the hardest case
for hiding a mark. All words and numbers are made up; no real brand appears.

A label is laid out on a canvas whose long side is 1024 units. A tile is a
square of that canvas, chosen at a random zoom: from the whole short side of
the label down to 96 units, a close-up of one letter of the title or of one
flat panel. The tile is not cut from a rendered 1024 px picture and resized.
The layout is drawn straight at the tile's scale, so a close-up is as sharp
as a whole label: no resampling, no blur, no JPEG, no noise. Shapes are drawn
at twice the size and averaged 2 x 2 down (antialiasing), as digital art is.

    label = build_label(seed)            # the layout, reproducible from seed
    tile = render_tile(seed)             # (TILE_PX, TILE_PX, 3) uint8
    whole = render_label(label, 512)     # the whole label, long side 512 px

    python labelgen.py --sheet outputs/labelgen_sheet.png        # seeds 0 to 63, 8 x 8
    python labelgen.py --bench 300 --workers 1 8                  # tiles per second

Fonts come from ``fonts/`` beside this file; their licences are in
THIRD_PARTY.md and next to the font files.
"""

from __future__ import annotations

import argparse
import colorsys
import math
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from constants import TILE_PX

HERE = Path(__file__).resolve().parent
FONT_DIR = HERE / "fonts"

CANVAS = 1024           # long side of a label, in layout units
MIN_CROP = 96           # smallest tile side in layout units (about 4x zoom on the canvas)
SUPERSAMPLE = 2         # drawn at this many times the tile size, then averaged down
FLAT_SHARE = 1 / 6      # labels that are mostly flat colour

# (file, style). Styles: sans, serif, mono, script; "bold" in the name for weight.
FONTS = {
    "sans": "DejaVuSans.ttf",
    "sans-bold": "DejaVuSans-Bold.ttf",
    "sans-oblique": "DejaVuSans-Oblique.ttf",
    "serif": "DejaVuSerif.ttf",
    "serif-bold": "DejaVuSerif-Bold.ttf",
    "mono": "DejaVuSansMono.ttf",
    "mono-bold": "DejaVuSansMono-Bold.ttf",
    "anon": "AnonymousPro-Regular.ttf",
    "anon-bold": "AnonymousPro-Bold.ttf",
    "courier": "CourierPrime-Regular.ttf",
    "courier-bold": "CourierPrime-Bold.ttf",
    "script": "LobsterTwo-Regular.ttf",
    "script-bold": "LobsterTwo-Bold.ttf",
}
TITLE_FONTS = ["sans-bold", "sans-bold", "serif-bold", "script", "script-bold", "mono-bold", "courier-bold", "serif", "sans"]
LINE_FONTS = ["sans", "sans-bold", "sans-oblique", "serif", "serif-bold", "courier", "anon-bold", "script"]
SMALL_FONTS = ["sans", "sans", "serif", "mono", "anon", "courier"]

Color = tuple[int, int, int]


@lru_cache(maxsize=512)
def font(name: str, size: float) -> ImageFont.FreeTypeFont:
    """A font at a size in pixels, cached. Sizes are rounded to a quarter pixel."""
    return ImageFont.truetype(str(FONT_DIR / FONTS[name]), max(1.0, round(size * 4) / 4))


REF_SIZE = 64.0  # layout measures text at this size and scales: lengths grow in proportion to the size


def text_length(name: str, text: str, size: float) -> float:
    """Advance width of ``text`` at ``size``, in the same units as ``size``."""
    return font(name, REF_SIZE).getlength(text) * size / REF_SIZE


def text_bbox(name: str, text: str, size: float, anchor: str = "la") -> tuple[float, float, float, float]:
    left, top, right, bottom = font(name, REF_SIZE).getbbox(text, anchor=anchor)
    k = size / REF_SIZE
    return left * k, top * k, right * k, bottom * k


# --------------------------------------------------------------------------- words

_ONSETS = ["b", "br", "c", "ch", "d", "dr", "f", "fl", "g", "gr", "h", "k", "kl", "l", "m", "n", "p", "pl", "r", "s", "st", "t", "tr", "v", "z"]
_VOWELS = ["a", "e", "i", "o", "u", "ai", "ea", "io", "ou", "y"]
_CODAS = ["", "", "", "n", "r", "l", "s", "x", "m", "nd", "rt", "sh"]
_UNITS = ["g", "kg", "ml", "l", "oz", "mg", "kcal", "kJ", "%"]


def word(rng: np.random.Generator, syllables: tuple[int, int] = (1, 3)) -> str:
    parts = [rng.choice(_ONSETS) + rng.choice(_VOWELS) for _ in range(rng.integers(syllables[0], syllables[1] + 1))]
    return "".join(parts) + rng.choice(_CODAS)


def number(rng: np.random.Generator) -> str:
    kind = rng.integers(0, 5)
    if kind == 0:
        return f"{rng.integers(1, 1000)} {rng.choice(_UNITS)}"
    if kind == 1:
        return f"{rng.uniform(0, 100):.1f}{rng.choice(['%', ' g', ' mg'])}"
    if kind == 2:
        return f"{rng.integers(2, 13)} x {rng.integers(10, 500)} {rng.choice(['g', 'ml'])}"
    if kind == 3:
        return f"No. {rng.integers(100, 99999)}"
    return f"{rng.integers(1, 32):02d}/{rng.integers(1, 13):02d}/{rng.integers(26, 40)}"


def title_text(rng: np.random.Generator) -> str:
    words = [word(rng, (2, 3)) for _ in range(rng.choice([1, 1, 1, 2]))]
    case = rng.integers(0, 3)
    if case == 0:
        return " ".join(words).upper()
    if case == 1:
        return " ".join(w.capitalize() for w in words)
    return " ".join(words).lower()


def line_text(rng: np.random.Generator) -> str:
    words = [word(rng) for _ in range(rng.integers(2, 5))]
    if rng.random() < 0.5:
        words.append(number(rng))
    text = " ".join(words)
    return text.upper() if rng.random() < 0.4 else text.capitalize()


def sentence(rng: np.random.Generator) -> str:
    out = []
    for _ in range(rng.integers(5, 16)):
        r = rng.random()
        out.append(number(rng) if r < 0.12 else word(rng) + ("," if r > 0.9 else ""))
    return " ".join(out).rstrip(",").capitalize() + "."


# --------------------------------------------------------------------------- colour


def hsv(h: float, s: float, v: float) -> Color:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0))
    return round(r * 255), round(g * 255), round(b * 255)


def luminance(c: Color) -> float:
    def lin(x: float) -> float:
        x /= 255
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2])


def contrast(a: Color, b: Color) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def make_palette(rng: np.random.Generator) -> tuple[str, list[Color]]:
    """2 to 5 colours of one character: saturated, pale, dark, muted or mono."""
    mood = rng.choice(["saturated", "pale", "dark", "muted", "mono"], p=[0.3, 0.22, 0.18, 0.18, 0.12])
    count = int(rng.integers(2, 6))
    base = rng.random()
    scheme = rng.choice(["analogous", "complement", "triad", "random"])
    hues = []
    for i in range(count):
        if scheme == "analogous":
            hues.append(base + rng.uniform(-0.08, 0.08) * i)
        elif scheme == "complement":
            hues.append(base + 0.5 * (i % 2) + rng.uniform(-0.03, 0.03))
        elif scheme == "triad":
            hues.append(base + (i % 3) / 3 + rng.uniform(-0.03, 0.03))
        else:
            hues.append(rng.random())
    colors = []
    for i, h in enumerate(hues):
        if mood == "saturated":
            c = hsv(h, rng.uniform(0.65, 1.0), rng.uniform(0.55, 1.0))
        elif mood == "pale":
            c = hsv(h, rng.uniform(0.04, 0.3), rng.uniform(0.86, 1.0))
        elif mood == "dark":
            c = hsv(h, rng.uniform(0.3, 0.9), rng.uniform(0.08, 0.35)) if i == 0 else hsv(h, rng.uniform(0.3, 1.0), rng.uniform(0.4, 1.0))
        elif mood == "muted":
            c = hsv(h, rng.uniform(0.15, 0.5), rng.uniform(0.35, 0.85))
        else:
            v = rng.uniform(0.0, 1.0)
            c = hsv(base, rng.uniform(0.0, 0.12), v)
        colors.append(c)
    if rng.random() < 0.35:  # many labels have plain white or near-black in them
        colors.append((255, 255, 255) if rng.random() < 0.6 else (20, 20, 22))
    return str(mood), colors


def ink_for(rng: np.random.Generator, under: Color, palette: list[Color], low_contrast: float = 0.08) -> Color:
    """A text colour that reads on ``under``: the palette colour with the most contrast, or black or white.

    A share ``low_contrast`` of the time any other palette colour is taken
    as it is, as pale-on-pale print does on real labels.
    """
    others = [c for c in palette if c != under]
    if others and rng.random() < low_contrast:
        return others[int(rng.integers(len(others)))]
    best = max(others or [under], key=lambda c: contrast(c, under))
    if contrast(best, under) >= 3.0:
        return best
    return (255, 255, 255) if contrast((255, 255, 255), under) > contrast((15, 15, 15), under) else (15, 15, 15)


# --------------------------------------------------------------------------- layout

Box = tuple[float, float, float, float]  # left, top, right, bottom in layout units


@dataclass
class Label:
    seed: int
    width: int
    height: int
    kind: str                    # "flat" or "normal"
    mood: str
    background: tuple            # ("flat", c) | ("gradient", c1, c2, angle) | ("stripes"|"dots"|"checks", c1, c2, period, angle)
    ops: list = field(default_factory=list)


def _text_op(name: str, size: float, text: str, xy: tuple[float, float], fill: Color, anchor: str = "la", rotate: int = 0) -> tuple:
    """One line of text. Its box in layout units is computed now, for culling."""
    left, top, right, bottom = text_bbox(name, text, size, anchor)
    x, y = xy
    if rotate:
        # Rotated 90 degrees counterclockwise about (x, y): the text runs upwards.
        box = (x + top, y - right, x + bottom, y - left)
    else:
        box = (x + left, y + top, x + right, y + bottom)
    return ("text", box, name, size, text, xy, fill, anchor, rotate)


def fit_size(name: str, text: str, width: float, start: float, smallest: float = 6.0) -> float:
    """``start``, or less so that ``text`` fits in ``width``; never below ``smallest``."""
    length = text_length(name, text, start)
    return max(smallest, start * min(1.0, width / length)) if length > 0 else start


def add_text_block(label: Label, rng: np.random.Generator, box: Box, under: Color, palette: list[Color], size: float, name: str) -> None:
    """Wrapped fine print filling ``box``."""
    left, top, right, bottom = box
    fill = ink_for(rng, under, palette)
    lead = size * rng.uniform(1.15, 1.5)
    space = text_length(name, " ", size)
    y = top
    words: list[str] = []
    while y + size <= bottom:
        line: list[str] = []
        used = 0.0
        while True:
            if not words:
                words = sentence(rng).split()
            length = text_length(name, words[0], size)
            if line and used + space + length > right - left:
                break
            used += (space if line else 0.0) + length
            line.append(words.pop(0))
            if used > right - left:
                break
        label.ops.append(_text_op(name, size, " ".join(line), (left, y), fill))
        y += lead


def add_logo(label: Label, rng: np.random.Generator, centre: tuple[float, float], radius: float, palette: list[Color], under: Color) -> None:
    cx, cy = centre
    a, b = rng.choice(len(palette), size=2, replace=len(palette) < 2)
    outer, inner = palette[a], palette[b]
    if outer == under:
        outer = ink_for(rng, under, palette, 0.0)
    if inner == outer:
        inner = ink_for(rng, outer, palette, 0.0)
    style = rng.integers(0, 6)
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    if style in (0, 1):
        label.ops.append(("ellipse", box, outer, None, 0))
    elif style == 2:
        label.ops.append(("rrect", box, outer, radius * rng.uniform(0.1, 0.5)))
    elif style == 3:
        label.ops.append(("polygon", box, _regular(cx, cy, radius, int(rng.integers(3, 9)), rng.uniform(0, math.pi)), outer))
    elif style == 4:
        label.ops.append(("ellipse", box, None, outer, max(2.0, radius * 0.12)))
    else:
        label.ops.append(("polygon", box, _star(cx, cy, radius, radius * rng.uniform(0.4, 0.7), int(rng.integers(5, 13))), outer))
    inside = rng.integers(0, 4)
    r2 = radius * rng.uniform(0.35, 0.6)
    box2 = (cx - r2, cy - r2, cx + r2, cy + r2)
    if inside == 0:
        label.ops.append(("polygon", box2, _regular(cx, cy, r2, 3, rng.uniform(0, math.pi)), inner))
    elif inside == 1:
        label.ops.append(("ellipse", box2, inner, None, 0))
        label.ops.append(("ellipse", (cx - r2 * 0.4 + r2 * 0.5, cy - r2 * 0.9, cx + r2 * 1.3, cy + r2 * 0.5), outer, None, 0))
    elif inside == 2:
        name = str(rng.choice(["sans-bold", "serif-bold", "script-bold"]))
        letter = str(rng.choice(list("ABDEGHKMNPRSTVZ")))
        label.ops.append(_text_op(name, radius * 1.1, letter, (cx, cy), inner, anchor="mm"))
    else:
        label.ops.append(("polygon", box2, _star(cx, cy, r2, r2 * 0.45, 5), inner))


def _regular(cx: float, cy: float, r: float, n: int, phase: float) -> list[tuple[float, float]]:
    return [(cx + r * math.cos(phase + 2 * math.pi * i / n), cy + r * math.sin(phase + 2 * math.pi * i / n)) for i in range(n)]


def _star(cx: float, cy: float, r: float, r_in: float, n: int) -> list[tuple[float, float]]:
    pts = []
    for i in range(2 * n):
        rr = r if i % 2 == 0 else r_in
        ang = -math.pi / 2 + math.pi * i / n
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    return pts


def add_barcode(label: Label, rng: np.random.Generator, box: Box) -> None:
    left, top, right, bottom = box
    label.ops.append(("rect", box, (255, 255, 255), None, 0))
    pad = (right - left) * 0.08
    digits_h = (bottom - top) * 0.2
    unit = (right - left - 2 * pad) / 95
    x = left + pad
    black = True
    while x < right - pad - unit:
        w = unit * int(rng.integers(1, 5))
        if black:
            label.ops.append(("rect", (x, top + pad * 0.6, min(x + w, right - pad), bottom - digits_h - pad * 0.3), (0, 0, 0), None, 0))
        x += w
        black = not black
    digits = "".join(str(d) for d in rng.integers(0, 10, size=13))
    size = fit_size("mono", digits, right - left - 2 * pad, digits_h * 0.9)
    label.ops.append(_text_op("mono", size, digits, ((left + right) / 2, bottom - pad * 0.2), (0, 0, 0), anchor="md"))


def add_table(label: Label, rng: np.random.Generator, box: Box, under: Color, palette: list[Color]) -> None:
    """A ruled table like a nutrition panel."""
    left, top, right, bottom = box
    ink = ink_for(rng, under, palette, 0.0)
    if rng.random() < 0.5:  # a white table on its own, as many labels print it
        under = (255, 255, 255)
        ink = (15, 15, 15)
        label.ops.append(("rect", box, under, None, 0))
    rule = max(1.5, (right - left) * rng.uniform(0.004, 0.01))
    label.ops.append(("rect", box, None, ink, rule * 1.5))
    rows = int(rng.integers(5, 12))
    row_h = (bottom - top) / (rows + 1.5)
    pad = (right - left) * 0.05
    head = fit_size("sans-bold", "Xxxxx Xxxxx", (right - left) * 0.8, row_h * 1.1)
    label.ops.append(_text_op("sans-bold", head, word(rng, (2, 3)).capitalize() + " " + word(rng).capitalize(), (left + pad, top + row_h * 0.25), ink))
    y = top + row_h * 1.5
    label.ops.append(("rect", (left + pad * 0.5, y - rule * 2, right - pad * 0.5, y + rule), ink, None, 0))
    size = row_h * 0.6
    for _ in range(rows):
        name = "sans-bold" if rng.random() < 0.25 else "sans"
        label.ops.append(_text_op(name, size, word(rng).capitalize(), (left + pad, y + row_h * 0.2), ink))
        label.ops.append(_text_op("sans", size, number(rng), (right - pad, y + row_h * 0.2), ink, anchor="ra"))
        y += row_h
        label.ops.append(("rect", (left + pad * 0.5, y - rule / 2, right - pad * 0.5, y + rule / 2), ink, None, 0))


def add_icons(label: Label, rng: np.random.Generator, box: Box, palette: list[Color], under: Color) -> None:
    left, top, right, bottom = box
    count = int(rng.integers(1, 5))
    size = min(bottom - top, (right - left) / count * 0.8)
    for i in range(count):
        cx = left + (i + 0.5) * (right - left) / count
        cy = (top + bottom) / 2
        r = size / 2
        c = ink_for(rng, under, palette, 0.3)
        shape = rng.integers(0, 4)
        b = (cx - r, cy - r, cx + r, cy + r)
        if shape == 0:
            label.ops.append(("ellipse", b, None, c, max(1.5, r * 0.12)))
            label.ops.append(_text_op("sans-bold", r * 1.0, str(rng.choice(list("ACEGKVX%"))), (cx, cy), c, anchor="mm"))
        elif shape == 1:
            label.ops.append(("rrect", b, c, r * 0.3))
            label.ops.append(("polygon", b, _regular(cx, cy, r * 0.55, 3, -math.pi / 2), ink_for(rng, c, palette, 0.0)))
        elif shape == 2:
            label.ops.append(("ellipse", (cx - r, cy - r * 0.5, cx + r, cy + r * 0.5), c, None, 0))
        else:
            label.ops.append(("polygon", b, _star(cx, cy, r, r * 0.5, 6), c))


def _overlaps(box: Box, taken: list[Box], pad: float = 0.0) -> bool:
    return any(box[0] < t[2] + pad and box[2] > t[0] - pad and box[1] < t[3] + pad and box[3] > t[1] - pad for t in taken)


def build_label(seed: int) -> Label:
    """The layout of label ``seed``. The same seed always gives the same label."""
    rng = np.random.default_rng([seed, 0])
    aspect = rng.uniform(0.65, 1.5)
    width, height = (CANVAS, round(CANVAS / aspect)) if aspect >= 1 else (round(CANVAS * aspect), CANVAS)
    mood, palette = make_palette(rng)
    flat = rng.random() < FLAT_SHARE
    bg = palette[0]
    if flat:
        background: tuple = ("flat", bg)
    else:
        pick = rng.choice(["flat", "gradient", "stripes", "dots", "checks"], p=[0.45, 0.25, 0.12, 0.1, 0.08])
        other = palette[1]
        if pick == "flat":
            background = ("flat", bg)
        elif pick == "gradient":
            background = ("gradient", bg, other, rng.uniform(0, 2 * math.pi))
        else:
            if rng.random() < 0.6:  # quiet patterns: the second colour close to the first
                other = tuple(int(np.clip(c + rng.integers(-28, 29), 0, 255)) for c in bg)
            period = float(rng.uniform(12, 90))
            angle = float(rng.choice([0.0, math.pi / 2, math.pi / 4, rng.uniform(0, math.pi)]))
            background = (str(pick), bg, other, period, angle)
    label = Label(seed, width, height, "flat" if flat else "normal", mood, background)
    w, h = float(width), float(height)
    m = min(w, h) * rng.uniform(0.04, 0.09)

    if flat:
        # Mostly flat colour: at most one small mark, often nothing at all.
        r = rng.random()
        under = bg
        if r < 0.35:
            name = str(rng.choice(TITLE_FONTS))
            text = title_text(rng)
            size = fit_size(name, text, w * 0.4, rng.uniform(30, 90))
            label.ops.append(_text_op(name, size, text, (rng.uniform(m, w * 0.5), rng.uniform(m, h - m - size)), ink_for(rng, under, palette)))
        elif r < 0.6:
            rad = min(w, h) * rng.uniform(0.04, 0.1)
            add_logo(label, rng, (rng.uniform(m + rad, w - m - rad), rng.uniform(m + rad, h - m - rad)), rad, palette, under)
        elif r < 0.75:
            add_text_block(label, rng, (m, h - m - h * 0.08, m + w * 0.35, h - m), under, palette, rng.uniform(9, 14), str(rng.choice(SMALL_FONTS)))
        return label

    density = rng.random()          # 0: nearly empty, 1: busy
    p = 0.15 + 0.75 * density       # chance of each optional element

    # Panels: bands, boxes, rounded boxes, circles. Text placed on one of them reads on its colour.
    panel_colors = palette[1:] or palette
    surfaces: list[tuple[Box, Color]] = [((0, 0, w, h), bg)]
    for _ in range(int(rng.integers(1, 5))):
        c = panel_colors[int(rng.integers(len(panel_colors)))]
        kind = rng.choice(["band", "box", "rbox", "circle", "slant"])
        if kind == "band":
            if rng.random() < 0.7:
                y0 = rng.uniform(0, h * 0.85)
                b = (0.0, y0, w, min(h, y0 + h * rng.uniform(0.08, 0.35)))
            else:
                x0 = rng.uniform(0, w * 0.85)
                b = (x0, 0.0, min(w, x0 + w * rng.uniform(0.08, 0.3)), h)
            label.ops.append(("rect", b, c, None, 0))
        elif kind in ("box", "rbox"):
            bw, bh = w * rng.uniform(0.25, 0.8), h * rng.uniform(0.15, 0.5)
            x0, y0 = rng.uniform(m, max(m, w - m - bw)), rng.uniform(m, max(m, h - m - bh))
            b = (x0, y0, x0 + bw, y0 + bh)
            if kind == "box":
                label.ops.append(("rect", b, c, None, 0))
            else:
                label.ops.append(("rrect", b, c, min(bw, bh) * rng.uniform(0.05, 0.35)))
        elif kind == "circle":
            r = min(w, h) * rng.uniform(0.12, 0.35)
            cx, cy = rng.uniform(r * 0.5, w - r * 0.5), rng.uniform(r * 0.5, h - r * 0.5)
            b = (cx - r, cy - r, cx + r, cy + r)
            label.ops.append(("ellipse", b, c, None, 0))
            b = (cx - r * 0.7, cy - r * 0.7, cx + r * 0.7, cy + r * 0.7)  # the part text can sit in
        else:
            y0, rise, thick = rng.uniform(0, h * 0.8), rng.uniform(-0.25, 0.25) * h, h * rng.uniform(0.08, 0.25)
            pts = [(0.0, y0), (w, y0 + rise), (w, y0 + rise + thick), (0.0, y0 + thick)]
            label.ops.append(("polygon", (0.0, min(y0, y0 + rise), w, max(y0, y0 + rise) + thick), pts, c))
            b = (w * 0.3, max(y0, y0 + rise), w * 0.7, min(y0, y0 + rise) + thick)
        if b[2] - b[0] > 40 and b[3] - b[1] > 30:
            surfaces.append((b, c))

    # Title, on the background or on a panel big enough.
    big = [s for s in surfaces[1:] if (s[0][2] - s[0][0]) > w * 0.35 and (s[0][3] - s[0][1]) > h * 0.1]
    (tb, under) = big[int(rng.integers(len(big)))] if big and rng.random() < 0.55 else surfaces[0]
    tl, tt, tr, tbm = tb
    name = str(rng.choice(TITLE_FONTS))
    text = title_text(rng)
    avail = (tr - tl) - 2 * m
    size = fit_size(name, text, max(60.0, avail * rng.uniform(0.6, 1.0)), rng.uniform(70, 260))
    size = min(size, (tbm - tt) * 0.8) if tb != surfaces[0][0] else size
    ty = rng.uniform(tt + m * 0.5, max(tt + m * 0.5, min(tbm - size * 1.2, tt + (tbm - tt) * 0.4)))
    align = rng.choice(["left", "centre"], p=[0.4, 0.6])
    title_ink = ink_for(rng, under, palette)
    if align == "centre":
        label.ops.append(_text_op(name, size, text, ((tl + tr) / 2, ty), title_ink, anchor="ma"))
    else:
        label.ops.append(_text_op(name, size, text, (tl + m, ty), title_ink))
    below = ty + size * 1.25
    taken: list[Box] = [label.ops[-1][1]]  # title, line, table and barcode: logos and icons keep off them

    # The smaller line under the title.
    if rng.random() < 0.85:
        lname = str(rng.choice(LINE_FONTS))
        ltext = line_text(rng)
        lsize = fit_size(lname, ltext, avail, size * rng.uniform(0.18, 0.4))
        x = (tl + tr) / 2 if align == "centre" else tl + m
        label.ops.append(_text_op(lname, lsize, ltext, (x, below), ink_for(rng, under, palette), anchor="ma" if align == "centre" else "la"))
        taken.append(label.ops[-1][1])
        below += lsize * 1.4

    # Logo.
    if rng.random() < 0.5 + 0.4 * density:
        rad = min(w, h) * rng.uniform(0.05, 0.14)
        spots = [(w - m - rad, m + rad), (m + rad, m + rad), (w / 2, min(h - m - rad, below + rad * 1.2)), (w - m - rad, h - m - rad)]
        for i in rng.permutation(len(spots)):
            cx, cy = spots[i]
            spot = (cx - rad, cy - rad, cx + rad, cy + rad)
            if not _overlaps(spot, taken, m * 0.3):
                add_logo(label, rng, (cx, cy), rad, palette, bg)
                taken.append(spot)
                break

    lower = (m, max(below, h * 0.55), w - m, h - m)
    if lower[3] - lower[1] < h * 0.15:
        lower = (m, h * 0.6, w - m, h - m)
    lx0, ly0, lx1, ly1 = lower
    col = lx0

    # Table like a nutrition panel.
    if rng.random() < 0.15 + 0.4 * density:
        tw = (lx1 - lx0) * rng.uniform(0.35, 0.55)
        add_table(label, rng, (col, ly0, col + tw, ly1), bg, palette)
        taken.append((col, ly0, col + tw, ly1))
        col += tw + m * 0.6

    # Barcode-like box.
    if rng.random() < 0.15 + 0.35 * density and lx1 - col > 120:
        bw = min(lx1 - col, w * rng.uniform(0.18, 0.3))
        bh = min(ly1 - ly0, bw * rng.uniform(0.5, 0.75))
        add_barcode(label, rng, (lx1 - bw, ly1 - bh, lx1, ly1))
        taken.append((lx1 - bw, ly1 - bh, lx1, ly1))
        fine_right = lx1 - bw - m * 0.6
    else:
        fine_right = lx1

    # Fine print in what is left of the lower part.
    if rng.random() < 0.3 + 0.65 * density and fine_right - col > 80:
        size = rng.uniform(9, 18)
        add_text_block(label, rng, (col, ly0, fine_right, ly1 - (ly1 - ly0) * rng.uniform(0, 0.4)), bg, palette, size, str(rng.choice(SMALL_FONTS)))

    # Small icons in a row.
    if rng.random() < 0.1 + 0.5 * density:
        isz = min(w, h) * rng.uniform(0.04, 0.08)
        span = isz * rng.integers(1, 5) * 1.3
        for _ in range(6):
            ix, iy = rng.uniform(m, max(m, w - m - span)), rng.uniform(h * 0.3, h - m - isz)
            row = (ix, iy, min(w - m, ix + span), iy + isz)
            if not _overlaps(row, taken, m * 0.2):
                add_icons(label, rng, row, palette, bg)
                break

    # Text rotated 90 degrees along a side edge.
    if rng.random() < 0.25:
        rname = str(rng.choice(SMALL_FONTS + LINE_FONTS))
        rtext = line_text(rng) if rng.random() < 0.6 else sentence(rng)
        rsize = fit_size(rname, rtext, h - 2 * m, rng.uniform(12, 40))
        x = m * 0.4 if rng.random() < 0.5 else w - m * 0.4 - rsize * 1.2
        label.ops.append(_text_op(rname, rsize, rtext, (x, h - m), ink_for(rng, bg, palette), rotate=90))

    # Border.
    if rng.random() < 0.3:
        bw = rng.uniform(3, 14)
        inset = rng.uniform(0, m * 0.6)
        c = ink_for(rng, bg, palette, 0.4)
        label.ops.append(("rect", (inset, inset, w - inset, h - inset), None, c, bw))
        if rng.random() < 0.3:
            label.ops.append(("rect", (inset + bw * 2.5, inset + bw * 2.5, w - inset - bw * 2.5, h - inset - bw * 2.5), None, c, bw * 0.4))
    return label


# --------------------------------------------------------------------------- drawing


def _background(label: Label, ox: float, oy: float, k: float, size: int) -> np.ndarray:
    """The background as (size, size, 3) uint8, for the window starting at (ox, oy), k pixels per unit."""
    kind = label.background[0]
    if kind == "flat":
        return np.broadcast_to(np.array(label.background[1], np.uint8), (size, size, 3)).copy()
    coords = (np.arange(size, dtype=np.float32) + 0.5) / k
    x = (ox + coords)[None, :]
    y = (oy + coords)[:, None]
    c1 = np.array(label.background[1], np.float32)
    c2 = np.array(label.background[2], np.float32)
    if kind == "gradient":
        angle = label.background[3]
        dx, dy = math.cos(angle), math.sin(angle)
        corners = [cx * dx + cy * dy for cx in (0, label.width) for cy in (0, label.height)]
        lo, hi = min(corners), max(corners)
        t = np.clip((x * dx + y * dy - lo) / (hi - lo), 0, 1)[..., None]
        return np.round(c1 + (c2 - c1) * t).astype(np.uint8)
    period, angle = label.background[3], label.background[4]
    dx, dy = math.cos(angle), math.sin(angle)
    if kind == "stripes":
        mask = ((x * dx + y * dy) % period) < period * 0.5
    elif kind == "checks":
        u, v = x * dx + y * dy, -x * dy + y * dx
        mask = ((np.floor(u / period) + np.floor(v / period)) % 2) == 0
    else:  # dots
        u, v = x * dx + y * dy, -x * dy + y * dx
        fu = (u % period) - period / 2
        fv = (v % period) - period / 2
        mask = fu**2 + fv**2 < (period * 0.25) ** 2
    out = np.where(mask[..., None], c2, c1)
    return np.round(out).astype(np.uint8)


def _draw(label: Label, ox: float, oy: float, k: float, size: int) -> Image.Image:
    """The window of the label starting at (ox, oy) with side size / k units, drawn size x size."""
    image = Image.fromarray(_background(label, ox, oy, k, size))
    draw = ImageDraw.Draw(image)
    view = (ox, oy, ox + size / k, oy + size / k)

    def tx(box: Box) -> list[float]:
        return [(box[0] - ox) * k, (box[1] - oy) * k, (box[2] - ox) * k, (box[3] - oy) * k]

    for op in label.ops:
        box = op[1]
        if box[2] < view[0] or box[0] > view[2] or box[3] < view[1] or box[1] > view[3]:
            continue
        kind = op[0]
        if kind == "rect":
            _, _, fill, outline, width = op
            b = tx(box)
            if b[2] - b[0] < 0.5 or b[3] - b[1] < 0.5:
                continue
            draw.rectangle(b, fill=fill, outline=outline, width=max(1, round(width * k)) if outline else 0)
        elif kind == "rrect":
            _, _, fill, radius = op
            b = tx(box)
            if b[2] - b[0] < 1 or b[3] - b[1] < 1:
                continue
            draw.rounded_rectangle(b, radius=min(radius * k, (b[2] - b[0]) / 2, (b[3] - b[1]) / 2), fill=fill)
        elif kind == "ellipse":
            _, _, fill, outline, width = op
            b = tx(box)
            if b[2] - b[0] < 1 or b[3] - b[1] < 1:
                continue
            draw.ellipse(b, fill=fill, outline=outline, width=max(1, round(width * k)) if outline else 0)
        elif kind == "polygon":
            _, _, pts, fill = op
            draw.polygon([((px - ox) * k, (py - oy) * k) for px, py in pts], fill=fill)
        elif kind == "text":
            _, _, name, fsize, text, (x, y), fill, anchor, rotate = op
            px = fsize * k
            if px < 1.0:
                continue
            f = font(name, px)
            if not rotate:
                draw.text(((x - ox) * k, (y - oy) * k), text, font=f, fill=fill, anchor=anchor)
            else:
                left, top, right, bottom = f.getbbox(text, anchor=anchor)
                w, h = int(math.ceil(right - left)) + 2, int(math.ceil(bottom - top)) + 2
                if w * h > 64_000_000:
                    continue
                mask = Image.new("L", (w, h), 0)
                ImageDraw.Draw(mask).text((-left + 1, -top + 1), text, font=f, fill=255, anchor=anchor)
                mask = mask.transpose(Image.Transpose.ROTATE_90)
                # The unrotated point (left, top) goes to (top, -right) about the anchor.
                image.paste(fill, (round((x - ox) * k + top - 1), round((y - oy) * k - right - 1)), mask)
    return image


def render_window(label: Label, box: tuple[float, float, float], out_px: int, supersample: int = SUPERSAMPLE) -> np.ndarray:
    """The square (left, top, side) of the label in layout units, as (out_px, out_px, 3) uint8."""
    left, top, side = box
    size = out_px * supersample
    image = _draw(label, left, top, size / side, size)
    if supersample > 1:
        image = image.reduce(supersample)  # mean of each supersample x supersample block
    return np.asarray(image, dtype=np.uint8)


def render_label(label: Label, long_side: int = 512) -> np.ndarray:
    """The whole label, long side ``long_side`` px, as (H, W, 3) uint8."""
    k = long_side / max(label.width, label.height)
    w, h = round(label.width * k), round(label.height * k)
    side = max(w, h)
    full = render_window(label, (0.0, 0.0, max(label.width, label.height)), side)
    return full[:h, :w]


def crop_box(label: Label, seed: int) -> tuple[float, float, float]:
    """(left, top, side) of the tile of ``seed``: side log-uniform from MIN_CROP to the label's short side."""
    rng = np.random.default_rng([seed, 1])
    short = min(label.width, label.height)
    side = float(math.exp(rng.uniform(math.log(MIN_CROP), math.log(short))))
    return float(rng.uniform(0, label.width - side)), float(rng.uniform(0, label.height - side)), side


def render_tile(seed: int, tile_px: int = TILE_PX) -> np.ndarray:
    """Tile ``seed``: a square of label ``seed`` at a random zoom, (tile_px, tile_px, 3) uint8."""
    label = build_label(seed)
    return render_window(label, crop_box(label, seed), tile_px)


# --------------------------------------------------------------------------- command line


def sheet(images: list[np.ndarray], columns: int = 8, gap: int = 6, cell: int | None = None) -> Image.Image:
    cell = cell or max(max(i.shape[:2]) for i in images)
    rows = -(-len(images) // columns)
    out = Image.new("RGB", (columns * cell + (columns - 1) * gap, rows * cell + (rows - 1) * gap), (128, 128, 128))
    for n, img in enumerate(images):
        h, w = img.shape[:2]
        x = (n % columns) * (cell + gap) + (cell - w) // 2
        y = (n // columns) * (cell + gap) + (cell - h) // 2
        out.paste(Image.fromarray(img), (x, y))
    return out


def _bench_worker(seeds: list[int]) -> float:
    start = time.perf_counter()
    for s in seeds:
        render_tile(s)
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", help="PNG of the tiles of seeds --first to --first + 63, 8 x 8")
    parser.add_argument("--labels-sheet", help="PNG of the same seeds' whole labels, 8 x 8")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--bench", type=int, default=0, help="tiles per worker to time")
    parser.add_argument("--workers", type=int, nargs="+", default=[1], help="process counts to time with --bench")
    args = parser.parse_args()

    seeds = list(range(args.first, args.first + args.count))
    if args.sheet:
        path = Path(args.sheet)
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet([render_tile(s) for s in seeds]).save(path)
        print(f"tiles of seeds {seeds[0]} to {seeds[-1]}: {path}")
    if args.labels_sheet:
        path = Path(args.labels_sheet)
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet([render_label(build_label(s), 384) for s in seeds], cell=384).save(path)
        print(f"whole labels of seeds {seeds[0]} to {seeds[-1]}: {path}")
    if args.bench:
        from multiprocessing import Pool

        _bench_worker(list(range(10_000_000, 10_000_005)))  # fonts loaded, caches warm
        for workers in args.workers:
            chunks = [list(range(1_000_000 * (i + 1), 1_000_000 * (i + 1) + args.bench)) for i in range(workers)]
            if workers == 1:
                start = time.perf_counter()
                _bench_worker(chunks[0])
                wall = time.perf_counter() - start
            else:
                with Pool(workers) as pool:
                    pool.map(_bench_worker, [list(range(20_000_000 + i * 10, 20_000_000 + i * 10 + 3)) for i in range(workers)])
                    start = time.perf_counter()
                    pool.map(_bench_worker, chunks)
                    wall = time.perf_counter() - start
            total = workers * args.bench
            print(f"{workers} worker(s): {total} tiles in {wall:.1f} s = {total / wall:.1f} tiles/s ({total / wall / workers:.1f} per worker)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
