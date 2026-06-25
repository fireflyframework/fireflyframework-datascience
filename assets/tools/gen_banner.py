# Copyright 2026 Firefly Software Foundation.
"""Generate the Firefly DataScience brand banner, nav logo and favicon as self-contained SVGs.

House style (matches the agentic/pyfly/rust ecosystem): **no external fonts**. The "Maven Pro"
wordmark is therefore converted to real Maven Pro *vector paths* — HarfBuzz (``uharfbuzz``) shapes
each string (applying the font's GPOS kerning) and fontTools extracts the glyph outlines. The
resulting SVGs render identically on GitHub ``<img>``, in MkDocs, and in WeasyPrint PDFs with zero
font dependency. Teal brand palette; amber shared-family firefly motif.

Build-time tooling only (not a runtime/docs dependency). Regenerate with::

    uv pip install uharfbuzz cairosvg     # one-off, into your working venv
    python assets/tools/gen_banner.py     # -> docs/img/{banner,logo,favicon}.svg

Maven Pro TTFs are read from ``$MAVEN_PRO_DIR`` or ``~/Library/Fonts`` (weights 400/500/600/700).
Generated SVGs are committed, so the published build never needs the font or these tools.
"""

from __future__ import annotations

import os
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

# --- Teal brand palette (the DataScience member colour) + shared amber firefly. -----------------
SKY1, SKY2, SKY3 = "#06121a", "#0a1b22", "#071419"
CYAN = "#06b6d4"
CYAN_L = "#67e8f9"
CYAN_XL = "#d6fbff"
CYAN_D = "#0e7d97"
EYEBROW = "#5fb3c6"
TAG = "#dff6fb"
SUBTAG = "#7fa9b1"
LINE = "#3fb6cf"
AMBER = "#f6a821"
AMBER_HI = "#ffd980"

_FONT_DIR = Path(os.environ.get("MAVEN_PRO_DIR", str(Path.home() / "Library" / "Fonts")))
FONTS = {w: _FONT_DIR / f"MavenPro-{w}.ttf" for w in (400, 500, 600, 700)}

_OUT = Path(__file__).resolve().parents[2] / "docs" / "img"

# Cache of loaded fonttools/HarfBuzz objects per weight.
_cache: dict[int, tuple] = {}


def _load(weight: int):
    if weight not in _cache:
        path = str(FONTS[weight])
        tt = TTFont(path)
        blob = hb.Blob.from_file_path(path)
        face = hb.Face(blob)
        font = hb.Font(face)  # default scale == upem, so advances/offsets are in font units
        _cache[weight] = (tt, font, tt["head"].unitsPerEm, tt.getGlyphOrder(), tt.getGlyphSet())
    return _cache[weight]


def wordmark(text: str, weight: int, size: float, tracking: float = 0.0, fill: str = "#000",
             baseline_xy: tuple[float, float] = (0.0, 0.0)) -> tuple[str, float, float]:
    """Return ``(svg_group, width_px, cap_height_px)`` placing ``text`` in Maven Pro.

    The group's baseline origin sits at ``baseline_xy`` (SVG coords, y-down). ``tracking`` is extra
    letter-spacing in px (like CSS ``letter-spacing``). Output is pure ``<path>`` data — no fonts.
    """
    tt, font, upem, order, glyphs = _load(weight)
    scale = size / upem
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True, "liga": True})

    tracking_units = tracking / scale
    bx, by = baseline_xy
    parts: list[str] = []
    x = 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        pen = SVGPathPen(glyphs)
        glyphs[order[info.codepoint]].draw(pen)
        d = pen.getCommands()
        if d:
            gx = bx + (x + pos.x_offset) * scale
            gy = by - pos.y_offset * scale
            parts.append(f'<path transform="translate({gx:.2f},{gy:.2f}) scale({scale:.5f},{-scale:.5f})" d="{d}"/>')
        x += pos.x_advance + tracking_units
    width = (x - tracking_units) * scale
    cap = getattr(tt["OS/2"], "sCapHeight", 700) * scale
    return f'<g fill="{fill}">{"".join(parts)}</g>', width, cap


# --- Reusable SVG fragments. --------------------------------------------------------------------
def _defs() -> str:
    return f"""<defs>
  <linearGradient id="sky" x1="0" y1="0" x2="1280" y2="320" gradientUnits="userSpaceOnUse">
    <stop offset="0" stop-color="{SKY1}"/><stop offset="0.5" stop-color="{SKY2}"/><stop offset="1" stop-color="{SKY3}"/>
  </linearGradient>
  <radialGradient id="glowA" cx="30%" cy="26%" r="62%">
    <stop offset="0" stop-color="{CYAN}" stop-opacity="0.26"/><stop offset="1" stop-color="{CYAN}" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="glowB" cx="88%" cy="80%" r="58%">
    <stop offset="0" stop-color="#22d3ee" stop-opacity="0.18"/><stop offset="1" stop-color="#22d3ee" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="wordmark" x1="82" y1="120" x2="720" y2="205" gradientUnits="userSpaceOnUse">
    <stop offset="0" stop-color="{CYAN_XL}"/><stop offset="0.42" stop-color="#22d3ee"/><stop offset="1" stop-color="{CYAN_D}"/>
  </linearGradient>
  <linearGradient id="rule" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{CYAN_L}"/><stop offset="1" stop-color="#22d3ee" stop-opacity="0.12"/>
  </linearGradient>
  <radialGradient id="spark" cx="50%" cy="50%" r="50%">
    <stop offset="0" stop-color="#FFF9C1"/><stop offset="1" stop-color="#F68000"/>
  </radialGradient>
  <pattern id="grid" width="34" height="34" patternUnits="userSpaceOnUse">
    <circle cx="1" cy="1" r="1" fill="#3aa6bd" opacity="0.10"/>
  </pattern>
  <g id="node"><circle r="11" fill="#22d3ee" opacity="0.12"/><circle r="6" fill="{CYAN_L}" opacity="0.28"/><circle r="3" fill="{CYAN_XL}"/></g>
  <g id="fly"><circle r="14" fill="{AMBER}" opacity="0.10"/><circle r="8" fill="#ffc24a" opacity="0.22"/><circle r="3.6" fill="{AMBER_HI}" opacity="0.75"/><circle r="1.8" fill="#fff7e8"/></g>
</defs>"""


def _constellation() -> str:
    """Right-side data constellation with the amber firefly as the hub (Candidate A)."""
    return f"""<g stroke="{LINE}" stroke-width="1.1" fill="none" opacity="0.42">
    <path d="M948 92 L1052 138"/><path d="M1052 138 L988 214"/><path d="M1052 138 L1148 116"/>
    <path d="M1052 138 L1158 206"/><path d="M988 214 L1108 250"/><path d="M948 92 L1148 116"/>
  </g>
  <use href="#node" transform="translate(948,92)"/>
  <use href="#fly"  transform="translate(1052,138) scale(1.18)"/>
  <use href="#node" transform="translate(988,214)"/>
  <use href="#node" transform="translate(1148,116) scale(0.85)"/>
  <use href="#node" transform="translate(1158,206) scale(0.8)"/>
  <use href="#node" transform="translate(1108,250) scale(0.7)"/>"""


# --- Asset builders. ----------------------------------------------------------------------------
def build_banner() -> str:
    eyebrow, _, _ = wordmark("THE FIREFLY FRAMEWORK", 700, 13, 3.0, EYEBROW, (88, 88))
    mark, _, _ = wordmark("Firefly DataScience", 700, 76, -1.0, "url(#wordmark)", (82, 176))
    tagline, _, _ = wordmark("AutoML that fuses GenAI with classical ML & Deep Learning.", 600, 21.5, 0.1, TAG, (86, 248))
    sub, _, _ = wordmark("Hexagonal · secure-by-default · governed GenAI · built on Firefly Agentic", 400, 15, 0.1, SUBTAG, (86, 280))
    aria = ("Firefly DataScience — AutoML that fuses GenAI with classical ML and Deep Learning, "
            "built on the Firefly Framework")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="320" viewBox="0 0 1280 320" role="img" aria-label="{aria}">
{_defs()}
  <rect width="1280" height="320" fill="url(#sky)"/>
  <rect width="1280" height="320" fill="url(#grid)"/>
  <rect width="1280" height="320" fill="url(#glowA)"/>
  <rect width="1280" height="320" fill="url(#glowB)"/>
  {_constellation()}
  <circle cx="556" cy="118" r="5" fill="url(#spark)" opacity="0.9"/>
  {eyebrow}
  {mark}
  <rect x="86" y="205" width="346" height="2.4" rx="1.2" fill="url(#rule)"/>
  {tagline}
  {sub}
</svg>"""


def _glyph(cx: float, cy: float, s: float) -> str:
    """Compact firefly-hub mark: amber firefly + 3 cyan nodes + edges. ``s`` = radius of node ring."""
    import math

    nodes = [(cx + s * math.cos(a), cy + s * math.sin(a)) for a in (-2.5, -0.4, 1.4)]
    edges = "".join(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{LINE}" stroke-width="{s*0.07:.2f}" opacity="0.55"/>'
                    for nx, ny in nodes)
    dots = "".join(f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="{s*0.16:.2f}" fill="{CYAN_L}"/>' for nx, ny in nodes)
    r = s * 0.42
    fly = (f'<circle cx="{cx}" cy="{cy}" r="{r*1.7:.2f}" fill="{AMBER}" opacity="0.18"/>'
           f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="#ffc24a"/>'
           f'<circle cx="{cx}" cy="{cy}" r="{r*0.5:.2f}" fill="#fff7e8"/>')
    return edges + dots + fly


def build_logo() -> str:
    """Transparent firefly-hub icon for the (dark) Material header, sits left of the site title."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40" role="img" aria-label="Firefly DataScience">
  {_glyph(20, 20, 13)}
</svg>"""


def build_favicon() -> str:
    """Square firefly-hub glyph on a dark rounded tile; legible at 16px."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64" role="img" aria-label="Firefly DataScience">
  <rect width="64" height="64" rx="14" fill="{SKY2}"/>
  <rect width="64" height="64" rx="14" fill="url(#tile)"/>
  <defs><radialGradient id="tile" cx="40%" cy="34%" r="65%">
    <stop offset="0" stop-color="{CYAN}" stop-opacity="0.22"/><stop offset="1" stop-color="{CYAN}" stop-opacity="0"/>
  </radialGradient></defs>
  {_glyph(32, 32, 18)}
</svg>"""


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    for name, svg in (("banner", build_banner()), ("logo", build_logo()), ("favicon", build_favicon())):
        path = _OUT / f"{name}.svg"
        path.write_text(svg)
        print(f"wrote {path.relative_to(_OUT.parents[1])}  ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
