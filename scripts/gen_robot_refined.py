"""Refined protein-cartoon wrap: PyMOL-style helix ribbon + a beta sheet.

* Helices: a fat, gently waving ribbon (small amplitude, large width) with
  evenly spaced diagonal turn ticks -- the PyMOL helix-rod look.
* Beta sheet: a flat ribbon ending in an arrowhead.
* Loops: thin smooth tubes.
Backbone stays on a clean, symmetric ellipse/circle, centred in the tile.
Output under ``assets/readme/logo-concepts/robot-refined/``.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "assets" / "readme" / "logo-concepts" / "robot-refined"
OUT.mkdir(parents=True, exist_ok=True)

CX, CY = 128, 128
LOOP_COL, HELIX_COL, TICK_COL = "#b29ae6", "#d8cbf6", "#9b81d4"
THETA0, THETA1 = 120.0, 420.0


def ept(theta: float, rx: float, ry: float) -> tuple[float, float]:
    t = math.radians(theta)
    return CX + rx * math.cos(t), CY + ry * math.sin(t)


def radial(p: tuple[float, float]) -> tuple[float, float]:
    dx, dy = p[0] - CX, p[1] - CY
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d


def centerline(ta, tb, rx, ry, n, amp=0.0, periods=1.5):
    pts = []
    for k in range(n + 1):
        u = k / n
        base = ept(ta + (tb - ta) * u, rx, ry)
        if amp:
            ux, uy = radial(base)
            off = amp * math.cos(periods * math.pi * (2 * u - 1))
            pts.append((base[0] + ux * off, base[1] + uy * off))
        else:
            pts.append(base)
    return pts


def normals(pts):
    out, n = [], len(pts)
    for i in range(n):
        a, b = pts[max(0, i - 1)], pts[min(n - 1, i + 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        L = math.hypot(tx, ty) or 1.0
        out.append((-ty / L, tx / L))
    return out


def poly(pts):
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)


def loop_path(ta, tb, rx, ry):
    return poly(centerline(ta, tb, rx, ry, 16))


def helix(ta, tb, rx, ry):
    """A real coiled ribbon: each turn's front passes over the turn behind,
    shaded light (front) over dark (back) -- a PyMOL-style helix, not a worm."""
    span = abs(tb - ta)
    k = max(3, round(span / 15.0))          # number of turns
    amp = 12.0                              # coil radius
    n = 64 * k
    samples = []
    for i in range(n + 1):
        u = i / n
        base = ept(ta + (tb - ta) * u, rx, ry)
        ux, uy = radial(base)
        phi = k * 2.0 * math.pi * u
        off = amp * math.sin(phi)
        samples.append(((base[0] + ux * off, base[1] + uy * off), math.cos(phi)))
    front, back, run = [], [], [samples[0][0]]
    pos = samples[0][1] >= 0
    for p, c in samples[1:]:
        run.append(p)
        if (c >= 0) != pos:
            (front if pos else back).append(run)
            run, pos = [p], c >= 0
    (front if pos else back).append(run)
    out = [f'<path d="{poly(r)}" fill="none" stroke="{TICK_COL}" stroke-width="12" '
           f'stroke-linecap="round" stroke-linejoin="round"/>' for r in back]
    out += [f'<path d="{poly(r)}" fill="none" stroke="{HELIX_COL}" stroke-width="15" '
            f'stroke-linecap="round" stroke-linejoin="round"/>' for r in front]
    return "".join(out)


def beta(ta, tb, rx, ry):
    n = 24
    cl = centerline(ta, tb, rx, ry, n)
    nm = normals(cl)
    ih = 15
    body, head = 9.0, 16.0
    left, right = [], []
    for i in range(ih + 1):
        left.append((cl[i][0] + nm[i][0] * body, cl[i][1] + nm[i][1] * body))
        right.append((cl[i][0] - nm[i][0] * body, cl[i][1] - nm[i][1] * body))
    left.append((cl[ih][0] + nm[ih][0] * head, cl[ih][1] + nm[ih][1] * head))
    right.append((cl[ih][0] - nm[ih][0] * head, cl[ih][1] - nm[ih][1] * head))
    for i in range(ih + 1, n + 1):
        hw = head * (n - i) / (n - ih)
        left.append((cl[i][0] + nm[i][0] * hw, cl[i][1] + nm[i][1] * hw))
        right.append((cl[i][0] - nm[i][0] * hw, cl[i][1] - nm[i][1] * hw))
    return f'<path d="{poly(left + right[::-1])} Z" fill="{HELIX_COL}"/>'


TILE = """
  <defs>
    <linearGradient id="tile" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#3a0d6e"/><stop offset="1" stop-color="#6b1ea8"/>
    </linearGradient>
    <radialGradient id="glow" cx="0.3" cy="0.24" r="0.95">
      <stop offset="0" stop-color="#ffffff" stop-opacity="0.18"/>
      <stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="256" height="256" rx="58" fill="url(#tile)"/>
  <rect width="256" height="256" rx="58" fill="url(#glow)"/>
"""

ROBOT_HEAD = """
  <g transform="translate(0 -6)">
  <rect x="77" y="124" width="9" height="24" rx="4" fill="#b9a5f0"/>
  <rect x="170" y="124" width="9" height="24" rx="4" fill="#b9a5f0"/>
  <line x1="128" y1="74" x2="128" y2="100" stroke="#d7ccf5"
        stroke-width="5" stroke-linecap="round"/>
  <circle cx="128" cy="69" r="7" fill="#4cc4e4"/>
  <circle cx="128" cy="69" r="2.6" fill="#ffffff"/>
  <rect x="85" y="94" width="86" height="80" rx="20" fill="#ffffff"/>
  <circle cx="112" cy="132" r="10" fill="#4cc4e4"/>
  <circle cx="144" cy="132" r="10" fill="#4cc4e4"/>
  <circle cx="108.5" cy="128.5" r="3.4" fill="#ffffff"/>
  <circle cx="140.5" cy="128.5" r="3.4" fill="#ffffff"/>
  <path d="M113 153 Q128 165 143 153" fill="none" stroke="#6b1ea8"
        stroke-width="5" stroke-linecap="round"/>
  </g>
"""


def build(seg_types, rx, ry):
    n = len(seg_types)
    th = [THETA0 + (THETA1 - THETA0) * i / n for i in range(n + 1)]
    loops, structures = [], []
    for i, kind in enumerate(seg_types):
        a, b = th[i], th[i + 1]
        if kind == "loop":
            loops.append(f'<path d="{loop_path(a, b, rx, ry)}"/>')
        elif kind == "helix":
            structures.append(helix(a, b, rx, ry))
        else:
            structures.append(beta(a, b, rx, ry))
    body = [f'<g fill="none" stroke="{LOOP_COL}" stroke-width="7" '
            f'stroke-linecap="round">{"".join(loops)}</g>']
    body.extend(structures)
    for theta, col in ((th[0], "#4cc4e4"), (th[-1], "#ff6aa6")):
        x, y = ept(theta, rx, ry)
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{col}"/>'
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="#ffffff"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" '
            f'viewBox="0 0 256 256">{TILE}{"".join(body)}{ROBOT_HEAD}</svg>\n')


H3 = ["loop", "helix", "loop", "helix", "loop", "helix", "loop"]
HB = ["loop", "helix", "loop", "beta", "loop"]

CONCEPTS = [
    ("r1-ellipse", "Ellipse · 3 helices", "Coiled-ribbon helices, front over back",
     lambda: build(H3, 94, 82)),
    ("r2-circle", "Circle · 3 helices", "Same, on a perfect circle",
     lambda: build(H3, 88, 88)),
    ("r3-helix-sheet", "Helix + beta sheet", "S3 with one helix swapped for a sheet",
     lambda: build(HB, 94, 82)),
]

renders: dict[str, Image.Image] = {}
for slug, name, _desc, fn in CONCEPTS:
    src = fn()
    (OUT / f"{slug}.svg").write_text(src, encoding="utf-8")
    png = cairosvg.svg2png(bytestring=src.encode(), output_width=600, output_height=600)
    (OUT / f"{slug}.png").write_bytes(png)
    renders[slug] = Image.open(io.BytesIO(png)).convert("RGBA")
    print(f"  rendered {slug}")

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
f_head = ImageFont.truetype(FB, 35)
f_name = ImageFont.truetype(FB, 26)
f_desc = ImageFont.truetype(FR, 19)

MARGIN, T, GAP = 44, 360, 36
n = len(CONCEPTS)
W = MARGIN * 2 + n * T + (n - 1) * GAP
H = MARGIN * 2 + 78 + T + 82
sheet = Image.new("RGB", (W, H), "#e9e7ef")
d = ImageDraw.Draw(sheet)
d.text((MARGIN, MARGIN), "Coiled-ribbon helix + a beta sheet", font=f_head, fill="#1a1535")
d.text((MARGIN, MARGIN + 44),
       "PyMOL-style helix: distinct turns, front passes over back -- not a worm.",
       font=f_desc, fill="#6b6491")
for i, (slug, name, desc, _fn) in enumerate(CONCEPTS):
    cx = MARGIN + i * (T + GAP)
    cy = MARGIN + 78
    tile = renders[slug].resize((T, T), Image.LANCZOS)
    sheet.paste(tile, (cx, cy), tile)
    d.text((cx + 4, cy + T + 14), name, font=f_name, fill="#1a1535")
    d.text((cx + 4, cy + T + 46), desc, font=f_desc, fill="#6b6491")
sheet.save(OUT / "_contact-sheet.png")
print(f"\nContact sheet -> {OUT / '_contact-sheet.png'}  ({W}x{H})")
