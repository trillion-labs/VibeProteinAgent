"""Robot wrapped in a protein cartoon on a clean, symmetric circle/ellipse.

The backbone centreline lies exactly on a smooth ellipse; helical spans
oscillate purely radially around it, so the overall wrap stays a tidy,
left-right symmetric ring. Output under
``assets/readme/logo-concepts/robot-symmetric/``.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "assets" / "readme" / "logo-concepts" / "robot-symmetric"
OUT.mkdir(parents=True, exist_ok=True)

CX, CY = 128, 128          # tile centre — composition is centred here
LOOP_COL, HELIX_COL = "#b29ae6", "#d8cbf6"
THETA0, THETA1 = 120.0, 420.0          # 60-degree gap centred on the bottom


def ept(theta: float, rx: float, ry: float) -> tuple[float, float]:
    t = math.radians(theta)
    return CX + rx * math.cos(t), CY + ry * math.sin(t)


def radial(p: tuple[float, float]) -> tuple[float, float]:
    dx, dy = p[0] - CX, p[1] - CY
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d


def loop_seg(ta: float, tb: float, rx: float, ry: float) -> str:
    pts = [ept(ta + (tb - ta) * k / 16, rx, ry) for k in range(17)]
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)


def helix_seg(ta: float, tb: float, rx: float, ry: float, amp: float,
              periods: float = 2.5):
    pts = []
    for k in range(41):
        u = k / 40
        base = ept(ta + (tb - ta) * u, rx, ry)
        ux, uy = radial(base)
        off = amp * math.cos(periods * math.pi * (2 * u - 1))
        pts.append((base[0] + ux * off, base[1] + uy * off))
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    ticks = []
    for k in (-2, -1, 0, 1, 2):
        u = (1 + k / periods) / 2
        base = ept(ta + (tb - ta) * u, rx, ry)
        ux, uy = radial(base)
        off = amp * math.cos(periods * math.pi * (2 * u - 1))
        px, py = base[0] + ux * off, base[1] + uy * off
        ticks.append(
            f'<line x1="{px - ux * 10:.1f}" y1="{py - uy * 10:.1f}" '
            f'x2="{px + ux * 10:.1f}" y2="{py + uy * 10:.1f}" '
            f'stroke="{LOOP_COL}" stroke-width="4" stroke-linecap="round"/>'
        )
    return d, "".join(ticks)


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
"""


def build(seg_types: list[str], rx: float, ry: float, amp: float = 8.0) -> str:
    n = len(seg_types)
    thetas = [THETA0 + (THETA1 - THETA0) * i / n for i in range(n + 1)]
    loops, helices, ticks = [], [], []
    for i, kind in enumerate(seg_types):
        ta, tb = thetas[i], thetas[i + 1]
        if kind == "loop":
            loops.append(f'<path d="{loop_seg(ta, tb, rx, ry)}"/>')
        else:
            d, tk = helix_seg(ta, tb, rx, ry, amp)
            helices.append(d)
            ticks.append(tk)
    body = [
        f'<g fill="none" stroke="{LOOP_COL}" stroke-width="7" '
        f'stroke-linecap="round">{"".join(loops)}</g>'
    ]
    for d in helices:
        body.append(f'<path d="{d}" fill="none" stroke="{HELIX_COL}" '
                    f'stroke-width="15" stroke-linecap="round" stroke-linejoin="round"/>')
    body.append("".join(ticks))
    for theta, col in ((thetas[0], "#4cc4e4"), (thetas[-1], "#ff6aa6")):
        x, y = ept(theta, rx, ry)
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{col}"/>'
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="#ffffff"/>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" '
        f'viewBox="0 0 256 256">{TILE}{"".join(body)}'
        f'<g transform="translate(0 -6)">{ROBOT_HEAD}</g></svg>\n'
    )


HELIX3 = ["loop", "helix", "loop", "helix", "loop", "helix", "loop"]
HELIX2 = ["loop", "helix", "loop", "helix", "loop"]

CONCEPTS = [
    ("s1-ellipse", "Ellipse · 3 helices", "Smooth ellipse, fully symmetric",
     lambda: build(HELIX3, 96, 84)),
    ("s2-circle", "Circle · 3 helices", "Perfect circle wrap",
     lambda: build(HELIX3, 88, 88)),
    ("s3-calm", "Ellipse · 2 helices", "Calmer — two helices, longer loops",
     lambda: build(HELIX2, 96, 84, amp=9)),
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
f_head = ImageFont.truetype(FB, 36)
f_name = ImageFont.truetype(FB, 26)
f_desc = ImageFont.truetype(FR, 19)

MARGIN, T, GAP = 44, 360, 36
n = len(CONCEPTS)
W = MARGIN * 2 + n * T + (n - 1) * GAP
H = MARGIN * 2 + 78 + T + 82
sheet = Image.new("RGB", (W, H), "#e9e7ef")
d = ImageDraw.Draw(sheet)
d.text((MARGIN, MARGIN), "Symmetric protein-cartoon wrap", font=f_head, fill="#1a1535")
d.text((MARGIN, MARGIN + 44),
       "Backbone on a clean ellipse; helices oscillate radially, left-right symmetric.",
       font=f_desc, fill="#6b6491")
for i, (slug, name, desc, _fn) in enumerate(CONCEPTS):
    cx = MARGIN + i * (T + GAP)
    cy = MARGIN + 78
    sheet.paste(renders[slug].resize((T, T), Image.LANCZOS), (cx, cy), renders[slug].resize((T, T), Image.LANCZOS))
    d.text((cx + 4, cy + T + 14), name, font=f_name, fill="#1a1535")
    d.text((cx + 4, cy + T + 46), desc, font=f_desc, fill="#6b6491")
sheet.save(OUT / "_contact-sheet.png")
print(f"\nContact sheet -> {OUT / '_contact-sheet.png'}  ({W}x{H})")
