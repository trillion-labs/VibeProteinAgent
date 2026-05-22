"""Wrap the robot in a protein *cartoon*: helices + connecting loops.

The orbit is replaced by one continuous backbone that wraps the robot once
-- not a ring, but an irregular chain alternating helical segments (coiled)
with random-coil loops, the way a real protein structure is drawn. Three
helix renderings are compared. Output under
``assets/readme/logo-concepts/robot-cartoon/``.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "assets" / "readme" / "logo-concepts" / "robot-cartoon"
OUT.mkdir(parents=True, exist_ok=True)

CX, CY = 128, 134
LOOP_COL, HELIX_COL, SHEEN = "#b29ae6", "#d8cbf6", "#efe9fc"

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

# anchor points of the backbone -- an irregular wrap, gap at the bottom
RX, RY, PHI = 96, 67, math.radians(-15)
RMUL = [1.00, 1.07, 0.96, 1.05, 0.97, 1.06, 1.00]


def anchors() -> list[tuple[float, float]]:
    pts = []
    t0, t1 = math.pi / 2 + 0.52, math.pi / 2 + 2 * math.pi - 0.52
    for i, m in enumerate(RMUL):
        t = t0 + (t1 - t0) * i / (len(RMUL) - 1)
        ex, ey = RX * m * math.cos(t), RY * m * math.sin(t)
        pts.append((CX + ex * math.cos(PHI) - ey * math.sin(PHI),
                    CY + ex * math.sin(PHI) + ey * math.cos(PHI)))
    return pts


def _frame(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1
    return dx, dy, L, dx / L, dy / L, -dy / L, dx / L  # dx,dy,L,ux,uy,px,py


def loop_path(a, b) -> str:
    dx, dy, L, ux, uy, px, py = _frame(a, b)
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    if (mx - CX) * px + (my - CY) * py < 0:   # bulge away from the robot
        px, py = -px, -py
    cx, cy = mx + px * 16, my + py * 16
    return f"M{a[0]:.1f} {a[1]:.1f} Q{cx:.1f} {cy:.1f} {b[0]:.1f} {b[1]:.1f}"


def helix_wave(a, b, amp: float, periods: float, n: int = 30) -> str:
    dx, dy, L, ux, uy, px, py = _frame(a, b)
    pts = []
    for k in range(n + 1):
        u = k / n
        off = amp * math.sin(u * periods * 2 * math.pi)
        pts.append((a[0] + dx * u + px * off, a[1] + dy * u + py * off))
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)


def helix_ticks(a, b, periods: float, length: float) -> str:
    dx, dy, L, ux, uy, px, py = _frame(a, b)
    ticks = []
    half = periods * 2
    for k in range(int(half)):
        u = (k + 0.5) / half
        cx = a[0] + dx * u
        cy = a[1] + dy * u
        ticks.append(
            f'<line x1="{cx - px * length:.1f}" y1="{cy - py * length:.1f}" '
            f'x2="{cx + px * length:.1f}" y2="{cy + py * length:.1f}" '
            f'stroke="{LOOP_COL}" stroke-width="4" stroke-linecap="round"/>'
        )
    return "".join(ticks)


# segment plan: which spans are helices, which are loops
SEGMENTS = ["loop", "helix", "loop", "helix", "loop", "helix"]


def backbone(style: str) -> str:
    pts = anchors()
    loops, helices, extra = [], [], []
    for i, kind in enumerate(SEGMENTS):
        a, b = pts[i], pts[i + 1]
        if kind == "loop":
            loops.append(f'<path d="{loop_path(a, b)}"/>')
            continue
        if style == "coil":
            d = helix_wave(a, b, amp=11, periods=3.5)
        else:
            d = helix_wave(a, b, amp=9, periods=2.5)
        helices.append(d)
        if style == "cartoon":
            extra.append(helix_ticks(a, b, 2.5, 11))
    out = [
        f'<g fill="none" stroke="{LOOP_COL}" stroke-width="7" '
        f'stroke-linecap="round">{"".join(loops)}</g>'
    ]
    hw = 13 if style == "coil" else 15
    for d in helices:
        out.append(f'<path d="{d}" fill="none" stroke="{HELIX_COL}" '
                   f'stroke-width="{hw}" stroke-linecap="round" stroke-linejoin="round"/>')
    if style == "ribbon":
        for d in helices:
            out.append(f'<path d="{d}" fill="none" stroke="{SHEEN}" '
                       f'stroke-width="4.5" stroke-linecap="round"/>')
    out.extend(extra)
    # N / C termini
    n, c = pts[0], pts[-1]
    out.append(f'<circle cx="{n[0]:.1f}" cy="{n[1]:.1f}" r="9" fill="#4cc4e4"/>'
               f'<circle cx="{n[0]:.1f}" cy="{n[1]:.1f}" r="3.4" fill="#ffffff"/>')
    out.append(f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="9" fill="#ff6aa6"/>'
               f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="3.4" fill="#ffffff"/>')
    return "".join(out)


def svg(style: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" '
        f'viewBox="0 0 256 256">{TILE}{backbone(style)}{ROBOT_HEAD}</svg>\n'
    )


CONCEPTS = [
    ("c1-ribbon", "Helix ribbon", "Smooth coiled ribbons + loops", "ribbon"),
    ("c2-coil", "Tight coil", "Springier helices, more turns", "coil"),
    ("c3-cartoon", "Cartoon", "Helices with turn ticks — PyMOL style", "cartoon"),
]

renders: dict[str, Image.Image] = {}
for slug, name, _desc, style in CONCEPTS:
    src = svg(style)
    (OUT / f"{slug}.svg").write_text(src, encoding="utf-8")
    png = cairosvg.svg2png(bytestring=src.encode(), output_width=600, output_height=600)
    (OUT / f"{slug}.png").write_bytes(png)
    renders[slug] = Image.open(io.BytesIO(png)).convert("RGBA")
    print(f"  rendered {slug}")

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
f_head = ImageFont.truetype(FB, 36)
f_name = ImageFont.truetype(FB, 27)
f_desc = ImageFont.truetype(FR, 19)

MARGIN, T, GAP = 44, 360, 36
n = len(CONCEPTS)
W = MARGIN * 2 + n * T + (n - 1) * GAP
H = MARGIN * 2 + 78 + T + 84
sheet = Image.new("RGB", (W, H), "#e9e7ef")
d = ImageDraw.Draw(sheet)
d.text((MARGIN, MARGIN), "Robot wrapped in a protein cartoon — helices + loops",
       font=f_head, fill="#1a1535")
d.text((MARGIN, MARGIN + 44),
       "One backbone wraps the robot once: coiled helical spans joined by loops.",
       font=f_desc, fill="#6b6491")

for i, (slug, name, desc, _s) in enumerate(CONCEPTS):
    cx = MARGIN + i * (T + GAP)
    cy = MARGIN + 78
    tile = renders[slug].resize((T, T), Image.LANCZOS)
    sheet.paste(tile, (cx, cy), tile)
    d.text((cx + 4, cy + T + 14), name, font=f_name, fill="#1a1535")
    d.text((cx + 4, cy + T + 48), desc, font=f_desc, fill="#6b6491")

sheet.save(OUT / "_contact-sheet.png")
print(f"\nContact sheet -> {OUT / '_contact-sheet.png'}  ({W}x{H})")
