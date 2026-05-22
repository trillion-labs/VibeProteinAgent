"""Refinements of the robot mark so it reads as a *protein* agent.

Keeps the well-liked robot head untouched; only reworks what surrounds it.
The generic elliptical "atom orbit" is replaced with a polypeptide chain
(residue beads, backbone bonds, a few side chains). Output under
``assets/readme/logo-concepts/robot-protein/``.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "assets" / "readme" / "logo-concepts" / "robot-protein"
OUT.mkdir(parents=True, exist_ok=True)

CX, CY = 128, 134
CYAN, PINK, VIOLET = "#4cc4e4", "#ff6aa6", "#a78bfa"

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


def svg(orbit: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" '
        f'viewBox="0 0 256 256">{TILE}{orbit}{ROBOT_HEAD}</svg>\n'
    )


def ring_points(n: int, rx: float, ry: float, deg: float = -20, phase: float = -0.6):
    phi = math.radians(deg)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n + phase
        ex, ey = rx * math.cos(t), ry * math.sin(t)
        pts.append((CX + ex * math.cos(phi) - ey * math.sin(phi),
                    CY + ex * math.sin(phi) + ey * math.cos(phi)))
    return pts


def side_chain(x: float, y: float, color: str, length: float = 15) -> str:
    """A short stub pointing radially outward, tipped with a small atom."""
    dx, dy = x - CX, y - CY
    d = math.hypot(dx, dy) or 1
    ux, uy = dx / d, dy / d
    x1, y1 = x + ux * 7, y + uy * 7
    x2, y2 = x + ux * (7 + length), y + uy * (7 + length)
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="4.5" stroke-linecap="round"/>'
        f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="4.2" fill="{color}"/>'
    )


def peptide(n: int, rx: float, ry: float, bead_r, accents: dict[int, str],
            chains: set[int]) -> str:
    """A closed polypeptide: backbone bonds + residue beads + side chains."""
    pts = ring_points(n, rx, ry)
    bonds = "".join(
        f'<line x1="{pts[i][0]:.1f}" y1="{pts[i][1]:.1f}" '
        f'x2="{pts[(i + 1) % n][0]:.1f}" y2="{pts[(i + 1) % n][1]:.1f}" '
        f'stroke="#9b86d6" stroke-width="5" stroke-linecap="round"/>'
        for i in range(n)
    )
    stubs, beads = [], []
    for i, (x, y) in enumerate(pts):
        col = accents.get(i, "#cbbdf2")
        r = 13 if i in accents else bead_r[i % len(bead_r)]
        if i in chains:
            stubs.append(side_chain(x, y, col, 17 if i in accents else 12))
        beads.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}"/>')
        if i in accents:
            beads.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.6" fill="#ffffff"/>')
    return bonds + "".join(stubs) + "".join(beads)


# --- original (for before/after comparison) -----------------------------------

ORIGINAL = (
    '<ellipse cx="128" cy="134" rx="98" ry="62" fill="none" stroke="#ffffff" '
    'stroke-opacity="0.20" stroke-width="3" transform="rotate(-20 128 134)"/>'
    '<g stroke="#ffffff" stroke-opacity="0.42" stroke-width="3" stroke-linecap="round">'
    '<line x1="206" y1="70" x2="128" y2="134"/>'
    '<line x1="52" y1="184" x2="128" y2="134"/>'
    '<line x1="210" y1="162" x2="128" y2="134"/></g>'
    f'<circle cx="206" cy="70" r="15" fill="{CYAN}"/><circle cx="206" cy="70" r="5" fill="#fff"/>'
    f'<circle cx="52" cy="184" r="15" fill="{PINK}"/><circle cx="52" cy="184" r="5" fill="#fff"/>'
    f'<circle cx="210" cy="162" r="15" fill="{VIOLET}"/><circle cx="210" cy="162" r="5" fill="#fff"/>'
)

# --- V1: fine peptide ring ----------------------------------------------------

V1 = peptide(
    n=16, rx=92, ry=58, bead_r=(6, 8, 5, 7, 6, 9, 6, 7),
    accents={2: CYAN, 7: PINK, 12: VIOLET},
    chains={2, 5, 7, 10, 12, 14},
)

# --- V2: bold peptide chain ---------------------------------------------------

V2 = peptide(
    n=9, rx=84, ry=54, bead_r=(11, 12, 10, 13, 11, 12),
    accents={1: CYAN, 4: PINK, 7: VIOLET},
    chains={1, 3, 4, 6, 7},
)

# --- V3: keep the orbit, turn tool nodes into small molecules -----------------

def molecule(x: float, y: float, color: str) -> str:
    return (
        f'<line x1="{x}" y1="{y}" x2="{x-14}" y2="{y-8}" stroke="{color}" stroke-width="4"/>'
        f'<line x1="{x}" y1="{y}" x2="{x+13}" y2="{y-10}" stroke="{color}" stroke-width="4"/>'
        f'<line x1="{x}" y1="{y}" x2="{x+4}" y2="{y+16}" stroke="{color}" stroke-width="4"/>'
        f'<circle cx="{x}" cy="{y}" r="10" fill="{color}"/>'
        f'<circle cx="{x-14}" cy="{y-8}" r="5.5" fill="{color}"/>'
        f'<circle cx="{x+13}" cy="{y-10}" r="5.5" fill="{color}"/>'
        f'<circle cx="{x+4}" cy="{y+16}" r="5.5" fill="{color}"/>'
        f'<circle cx="{x}" cy="{y}" r="4" fill="#ffffff"/>'
    )

V3 = (
    '<ellipse cx="128" cy="134" rx="98" ry="62" fill="none" stroke="#ffffff" '
    'stroke-opacity="0.20" stroke-width="3" transform="rotate(-20 128 134)"/>'
    '<g stroke="#ffffff" stroke-opacity="0.40" stroke-width="3" stroke-linecap="round">'
    '<line x1="204" y1="74" x2="128" y2="134"/>'
    '<line x1="54" y1="182" x2="128" y2="134"/>'
    '<line x1="206" y1="160" x2="128" y2="134"/></g>'
    + molecule(204, 74, CYAN) + molecule(54, 182, PINK) + molecule(206, 160, VIOLET)
)

CONCEPTS = [
    ("v0-original", "Now", "Generic agent — reads as an atom orbit", ORIGINAL),
    ("v1-peptide-fine", "Peptide ring", "Residue chain + side chains around it", V1),
    ("v2-peptide-bold", "Bold peptide", "A clear necklace of residues", V2),
    ("v3-molecule-nodes", "Molecule nodes", "Lightest touch — tools become molecules", V3),
]

# ------------------------------------------------------------------- rendering

renders: dict[str, Image.Image] = {}
for slug, name, _desc, orbit in CONCEPTS:
    src = svg(orbit)
    (OUT / f"{slug}.svg").write_text(src, encoding="utf-8")
    png = cairosvg.svg2png(bytestring=src.encode(), output_width=600, output_height=600)
    (OUT / f"{slug}.png").write_bytes(png)
    renders[slug] = Image.open(io.BytesIO(png)).convert("RGBA")
    print(f"  rendered {slug}")

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
f_head = ImageFont.truetype(FB, 38)
f_name = ImageFont.truetype(FB, 29)
f_desc = ImageFont.truetype(FR, 20)

MARGIN, T, GAP, LABEL = 48, 372, 44, 96
cols = 2
rows = (len(CONCEPTS) + cols - 1) // cols
W = MARGIN * 2 + cols * T + (cols - 1) * GAP
H = MARGIN * 2 + 84 + rows * (T + LABEL) + (rows - 1) * GAP
sheet = Image.new("RGB", (W, H), "#e9e7ef")
d = ImageDraw.Draw(sheet)
d.text((MARGIN, MARGIN), "Making the robot a protein agent", font=f_head, fill="#1a1535")
d.text((MARGIN, MARGIN + 46),
       "Same robot head; the orbit becomes the protein it works on.",
       font=f_desc, fill="#6b6491")

y0 = MARGIN + 84
for i, (slug, name, desc, _o) in enumerate(CONCEPTS):
    cx = MARGIN + (i % cols) * (T + GAP)
    cy = y0 + (i // cols) * (T + LABEL + GAP)
    tile = renders[slug].resize((T, T), Image.LANCZOS)
    sheet.paste(tile, (cx, cy), tile)
    d.text((cx + 4, cy + T + 16), name, font=f_name, fill="#1a1535")
    d.text((cx + 4, cy + T + 54), desc, font=f_desc, fill="#6b6491")

sheet.save(OUT / "_contact-sheet.png")
print(f"\nContact sheet -> {OUT / '_contact-sheet.png'}  ({W}x{H})")
