"""Robot-face variants on the locked S3 (helix + beta sheet) design.

Everything around the robot is kept; only the face changes -- expression
(eyes / mouth shape) and eye / mouth colour. Output under
``assets/readme/logo-concepts/robot-faces/``.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "assets" / "readme" / "logo-concepts" / "robot-faces"
OUT.mkdir(parents=True, exist_ok=True)

CX, CY = 128, 128
LOOP_COL, HELIX_COL, TICK_COL = "#b29ae6", "#d8cbf6", "#9b81d4"
THETA0, THETA1 = 120.0, 420.0
CYAN, PURPLE, VIOLET, PINK = "#4cc4e4", "#6b1ea8", "#a78bfa", "#ff6aa6"


def ept(theta, rx, ry):
    t = math.radians(theta)
    return CX + rx * math.cos(t), CY + ry * math.sin(t)


def radial(p):
    dx, dy = p[0] - CX, p[1] - CY
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d


def centerline(ta, tb, rx, ry, n):
    return [ept(ta + (tb - ta) * k / n, rx, ry) for k in range(n + 1)]


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


def helix(ta, tb, rx, ry):
    span = abs(tb - ta)
    k = max(3, round(span / 15.0))
    amp, n = 12.0, 64 * max(3, round(span / 15.0))
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
    n, ih, body, head = 24, 15, 9.0, 16.0
    cl = centerline(ta, tb, rx, ry, n)
    nm = normals(cl)
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

HB = ["loop", "helix", "loop", "beta", "loop"]


def wrap_body():
    """The S3 protein wrap (loops + helix + beta + termini), no robot head."""
    rx, ry = 94, 82
    n = len(HB)
    th = [THETA0 + (THETA1 - THETA0) * i / n for i in range(n + 1)]
    loops, structures = [], []
    for i, kind in enumerate(HB):
        a, b = th[i], th[i + 1]
        if kind == "loop":
            loops.append(f'<path d="{poly(centerline(a, b, rx, ry, 16))}"/>')
        elif kind == "helix":
            structures.append(helix(a, b, rx, ry))
        else:
            structures.append(beta(a, b, rx, ry))
    body = [f'<g fill="none" stroke="{LOOP_COL}" stroke-width="7" '
            f'stroke-linecap="round">{"".join(loops)}</g>']
    body.extend(structures)
    for theta, col in ((th[0], CYAN), (th[-1], PINK)):
        x, y = ept(theta, rx, ry)
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{col}"/>'
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="#ffffff"/>')
    return "".join(body)


WRAP = wrap_body()

# ---------------------------------------------------------------- face pieces

def eyes_round(c):
    return (f'<circle cx="112" cy="132" r="10" fill="{c}"/>'
            f'<circle cx="144" cy="132" r="10" fill="{c}"/>'
            f'<circle cx="108.5" cy="128.5" r="3.4" fill="#ffffff"/>'
            f'<circle cx="140.5" cy="128.5" r="3.4" fill="#ffffff"/>')


def eyes_happy(c):
    return (f'<path d="M104 135 Q112 125 120 135" fill="none" stroke="{c}" '
            f'stroke-width="6" stroke-linecap="round"/>'
            f'<path d="M136 135 Q144 125 152 135" fill="none" stroke="{c}" '
            f'stroke-width="6" stroke-linecap="round"/>')


def mouth_smile(c):
    return (f'<path d="M113 153 Q128 165 143 153" fill="none" stroke="{c}" '
            f'stroke-width="5" stroke-linecap="round"/>')


def mouth_open(c):
    return f'<path d="M116 152 Q128 156 140 152 Q128 169 116 152 Z" fill="{c}"/>'


def cheeks(c):
    return (f'<circle cx="99" cy="149" r="6.5" fill="{c}" opacity="0.5"/>'
            f'<circle cx="157" cy="149" r="6.5" fill="{c}" opacity="0.5"/>')


def head(eye_col, face):
    return (f'<g transform="translate(0 -6)">'
            f'<rect x="77" y="124" width="9" height="24" rx="4" fill="#b9a5f0"/>'
            f'<rect x="170" y="124" width="9" height="24" rx="4" fill="#b9a5f0"/>'
            f'<line x1="128" y1="74" x2="128" y2="100" stroke="#d7ccf5" '
            f'stroke-width="5" stroke-linecap="round"/>'
            f'<circle cx="128" cy="69" r="7" fill="{eye_col}"/>'
            f'<circle cx="128" cy="69" r="2.6" fill="#ffffff"/>'
            f'<rect x="85" y="94" width="86" height="80" rx="20" fill="#ffffff"/>'
            f'{face}</g>')


def svg(eye_col, face):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" '
            f'viewBox="0 0 256 256">{TILE}{WRAP}{head(eye_col, face)}</svg>\n')


CONCEPTS = [
    ("f1-classic", "Classic", "Cyan eyes · purple smile",
     CYAN, eyes_round(CYAN) + mouth_smile(PURPLE)),
    ("f2-cheerful", "Cheerful", "Cyan eyes · open smile",
     CYAN, eyes_round(CYAN) + mouth_open(PURPLE)),
    ("f3-sweet", "Sweet", "Happy ^^ eyes · purple smile",
     CYAN, eyes_happy(CYAN) + mouth_smile(PURPLE)),
    ("f4-aqua", "Aqua", "All-cyan face",
     CYAN, eyes_round(CYAN) + mouth_smile(CYAN)),
    ("f5-violet", "Violet", "Violet eyes · violet smile",
     VIOLET, eyes_round(VIOLET) + mouth_smile(VIOLET)),
    ("f6-pink", "Pink", "Cyan eyes · pink smile + blush",
     CYAN, cheeks(PINK) + eyes_round(CYAN) + mouth_smile(PINK)),
]

renders: dict[str, Image.Image] = {}
for slug, name, _desc, eye_col, face in CONCEPTS:
    src = svg(eye_col, face)
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

MARGIN, T, GAP, LABEL = 44, 320, 32, 84
cols = 3
rows = 2
W = MARGIN * 2 + cols * T + (cols - 1) * GAP
H = MARGIN * 2 + 78 + rows * (T + LABEL) + (rows - 1) * GAP
sheet = Image.new("RGB", (W, H), "#e9e7ef")
d = ImageDraw.Draw(sheet)
d.text((MARGIN, MARGIN), "Robot face variants — expression + eye/mouth colour",
       font=f_head, fill="#1a1535")
d.text((MARGIN, MARGIN + 44),
       "Everything around the robot is the locked S3 design; only the face changes.",
       font=f_desc, fill="#6b6491")

y0 = MARGIN + 78
for i, (slug, name, desc, _e, _f) in enumerate(CONCEPTS):
    cx = MARGIN + (i % cols) * (T + GAP)
    cy = y0 + (i // cols) * (T + LABEL + GAP)
    tile = renders[slug].resize((T, T), Image.LANCZOS)
    sheet.paste(tile, (cx, cy), tile)
    d.text((cx + 4, cy + T + 12), name, font=f_name, fill="#1a1535")
    d.text((cx + 4, cy + T + 46), desc, font=f_desc, fill="#6b6491")

sheet.save(OUT / "_contact-sheet.png")
print(f"\nContact sheet -> {OUT / '_contact-sheet.png'}  ({W}x{H})")
