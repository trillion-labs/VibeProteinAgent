"""Minimal, professional logo marks for VibeProteinAgent.

Single-colour geometric marks in the spirit of the Claude / OpenAI logos:
one idea each, clean, no clutter. Output under
``assets/readme/logo-concepts/minimal/``.
"""
from __future__ import annotations

import io
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "assets" / "readme" / "logo-concepts" / "minimal"
OUT.mkdir(parents=True, exist_ok=True)

BRAND = "#6b1ea8"
INK = "#1a1535"


def page(inner: str, bg: str = "#ffffff") -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" '
        'viewBox="0 0 256 256">\n'
        f'  <rect width="256" height="256" fill="{bg}"/>\n  {inner}\n</svg>\n'
    )


# Side-view alpha-helix coil. One continuous line; the back of each turn is
# broken where the front passes over it (erase discs at the three crossings).
_COIL = ("M96 50 C96 106 160 106 160 76 C160 67 96 67 96 100 "
         "C96 156 160 156 160 126 C160 117 96 117 96 150 "
         "C96 206 160 206 160 176 C160 167 96 167 96 200")
_FRONT = ("M96 50 C96 106 160 106 160 76 M96 100 C96 156 160 156 160 126 "
          "M96 150 C96 206 160 206 160 176")


def _coil(stroke: str, gap: str) -> str:
    discs = "".join(
        f'<circle stroke="none" fill="{gap}" cx="106" cy="{cy}" r="13"/>'
        for cy in (81, 131, 181)
    )
    return (
        '<g fill="none" stroke-linecap="round">'
        f'<path stroke="{stroke}" stroke-width="16" d="{_COIL}"/>'
        f"{discs}"
        f'<path stroke="{stroke}" stroke-width="16" d="{_FRONT}"/>'
        "</g>"
    )


# --- 1. Monoline Helix -- the protein structure, one clean continuous line ---

def m1(color: str) -> str:
    return page(_coil(color, "#ffffff"))


# --- 2. Backbone Fold -- a single chain folding back on itself ---

def m2(color: str) -> str:
    d = ("M96 172 L96 88 A16 16 0 0 1 128 88 L128 160 "
         "A16 16 0 0 0 160 160 L160 76")
    return page(
        f'<path fill="none" stroke="{color}" stroke-width="21" '
        f'stroke-linecap="round" stroke-linejoin="round" d="{d}"/>'
    )


# --- 3. Helix Mark -- a bold solid icon, helix carved in negative space ---

def m3(color: str) -> str:
    return page(
        f'<rect x="24" y="24" width="208" height="208" rx="58" fill="{color}"/>'
        f'<g transform="translate(19.2 18.75) scale(0.85)">{_coil("#ffffff", color)}</g>'
    )


# --- 4. Triskelion -- three spiral arms: the repo's three ways to run ---

def m4(color: str) -> str:
    arm = "M128 128 C120 100 140 78 172 84 C192 88 196 112 180 122"
    arms = "".join(
        f'<path d="{arm}" transform="rotate({a} 128 128)"/>' for a in (0, 120, 240)
    )
    return page(
        f'<g fill="none" stroke="{color}" stroke-width="18" stroke-linecap="round">'
        f'{arms}</g>'
    )


CONCEPTS = [
    ("m1-monoline-helix", "Monoline Helix", "The protein, one continuous line", m1),
    ("m2-backbone-fold", "Backbone Fold", "A chain folding into structure", m2),
    ("m3-helix-mark", "Helix Mark", "Helix carved in negative space", m3),
    ("m4-triskelion", "Triskelion", "Three arms — three ways to run", m4),
]

# ------------------------------------------------------------------- rendering

renders: dict[str, Image.Image] = {}
for slug, name, _desc, fn in CONCEPTS:
    src = fn(BRAND)
    (OUT / f"{slug}.svg").write_text(src, encoding="utf-8")
    png = cairosvg.svg2png(bytestring=src.encode(), output_width=600, output_height=600)
    (OUT / f"{slug}.png").write_bytes(png)
    renders[slug] = Image.open(io.BytesIO(png)).convert("RGBA")
    mono = cairosvg.svg2png(bytestring=fn(INK).encode(), output_width=140, output_height=140)
    renders[slug + "_ink"] = Image.open(io.BytesIO(mono)).convert("RGBA")
    print(f"  rendered {slug}")

# ----------------------------------------------------------------- contact sheet

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
f_head = ImageFont.truetype(FB, 38)
f_name = ImageFont.truetype(FB, 29)
f_desc = ImageFont.truetype(FR, 20)
f_tiny = ImageFont.truetype(FR, 15)

MARGIN, TILE, GAP, LABEL = 48, 372, 44, 156
cols = 2
rows = (len(CONCEPTS) + cols - 1) // cols
W = MARGIN * 2 + cols * TILE + (cols - 1) * GAP
H = MARGIN * 2 + 92 + rows * (TILE + LABEL) + (rows - 1) * GAP
sheet = Image.new("RGB", (W, H), "#e9e7ef")
d = ImageDraw.Draw(sheet)

d.text((MARGIN, MARGIN), "VibeProteinAgent — Minimal Marks", font=f_head, fill="#1a1535")
d.text((MARGIN, MARGIN + 48),
       "One idea per mark. Single colour, scales to a favicon, works in one ink.",
       font=f_desc, fill="#6b6491")

y0 = MARGIN + 92
for i, (slug, name, desc, _fn) in enumerate(CONCEPTS):
    cx = MARGIN + (i % cols) * (TILE + GAP)
    cy = y0 + (i // cols) * (TILE + LABEL + GAP)
    tile = renders[slug].resize((TILE, TILE), Image.LANCZOS)
    sheet.paste(tile, (cx, cy), tile)
    d.rectangle([cx, cy, cx + TILE - 1, cy + TILE - 1], outline="#d6d2e2", width=1)
    d.text((cx + 4, cy + TILE + 18), name, font=f_name, fill="#1a1535")
    d.text((cx + 4, cy + TILE + 56), desc, font=f_desc, fill="#6b6491")
    # small-size + single-ink proofs
    for j, (key, cap) in enumerate(((slug, "32 px"), (slug + "_ink", "1-ink"))):
        px = cx + 4 + j * 124
        proof = renders[key].resize((48, 48), Image.LANCZOS)
        sheet.paste(proof, (px, cy + TILE + 90), proof)
        sm = renders[key].resize((20, 20), Image.LANCZOS)
        sheet.paste(sm, (px + 58, cy + TILE + 104), sm)
        d.text((px + 84, cy + TILE + 106), cap, font=f_tiny, fill="#9b95b3")

sheet.save(OUT / "_contact-sheet.png")
print(f"\nContact sheet -> {OUT / '_contact-sheet.png'}  ({W}x{H})")

# ------------------------------------------------------------------- lockups

f_word = ImageFont.truetype(FB, 46)
LM, MARK, ROW = 44, 96, 138
lock = Image.new("RGB", (760, LM * 2 + len(CONCEPTS) * ROW), "#ffffff")
ld = ImageDraw.Draw(lock)
for i, (slug, name, _desc, _fn) in enumerate(CONCEPTS):
    ry = LM + i * ROW
    mk = renders[slug].resize((MARK, MARK), Image.LANCZOS)
    lock.paste(mk, (LM, ry + (ROW - MARK) // 2 - 8), mk)
    tx = LM + MARK + 30
    ty = ry + (ROW - 46) // 2 - 14
    ld.text((tx, ty), "VibeProteinAgent", font=f_word, fill="#1a1535")
    ld.text((tx + 2, ty + 56), name, font=f_tiny, fill="#9b95b3")
lock.save(OUT / "_lockups.png")
print(f"Lockups      -> {OUT / '_lockups.png'}  ({lock.size[0]}x{lock.size[1]})")
