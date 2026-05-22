"""Original, non-cliche logo marks for VibeProteinAgent.

Deliberately avoids the over-used alpha-helix. Each mark draws on a less
obvious facet of the repo: a beta-hairpin (protein structure that is not a
helix), molecular binding, a contact map, and the agent's prompt. Output
under ``assets/readme/logo-concepts/original/``.
"""
from __future__ import annotations

import io
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "assets" / "readme" / "logo-concepts" / "original"
OUT.mkdir(parents=True, exist_ok=True)

BRAND, LIGHT = "#6b1ea8", "#a779d0"
INK, INK2 = "#1a1535", "#8b86a0"


def page(inner: str, bg: str = "#ffffff") -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" '
        'viewBox="0 0 256 256">\n'
        f'  <rect width="256" height="256" fill="{bg}"/>\n  {inner}\n</svg>\n'
    )


# --- 1. Hairpin -- a beta-hairpin: a chain folds back, a strand arrow out ---

def o1(c1: str, c2: str) -> str:
    return page(
        f'<path fill="none" stroke="{c1}" stroke-width="23" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'd="M100 198 L100 86 A28 28 0 0 1 156 86 L156 150"/>'
        f'<circle cx="100" cy="198" r="15" fill="{c1}"/>'
        f'<path fill="{c1}" d="M156 196 L133 150 L179 150 Z"/>'
    )


# --- 2. Binder -- two complementary shapes that lock together ---

def o2(c1: str, c2: str) -> str:
    return page(
        f'<circle cx="108" cy="128" r="62" fill="{c1}"/>'
        f'<circle cx="170" cy="128" r="31" fill="#ffffff"/>'
        f'<circle cx="172" cy="128" r="25" fill="{c2}"/>'
    )


# --- 3. Matrix -- a residue contact map, a protein's structural fingerprint -

def o3(c1: str, c2: str) -> str:
    fill: dict[tuple[int, int], str] = {(i, i): c1 for i in range(5)}
    for rc in ((1, 3), (3, 1), (0, 2), (2, 0)):
        fill[rc] = c2
    cells = [
        f'<rect x="{46 + c * 34}" y="{46 + r * 34}" width="28" height="28" '
        f'rx="8" fill="{col}"/>'
        for (r, c), col in fill.items()
    ]
    return page("".join(cells))


# --- 4. Prompt -- a command caret that generates a molecule ---

def o4(c1: str, c2: str) -> str:
    bonds = "M170 100 L198 130 M198 130 L162 156 M162 156 L170 100"
    return page(
        f'<path fill="none" stroke="{c1}" stroke-width="24" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'd="M70 78 L120 128 L70 178"/>'
        f'<path fill="none" stroke="{c1}" stroke-width="10" '
        f'stroke-linejoin="round" d="{bonds}"/>'
        f'<g fill="{c1}">'
        f'<circle cx="170" cy="100" r="15"/>'
        f'<circle cx="198" cy="130" r="15"/>'
        f'<circle cx="162" cy="156" r="15"/>'
        f'</g>'
    )


CONCEPTS = [
    ("o1-hairpin", "Hairpin", "A folded chain — not a helix", o1),
    ("o2-binder", "Binder", "Two shapes that lock together", o2),
    ("o3-matrix", "Matrix", "A contact map fingerprint", o3),
    ("o4-prompt", "Prompt", "A command builds a molecule", o4),
]

# ------------------------------------------------------------------- rendering

renders: dict[str, Image.Image] = {}
for slug, name, _desc, fn in CONCEPTS:
    src = fn(BRAND, LIGHT)
    (OUT / f"{slug}.svg").write_text(src, encoding="utf-8")
    png = cairosvg.svg2png(bytestring=src.encode(), output_width=600, output_height=600)
    (OUT / f"{slug}.png").write_bytes(png)
    renders[slug] = Image.open(io.BytesIO(png)).convert("RGBA")
    mono = cairosvg.svg2png(bytestring=fn(INK, INK2).encode(),
                            output_width=140, output_height=140)
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

d.text((MARGIN, MARGIN), "VibeProteinAgent — Original Marks", font=f_head, fill="#1a1535")
d.text((MARGIN, MARGIN + 48),
       "No helix. Each mark draws on a different facet of the repo.",
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
