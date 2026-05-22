"""Generate original VibeProteinAgent logo concepts as SVG + PNG.

Produces six distinct, hand-built vector marks (no unicode glyphs) under
``assets/readme/logo-concepts/`` plus a contact sheet for side-by-side review.
"""
from __future__ import annotations

import io
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "assets" / "readme" / "logo-concepts"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- shared defs

TILE_DEFS = """
  <linearGradient id="tile" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#3a0d6e"/>
    <stop offset="1" stop-color="#6b1ea8"/>
  </linearGradient>
  <radialGradient id="glow" cx="0.3" cy="0.24" r="0.95">
    <stop offset="0" stop-color="#ffffff" stop-opacity="0.18"/>
    <stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
  </radialGradient>
"""

TILE_BG = (
    '<rect width="256" height="256" rx="58" fill="url(#tile)"/>'
    '<rect width="256" height="256" rx="58" fill="url(#glow)"/>'
)


def svg(defs: str, body: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" '
        'viewBox="0 0 256 256">\n'
        f"  <defs>{TILE_DEFS}{defs}</defs>\n"
        f"  {TILE_BG}\n  {body}\n</svg>\n"
    )


# ---------------------------------------------------------------- 1. Helix Coil

helix = svg(
    """
  <linearGradient id="spine" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#5fd3ec"/>
    <stop offset="1" stop-color="#ff63a6"/>
  </linearGradient>
""",
    """
  <!-- back arcs of the coil -->
  <path fill="none" stroke="#7c3aed" stroke-width="12" stroke-linecap="round"
        d="M170 78 C170 70 86 70 86 104
           M170 130 C170 122 86 122 86 156
           M170 182 C170 174 86 174 86 208"/>
  <!-- front arcs of the coil -->
  <path fill="none" stroke="#ffffff" stroke-width="17" stroke-linecap="round"
        d="M86 52 C86 110 170 110 170 78
           M86 104 C86 162 170 162 170 130
           M86 156 C86 214 170 214 170 182"/>
  <!-- ribbon spine highlight -->
  <path fill="none" stroke="url(#spine)" stroke-width="6" stroke-linecap="round"
        d="M86 52 C86 110 170 110 170 78
           M86 104 C86 162 170 162 170 130
           M86 156 C86 214 170 214 170 182"/>
  <!-- chain termini -->
  <circle cx="86" cy="52" r="11" fill="#5fd3ec"/>
  <circle cx="86" cy="52" r="4.5" fill="#0d2b33"/>
  <circle cx="86" cy="208" r="11" fill="#ff63a6"/>
  <circle cx="86" cy="208" r="4.5" fill="#3d1029"/>
""",
)

# --------------------------------------------------------- 2. Agent Robot Orbit

agent = svg(
    "",
    """
  <!-- orbit -->
  <ellipse cx="128" cy="134" rx="98" ry="66" fill="none"
           stroke="#ffffff" stroke-opacity="0.20" stroke-width="3"
           transform="rotate(-20 128 134)"/>
  <!-- links from tool nodes to the robot -->
  <g stroke="#ffffff" stroke-opacity="0.42" stroke-width="3" stroke-linecap="round">
    <line x1="206" y1="70" x2="128" y2="134"/>
    <line x1="52" y1="184" x2="128" y2="134"/>
    <line x1="210" y1="162" x2="128" y2="134"/>
  </g>
  <!-- satellite tool nodes -->
  <g>
    <circle cx="206" cy="70" r="15" fill="#4cc4e4"/>
    <circle cx="206" cy="70" r="5" fill="#ffffff"/>
    <circle cx="52" cy="184" r="15" fill="#ff63a6"/>
    <circle cx="52" cy="184" r="5" fill="#ffffff"/>
    <circle cx="210" cy="162" r="15" fill="#a78bfa"/>
    <circle cx="210" cy="162" r="5" fill="#ffffff"/>
  </g>
  <!-- robot ears -->
  <rect x="77" y="124" width="9" height="24" rx="4" fill="#b9a5f0"/>
  <rect x="170" y="124" width="9" height="24" rx="4" fill="#b9a5f0"/>
  <!-- antenna -->
  <line x1="128" y1="74" x2="128" y2="100" stroke="#d7ccf5"
        stroke-width="5" stroke-linecap="round"/>
  <circle cx="128" cy="69" r="7" fill="#4cc4e4"/>
  <circle cx="128" cy="69" r="2.6" fill="#ffffff"/>
  <!-- robot head -->
  <rect x="85" y="94" width="86" height="80" rx="20" fill="#ffffff"/>
  <!-- eyes -->
  <circle cx="112" cy="132" r="10" fill="#4cc4e4"/>
  <circle cx="144" cy="132" r="10" fill="#4cc4e4"/>
  <circle cx="108.5" cy="128.5" r="3.4" fill="#ffffff"/>
  <circle cx="140.5" cy="128.5" r="3.4" fill="#ffffff"/>
  <!-- smile -->
  <path d="M113 153 Q128 165 143 153" fill="none" stroke="#6b1ea8"
        stroke-width="5" stroke-linecap="round"/>
""",
)

# ----------------------------------------------------------------- 3. Fold Path

fold = svg(
    """
  <linearGradient id="foldgrad" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#5fd3ec"/>
    <stop offset="0.5" stop-color="#a78bfa"/>
    <stop offset="1" stop-color="#ffffff"/>
  </linearGradient>
""",
    """
  <path fill="none" stroke="url(#foldgrad)" stroke-width="15"
        stroke-linecap="round" stroke-linejoin="round"
        d="M34 174 L66 84 L98 174 L124 96
           C142 58 200 70 196 118
           C193 156 150 158 156 122
           C160 100 188 104 186 130"/>
  <!-- residue dots along the loose chain -->
  <g fill="#ffffff" fill-opacity="0.92">
    <circle cx="66" cy="84" r="6.5"/>
    <circle cx="98" cy="174" r="6.5"/>
    <circle cx="124" cy="96" r="6.5"/>
  </g>
  <circle cx="34" cy="174" r="10" fill="#5fd3ec"/>
  <!-- folded result -->
  <circle cx="186" cy="130" r="15" fill="#ffffff"/>
  <circle cx="186" cy="130" r="6.5" fill="#6b1ea8"/>
""",
)

# ----------------------------------------------------------------- 4. Vibe Wave

vibe = svg(
    "",
    """
  <!-- base-pair rungs -->
  <g stroke="#ffffff" stroke-opacity="0.4" stroke-width="7" stroke-linecap="round">
    <line x1="88" y1="100" x2="88" y2="156"/>
    <line x1="167" y1="100" x2="167" y2="156"/>
  </g>
  <!-- strand A -->
  <path fill="none" stroke="#a78bfa" stroke-width="13" stroke-linecap="round"
        d="M46 128 C78 78 100 78 128 128 C156 178 178 178 210 128"/>
  <!-- strand B -->
  <path fill="none" stroke="#5fd3ec" stroke-width="13" stroke-linecap="round"
        d="M46 128 C78 178 100 178 128 128 C156 78 178 78 210 128"/>
  <!-- nodes -->
  <g fill="#ffffff">
    <circle cx="88" cy="90" r="10"/>
    <circle cx="167" cy="90" r="10"/>
    <circle cx="88" cy="166" r="10"/>
    <circle cx="167" cy="166" r="10"/>
    <circle cx="46" cy="128" r="8"/>
    <circle cx="210" cy="128" r="8"/>
  </g>
  <g>
    <circle cx="88" cy="90" r="4.5" fill="#7c3aed"/>
    <circle cx="167" cy="90" r="4.5" fill="#2a8aa3"/>
    <circle cx="88" cy="166" r="4.5" fill="#2a8aa3"/>
    <circle cx="167" cy="166" r="4.5" fill="#7c3aed"/>
  </g>
""",
)

# -------------------------------------------------------------- 5. Peptide Loop

peptide = svg(
    "",
    """
  <!-- peptide backbone bonds (open chain) -->
  <path fill="none" stroke="#c4b5fd" stroke-width="9"
        stroke-linecap="round" stroke-linejoin="round"
        d="M148 73 L183 108 L176 161 L128 186 L80 161 L73 108"/>
  <!-- residue beads -->
  <g>
    <circle cx="183" cy="108" r="15" fill="#ffffff"/>
    <circle cx="176" cy="161" r="15" fill="#ffffff"/>
    <circle cx="128" cy="186" r="17" fill="#ffffff"/>
    <circle cx="80" cy="161" r="15" fill="#ffffff"/>
  </g>
  <g fill="#6b1ea8">
    <circle cx="183" cy="108" r="5"/>
    <circle cx="176" cy="161" r="5"/>
    <circle cx="128" cy="186" r="6"/>
    <circle cx="80" cy="161" r="5"/>
  </g>
  <!-- termini -->
  <circle cx="148" cy="73" r="15" fill="#5fd3ec"/>
  <circle cx="148" cy="73" r="5.5" fill="#0d2b33"/>
  <circle cx="73" cy="108" r="15" fill="#ff63a6"/>
  <circle cx="73" cy="108" r="5.5" fill="#3d1029"/>
""",
)

# -------------------------------------------------------------- 6. Prompt Bubble

prompt = svg(
    """
  <linearGradient id="ribA" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#7c3aed"/>
    <stop offset="1" stop-color="#a78bfa"/>
  </linearGradient>
""",
    """
  <!-- speech bubble + tail -->
  <path fill="#ffffff"
        d="M82 52 H198 a26 26 0 0 1 26 26 V150 a26 26 0 0 1 -26 26
           H120 L86 206 L92 176 H82 a26 26 0 0 1 -26 -26 V78
           a26 26 0 0 1 26 -26 Z"/>
  <!-- protein ribbons inside -->
  <path fill="none" stroke="url(#ribA)" stroke-width="13" stroke-linecap="round"
        d="M84 110 C104 82 124 92 140 116 C154 138 178 136 196 104"/>
  <path fill="none" stroke="#4cc4e4" stroke-width="9" stroke-linecap="round"
        d="M86 140 C112 156 142 138 166 148 C178 153 188 148 196 138"/>
  <circle cx="140" cy="116" r="6" fill="#ff63a6"/>
""",
)

# ----------------------------------------------- 7. Agent Helix (fusion: 1 + 2)

helix_agent = svg(
    """
  <linearGradient id="spine" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#5fd3ec"/>
    <stop offset="1" stop-color="#ff63a6"/>
  </linearGradient>
""",
    """
  <g transform="translate(-25.6 -24.8) scale(1.2)">
  <!-- robot ears -->
  <rect x="77" y="124" width="9" height="24" rx="4" fill="#b9a5f0"/>
  <rect x="170" y="124" width="9" height="24" rx="4" fill="#b9a5f0"/>
  <!-- antenna -->
  <line x1="128" y1="74" x2="128" y2="100" stroke="#d7ccf5"
        stroke-width="5" stroke-linecap="round"/>
  <circle cx="128" cy="69" r="7" fill="#4cc4e4"/>
  <circle cx="128" cy="69" r="2.6" fill="#ffffff"/>
  <!-- robot head -->
  <rect x="85" y="94" width="86" height="80" rx="20" fill="#ffffff"/>
  <!-- visor screen -->
  <rect x="97" y="108" width="62" height="58" rx="14" fill="#2a0a52"/>
  <!-- alpha-helix protein glowing on the visor -->
  <path fill="none" stroke="#7c3aed" stroke-width="6" stroke-linecap="round"
        d="M144 128 C144 122 112 122 112 138"/>
  <path fill="none" stroke="#ffffff" stroke-width="9" stroke-linecap="round"
        d="M112 118 C112 138 144 138 144 128
           M112 138 C112 158 144 158 144 148"/>
  <path fill="none" stroke="url(#spine)" stroke-width="3" stroke-linecap="round"
        d="M112 118 C112 138 144 138 144 128
           M112 138 C112 158 144 158 144 148"/>
  <circle cx="112" cy="118" r="4.6" fill="#5fd3ec"/>
  <circle cx="144" cy="148" r="4.6" fill="#ff63a6"/>
  </g>
""",
)

CONCEPTS = [
    ("1-helix-coil", "Helix Coil", "Classic alpha-helix ribbon", helix),
    ("2-agent-orbit", "Agent Robot", "Robot agent wired to tool nodes", agent),
    ("3-fold-path", "Fold Path", "Loose chain folding into structure", fold),
    ("4-vibe-wave", "Vibe Wave", "Twin strands, one energy pulse", vibe),
    ("5-peptide-loop", "Peptide Loop", "Folded peptide bead-chain", peptide),
    ("6-prompt-bubble", "Prompt Bubble", "Speech bubble + protein ribbon", prompt),
    ("7-agent-helix", "Agent Helix  (1+2)", "Robot agent with a helix visor", helix_agent),
]

# ------------------------------------------------------------------- rendering

renders: dict[str, Image.Image] = {}
for slug, name, _desc, src in CONCEPTS:
    (OUT / f"{slug}.svg").write_text(src, encoding="utf-8")
    png = cairosvg.svg2png(bytestring=src.encode(), output_width=600, output_height=600)
    (OUT / f"{slug}.png").write_bytes(png)
    renders[slug] = Image.open(io.BytesIO(png)).convert("RGBA")
    print(f"  rendered {slug}")

# ----------------------------------------------------------------- contact sheet

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
f_head = ImageFont.truetype(FB, 40)
f_name = ImageFont.truetype(FB, 30)
f_desc = ImageFont.truetype(FR, 21)

MARGIN, TILE, GAP, LABEL = 48, 360, 48, 104
cols = 3
rows = (len(CONCEPTS) + cols - 1) // cols
W = MARGIN * 2 + cols * TILE + (cols - 1) * GAP
H = MARGIN * 2 + 96 + rows * (TILE + LABEL) + (rows - 1) * GAP
sheet = Image.new("RGB", (W, H), "#f4f1fb")
d = ImageDraw.Draw(sheet)

d.text((MARGIN, MARGIN), "VibeProteinAgent — Logo Concepts", font=f_head, fill="#1a1535")
d.text((MARGIN, MARGIN + 52), "Original vector marks. #7 fuses the helix (#1) with the robot agent (#2).",
       font=f_desc, fill="#6b6491")

y0 = MARGIN + 96
for i, (slug, name, desc, _src) in enumerate(CONCEPTS):
    cx = MARGIN + (i % cols) * (TILE + GAP)
    cy = y0 + (i // cols) * (TILE + LABEL + GAP)
    tile = renders[slug].resize((TILE, TILE), Image.LANCZOS)
    sheet.paste(tile, (cx, cy), tile)
    d.text((cx + 6, cy + TILE + 18), name, font=f_name, fill="#1a1535")
    d.text((cx + 6, cy + TILE + 58), desc, font=f_desc, fill="#6b6491")
    # favicon-size legibility check, top-right of the label band
    fav = renders[slug].resize((40, 40), Image.LANCZOS)
    sheet.paste(fav, (cx + TILE - 46, cy + TILE + 24), fav)

sheet.save(OUT / "_contact-sheet.png")
print(f"\nContact sheet -> {OUT / '_contact-sheet.png'}  ({W}x{H})")
