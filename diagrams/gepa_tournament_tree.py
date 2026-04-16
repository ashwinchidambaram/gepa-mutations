"""Tournament: single-elimination bracket with converging bracket lines.

Shows the actual bracket structure — pairs merge into winners at each round.
Compressed view: 16 representative entries through 4 visible rounds to champion,
with labels showing the full 64→32→16→8→4→2→1 progression.
"""

import json
import zlib

def _s(n): return zlib.adler32(n.encode())

def circ(id, x, y, r, fill, stroke, opacity=100, sw=2):
    s = _s(id)
    return {"type":"ellipse","id":id,"x":x-r,"y":y-r,"width":r*2,"height":r*2,
        "strokeColor":stroke,"backgroundColor":fill,
        "fillStyle":"solid","strokeWidth":sw,"strokeStyle":"solid",
        "roughness":1,"opacity":opacity,"angle":0,
        "seed":s,"version":1,"versionNonce":s^0xFFFF,
        "isDeleted":False,"groupIds":[],"boundElements":None,
        "link":None,"locked":False}

def rect(id, x, y, w, h, fill, stroke, opacity=100, sw=1.5):
    s = _s(id)
    return {"type":"rectangle","id":id,"x":x,"y":y,"width":w,"height":h,
        "strokeColor":stroke,"backgroundColor":fill,
        "fillStyle":"solid","strokeWidth":sw,"strokeStyle":"solid",
        "roughness":1,"opacity":opacity,"angle":0,
        "seed":s,"version":1,"versionNonce":s^0xFFFF,
        "isDeleted":False,"groupIds":[],"boundElements":None,
        "link":None,"locked":False,"roundness":{"type":3}}

def txt(id, x, y, w, h, content, size=14, color="#374151", align="left"):
    s = _s(id)
    return {"type":"text","id":id,"x":x,"y":y,"width":w,"height":h,
        "text":content,"originalText":content,
        "fontSize":size,"fontFamily":1,"textAlign":align,"verticalAlign":"top",
        "strokeColor":color,"backgroundColor":"transparent",
        "fillStyle":"solid","strokeWidth":1,"strokeStyle":"solid",
        "roughness":0,"opacity":100,"angle":0,
        "seed":s,"version":1,"versionNonce":s^0xFFFF,
        "isDeleted":False,"groupIds":[],"boundElements":None,
        "link":None,"locked":False,"containerId":None,"lineHeight":1.25}

def ln(id, pts, stroke, width=2, opacity=100):
    s = _s(id)
    mx = min(p[0] for p in pts); my = min(p[1] for p in pts)
    rel = [[p[0]-mx,p[1]-my] for p in pts]
    return {"type":"line","id":id,"x":mx,"y":my,
        "width":max(p[0] for p in rel) or 1,"height":max(p[1] for p in rel) or 1,
        "strokeColor":stroke,"backgroundColor":"transparent",
        "fillStyle":"solid","strokeWidth":width,"strokeStyle":"solid",
        "roughness":1,"opacity":opacity,"angle":0,
        "seed":s,"version":1,"versionNonce":s^0xFFFF,
        "isDeleted":False,"groupIds":[],"boundElements":None,
        "link":None,"locked":False,"points":rel}

def arr(id, x, y, pts, stroke="#374151", width=2, opacity=60):
    s = _s(id)
    return {"type":"arrow","id":id,"x":x,"y":y,
        "width":abs(pts[-1][0]) or 1,"height":abs(pts[-1][1]) or 1,
        "strokeColor":stroke,"backgroundColor":"transparent",
        "fillStyle":"solid","strokeWidth":width,"strokeStyle":"solid",
        "roughness":1,"opacity":opacity,"angle":0,
        "seed":s,"version":1,"versionNonce":s^0xFFFF,
        "isDeleted":False,"groupIds":[],"boundElements":None,
        "link":None,"locked":False,"points":pts,
        "startBinding":None,"endBinding":None,
        "startArrowhead":None,"endArrowhead":"arrow"}


elements = []

BLUE_F = "#DBEAFE"; BLUE_S = "#3B82F6"
GREEN = "#16A34A"; GREEN_S = "#15803D"
ORANGE_F = "#FEF3C7"; ORANGE_S = "#F59E0B"
PURPLE_F = "#EDE9FE"; PURPLE_S = "#7C3AED"
RED_F = "#FEE2E2"; RED_S = "#EF4444"
GRAY = "#9CA3AF"


# ── Title ────────────────────────────────────────────────────────

elements.append(txt("title", 280, 20, 900, 42,
    "Tournament: Single-Elimination Bracket",
    size=36, color="#1F2937", align="center"))
elements.append(txt("sub", 220, 72, 1020, 18,
    "Maximum initial diversity, head-to-head elimination, no mutation — the pool is frozen after generation",
    size=13, color="#6B7280", align="center"))


# ── Generation strategies (left) ─────────────────────────────────

elements.append(txt("gen_h", 50, 120, 180, 16,
    "4 Generation Strategies", size=13, color=ORANGE_S))

strats = [
    ("Reasoning (x16)", ORANGE_F, ORANGE_S),
    ("Format (x16)", BLUE_F, BLUE_S),
    ("Detail Level (x16)", PURPLE_F, PURPLE_S),
    ("Error Prev. (x16)", RED_F, RED_S),
]
sy = 148
for i, (name, fill, stroke) in enumerate(strats):
    elements.append(rect(f"st_{i}", 50, sy, 140, 22, fill, stroke, 85))
    elements.append(txt(f"stn_{i}", 56, sy+3, 130, 14, name, size=9, color="#374151"))
    sy += 28

elements.append(circ("seed_d", 60, sy+6, 5, GREEN, GREEN_S))
elements.append(txt("seed_l", 70, sy, 60, 12, "+ seed", size=9, color=GRAY))

# Arrow to bracket
elements.append(arr("a_gen", 198, 210, [[0,0],[60,0]], ORANGE_S, 2, 50))
elements.append(txt("pool_l", 205, 195, 50, 14, "= 64", size=11, color=ORANGE_S))


# ══════════════════════════════════════════════════════════════════
# BRACKET — converging lines showing pairs → winners
# Show 16 entries converging through 4 visible rounds to champion
# (represents the full 64→32→16→8→4→2→1)
# ══════════════════════════════════════════════════════════════════

BRACKET_LEFT = 280
BRACKET_GAP = 180   # horizontal gap between round columns
DOT_R = 6
ENTRY_SPACING = 28  # vertical spacing between entries in round 1

# Round column x positions
R_X = [BRACKET_LEFT + i * BRACKET_GAP for i in range(5)]  # R1, R2, R3, R4, Final

# Round 1: 16 entries (representing 32 survivors from 64)
r1_entries = 16
r1_top = 140
r1_positions = [r1_top + i * ENTRY_SPACING for i in range(r1_entries)]

# Draw R1 dots
for i, y in enumerate(r1_positions):
    fill = GREEN if i == 0 else BLUE_F
    stroke = GREEN_S if i == 0 else BLUE_S
    elements.append(circ(f"r1_{i}", R_X[0], y, DOT_R, fill, stroke, 75, 1.5))

# Round 2: 8 winners (each pair → 1 winner)
r2_positions = [(r1_positions[i*2] + r1_positions[i*2+1]) / 2 for i in range(8)]

for i, y in enumerate(r2_positions):
    elements.append(circ(f"r2_{i}", R_X[1], y, DOT_R, BLUE_F, BLUE_S, 80, 1.5))

# Bracket lines: R1 pairs → R2 winners
for i in range(8):
    y_top = r1_positions[i*2]
    y_bot = r1_positions[i*2+1]
    y_mid = r2_positions[i]
    # Horizontal from top entry
    elements.append(ln(f"br1_t{i}", [[R_X[0]+DOT_R+2, y_top], [R_X[0]+30, y_top], [R_X[0]+30, y_mid], [R_X[1]-DOT_R-2, y_mid]],
        BLUE_S, 1, 40))
    # Horizontal from bottom entry
    elements.append(ln(f"br1_b{i}", [[R_X[0]+DOT_R+2, y_bot], [R_X[0]+30, y_bot], [R_X[0]+30, y_mid]],
        BLUE_S, 1, 40))

# Round 3: 4 winners
r3_positions = [(r2_positions[i*2] + r2_positions[i*2+1]) / 2 for i in range(4)]

for i, y in enumerate(r3_positions):
    elements.append(circ(f"r3_{i}", R_X[2], y, DOT_R+1, BLUE_F, BLUE_S, 85, 1.5))

# Bracket lines: R2 → R3
for i in range(4):
    y_top = r2_positions[i*2]
    y_bot = r2_positions[i*2+1]
    y_mid = r3_positions[i]
    elements.append(ln(f"br2_t{i}", [[R_X[1]+DOT_R+2, y_top], [R_X[1]+35, y_top], [R_X[1]+35, y_mid], [R_X[2]-DOT_R-3, y_mid]],
        BLUE_S, 1.2, 45))
    elements.append(ln(f"br2_b{i}", [[R_X[1]+DOT_R+2, y_bot], [R_X[1]+35, y_bot], [R_X[1]+35, y_mid]],
        BLUE_S, 1.2, 45))

# Round 4: 2 winners (semifinals)
r4_positions = [(r3_positions[i*2] + r3_positions[i*2+1]) / 2 for i in range(2)]

for i, y in enumerate(r4_positions):
    elements.append(circ(f"r4_{i}", R_X[3], y, DOT_R+2, "#93C5FD", "#2563EB", 90, 2))

# Bracket lines: R3 → R4
for i in range(2):
    y_top = r3_positions[i*2]
    y_bot = r3_positions[i*2+1]
    y_mid = r4_positions[i]
    elements.append(ln(f"br3_t{i}", [[R_X[2]+DOT_R+3, y_top], [R_X[2]+40, y_top], [R_X[2]+40, y_mid], [R_X[3]-DOT_R-4, y_mid]],
        "#2563EB", 1.5, 50))
    elements.append(ln(f"br3_b{i}", [[R_X[2]+DOT_R+3, y_bot], [R_X[2]+40, y_bot], [R_X[2]+40, y_mid]],
        "#2563EB", 1.5, 50))

# Final: 1 champion
final_y = (r4_positions[0] + r4_positions[1]) / 2

# Bracket lines: R4 → Final
elements.append(ln("br4_t", [[R_X[3]+DOT_R+4, r4_positions[0]], [R_X[3]+45, r4_positions[0]], [R_X[3]+45, final_y], [R_X[4]-30, final_y]],
    GREEN_S, 2, 60))
elements.append(ln("br4_b", [[R_X[3]+DOT_R+4, r4_positions[1]], [R_X[3]+45, r4_positions[1]], [R_X[3]+45, final_y]],
    GREEN_S, 2, 60))

# Champion circle
elements.append(circ("champion", R_X[4], final_y, 26, GREEN, GREEN_S))
elements.append(txt("champ_l", R_X[4]-30, final_y+34, 70, 16,
    "Champion", size=14, color=GREEN))


# ── Round headers (above bracket) ────────────────────────────────

round_headers = [
    ("R1\n5 ex", "64→32"),
    ("R2\n7 ex", "32→16"),
    ("R3\n10 ex", "16→8"),
    ("R4\n15 ex", "8→4"),
    ("Final\n20 ex", "2→1"),
]
for i, (label, elim) in enumerate(round_headers):
    elements.append(txt(f"rh_{i}", R_X[i]-20, 108, 60, 30,
        label, size=10, color=BLUE_S, align="center"))


# ── Annotations ──────────────────────────────────────────────────

# "POOL FROZEN" near the bracket entry
elements.append(txt("frozen", BRACKET_LEFT-15, r1_positions[-1]+20, 100, 14,
    "POOL FROZEN", size=10, color=RED_S))
elements.append(txt("frozen2", BRACKET_LEFT-15, r1_positions[-1]+35, 120, 12,
    "no mutation\nbetween rounds", size=9, color=GRAY))

# Head-to-head note
elements.append(txt("h2h", R_X[1]-20, r2_positions[-1]+30, 150, 12,
    "head-to-head on\nsame examples", size=9, color=BLUE_S))

# Champion notes
elements.append(txt("champ_note1", R_X[4]-50, final_y+55, 150, 14,
    "full validation set", size=11, color=GRAY))

# Key insight
elements.append(txt("insight1", R_X[4]-60, final_y+80, 220, 14,
    "No mutation between rounds.", size=12, color=RED_S))
elements.append(txt("insight2", R_X[4]-60, final_y+98, 220, 22,
    "If the best prompt wasn't\ngenerated in step 1,\nit can never be found.", size=10, color=GRAY))


# ── Bottom info ──────────────────────────────────────────────────

info_y = max(r1_positions[-1] + 70, final_y + 130)
elements.append(txt("rigor", 280, info_y, 500, 14,
    "evaluation rigor:  5 → 7 → 10 → 15 → 20 → full valset",
    size=11, color=BLUE_S))
elements.append(txt("total", 280, info_y+18, 400, 14,
    "total: ~904 preliminary rollouts  |  ties favor lower seed",
    size=11, color=GRAY))


# ── Legend ────────────────────────────────────────────────────────

ly = info_y + 45
elements.append(rect("leg_bg", 50, ly, 700, 42, "#FAFAFA", "#E5E7EB", 70, 1))
elements.append(txt("leg_t", 60, ly+5, 50, 14, "Legend:", size=11, color="#374151"))

elements.append(circ("lc1", 130, ly+14, 6, GREEN, GREEN_S, 100, 1.5))
elements.append(txt("lt1", 140, ly+6, 80, 14, "Seed prompt", size=10, color=GRAY))

elements.append(circ("lc2", 245, ly+14, 6, BLUE_F, BLUE_S, 80, 1.5))
elements.append(txt("lt2", 255, ly+6, 80, 14, "Candidate", size=10, color=GRAY))

elements.append(ln("lc3", [[355, ly+14],[380, ly+14]], BLUE_S, 1.5, 50))
elements.append(txt("lt3", 385, ly+6, 100, 14, "Bracket line", size=10, color=GRAY))

elements.append(circ("lc4", 505, ly+14, 8, GREEN, GREEN_S, 100, 2))
elements.append(txt("lt4", 517, ly+6, 80, 14, "Champion", size=10, color=GRAY))

elements.append(txt("leg_foot", 60, ly+26, 650, 12,
    "Both prompts in a matchup evaluated on the SAME examples  |  Bracket shows 16 of 64 entries (compressed)",
    size=9, color=GRAY))


# ── Render ───────────────────────────────────────────────────────

doc = {
    "type":"excalidraw","version":2,"source":"plotcraft",
    "elements":elements,
    "appState":{"viewBackgroundColor":"#FFFFFF","gridSize":None},
    "files":{},
}

out = "examples/renders/gepa_tournament_tree.excalidraw"
with open(out, "w") as f:
    json.dump(doc, f, indent=2)

import subprocess
r = subprocess.run(
    ["uv","run","python","skills/plotcraft-diagram/references/render_excalidraw.py",out],
    capture_output=True, text=True, timeout=30)
print(f"-> {r.stdout.strip()}" if r.returncode==0 else f"ERROR: {r.stderr}")
