"""Slime Mold: progressive pruning with failure-guided mutation.

Visually differentiated from Tournament by:
- Dots change color after each mutation (light→dark blue = refinement)
- Mutation zones are prominent with failure info annotations
- Vertical stagger shows the TRANSFORMATION, not just elimination
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

def diamond(id, x, y, w, h, fill, stroke, opacity=100, sw=2):
    s = _s(id)
    return {"type":"diamond","id":id,"x":x-w/2,"y":y-h/2,"width":w,"height":h,
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

# Progressive blue colors — dots get DARKER after each mutation round
# This is the key visual difference from Tournament (uniform blue)
BLUES = [
    ("#DBEAFE", "#93C5FD"),  # R1: lightest (raw generated)
    ("#BFDBFE", "#60A5FA"),  # R2: after 1st mutation
    ("#93C5FD", "#3B82F6"),  # R3: after 2nd mutation
    ("#60A5FA", "#2563EB"),  # R4: after 3rd mutation (darkest)
]

GREEN = "#16A34A"; GREEN_S = "#15803D"
PURPLE_F = "#EDE9FE"; PURPLE_S = "#7C3AED"; PURPLE_DARK = "#4C1D95"
ORANGE_F = "#FEF3C7"; ORANGE_S = "#F59E0B"
GRAY = "#9CA3AF"


# Layout — staggered vertically to show transformation
GEN_X = 60
ROUND_X = [280, 520, 730, 910]    # round dot positions
MUT_X = [410, 635, 830]           # mutation diamond positions
CHAMP_X = 1100
DOT_Y_BASE = 200
DOT_SIZE = 6
DOT_SPACE = 16


# ── Title ────────────────────────────────────────────────────────

elements.append(txt("title", 250, 20, 900, 42,
    "Slime Mold: Progressive Pruning",
    size=36, color="#1F2937", align="center"))
elements.append(txt("sub", 140, 72, 1120, 18,
    "Explore widely, eliminate weakest, mutate survivors via failure analysis, repeat until one champion remains",
    size=13, color="#6B7280", align="center"))


# ── Generation (left) ────────────────────────────────────────────

elements.append(txt("gen_h", GEN_X, 120, 170, 16,
    "4 Diverse Strategies", size=13, color=ORANGE_S))

for i, name in enumerate(["Analytical", "Creative", "Minimal", "Expert"]):
    sy = 148 + i * 26
    elements.append(rect(f"strat_{i}", GEN_X, sy, 120, 20,
        ORANGE_F, ORANGE_S, 80))
    elements.append(txt(f"stratn_{i}", GEN_X+6, sy+3, 110, 14,
        f"{name} (x5)", size=9, color="#374151"))

elements.append(circ("seed_dot", GEN_X+8, 260, 5, GREEN, GREEN_S))
elements.append(txt("seed_l", GEN_X+18, 254, 60, 12, "+ seed", size=9, color=GRAY))

elements.append(arr("a_gen", GEN_X+130, 215,
    [[0,0],[ROUND_X[0]-GEN_X-145, 0]], ORANGE_S, 2, 50))


# ── Round columns with dots ──────────────────────────────────────

round_info = [
    ("R1: 10 examples", 20),
    ("R2: 15 examples", 10),
    ("R3: 20 examples", 5),
    ("R4: 30 examples", 3),
]

for i, (header, n_cands) in enumerate(round_info):
    rx = ROUND_X[i]
    fill, stroke = BLUES[i]

    # Header
    elements.append(txt(f"rh_{i}", rx, 150, 120, 16,
        header, size=12, color=stroke))

    # Dots
    cols = min(5, n_cands)
    for j in range(n_cands):
        row = j // cols
        col = j % cols
        dx = rx + col * DOT_SPACE
        dy = DOT_Y_BASE + row * DOT_SPACE
        is_seed = (i == 0 and j == 0)
        f = GREEN if is_seed else fill
        s = GREEN_S if is_seed else stroke
        elements.append(circ(f"d_{i}_{j}", dx, dy, DOT_SIZE, f, s, 85, 1.5))


# ── Mutation diamonds between rounds ─────────────────────────────

for i, mx in enumerate(MUT_X):
    # Diamond
    elements.append(diamond(f"mut_{i}", mx, DOT_Y_BASE + 15, 50, 50,
        PURPLE_F, PURPLE_S, 80, 2))

    # "mutate" label below diamond
    elements.append(txt(f"mutn_{i}", mx-35, DOT_Y_BASE + 48, 75, 22,
        "mutate via\nfailure info", size=9, color=PURPLE_S, align="center"))

    # Small arrows showing failure info flowing INTO the diamond
    elements.append(arr(f"fail_in_{i}", mx, DOT_Y_BASE + 80,
        [[0,0],[0,20]], PURPLE_S, 1, 35))
    elements.append(txt(f"fail_l_{i}", mx-25, DOT_Y_BASE + 102, 55, 12,
        "failures", size=8, color=PURPLE_DARK, align="center"))


# ── Flow underline ───────────────────────────────────────────────
# R1 has 20 dots: 4 rows × 16px = 64px. DOT_Y=200, bottom=264
# Mutation labels end at ~DOT_Y+115 = 315
# Flow line at 340 gives clear space

flow_y = 340
elements.append(arr("flow", ROUND_X[0], flow_y,
    [[0,0],[ROUND_X[3]+50-ROUND_X[0], 0]], "#3B82F6", 2, 35))

elims = ["20→10", "10→5", "5→3", "3→1"]
for i, rx in enumerate(ROUND_X):
    elements.append(ln(f"ftk_{i}", [[rx, flow_y-4],[rx, flow_y+4]], "#3B82F6", 1.5, 35))
    if i < len(elims):
        elements.append(txt(f"fe_{i}", rx-8, flow_y+14, 55, 12,
            elims[i], size=9, color="#3B82F6", align="center"))


# ── Champion ─────────────────────────────────────────────────────

elements.append(arr("a_champ", ROUND_X[3]+50, DOT_Y_BASE+5,
    [[0,0],[CHAMP_X-ROUND_X[3]-80, 0]], GREEN_S, 3, 65))

elements.append(circ("champion", CHAMP_X, DOT_Y_BASE+5, 28, GREEN, GREEN_S))
elements.append(txt("champ_l", CHAMP_X-30, DOT_Y_BASE+40, 70, 16,
    "Champion", size=14, color=GREEN))
elements.append(txt("champ_l2", CHAMP_X-50, DOT_Y_BASE+58, 110, 14,
    "full validation set", size=11, color=GRAY))


# ── Key insight (below champion, clear of flow line) ─────────────

elements.append(txt("key1", CHAMP_X-60, flow_y-30, 240, 16,
    "Survivors REWRITTEN each round", size=12, color=PURPLE_S))
elements.append(txt("key2", CHAMP_X-60, flow_y-12, 240, 14,
    "(not just filtered — unlike Tournament)", size=10, color=GRAY))


# ── Color progression note ───────────────────────────────────────

elements.append(txt("color_note", ROUND_X[0], flow_y+35, 500, 14,
    "dots darken after each mutation = prompts progressively refined",
    size=10, color="#6B7280"))

# Bottom info
elements.append(txt("rigor", ROUND_X[0], flow_y+55, 500, 14,
    "evaluation rigor: 10 → 15 → 20 → 30 → full valset  |  ~540 rollouts  |  ~22 LLM calls",
    size=11, color="#3B82F6"))


# ── Legend ────────────────────────────────────────────────────────

ly = flow_y + 85
elements.append(rect("leg_bg", 60, ly, 750, 42, "#FAFAFA", "#E5E7EB", 70, 1))
elements.append(txt("leg_t", 70, ly+5, 50, 14, "Legend:", size=11, color="#374151"))

# Show the color progression in the legend
for i, (fill, stroke) in enumerate(BLUES):
    lx = 135 + i * 40
    elements.append(circ(f"leg_c{i}", lx, ly+14, 6, fill, stroke, 85, 1.5))
elements.append(txt("leg_prog", 300, ly+6, 130, 14,
    "light → dark = refined", size=10, color=GRAY))

elements.append(diamond("leg_mut", 460, ly+14, 14, 14, PURPLE_F, PURPLE_S, 80, 1.5))
elements.append(txt("leg_mutt", 472, ly+6, 100, 14, "Mutation step", size=10, color=GRAY))

elements.append(circ("leg_champ", 590, ly+14, 8, GREEN, GREEN_S, 100, 2))
elements.append(txt("leg_champt", 602, ly+6, 80, 14, "Champion", size=10, color=GRAY))

elements.append(txt("leg_foot", 70, ly+26, 700, 12,
    "All candidates evaluated on same example subset per round  |  Failures guide mutation between rounds",
    size=9, color=GRAY))


# ── Render ───────────────────────────────────────────────────────

doc = {
    "type":"excalidraw","version":2,"source":"plotcraft",
    "elements":elements,
    "appState":{"viewBackgroundColor":"#FFFFFF","gridSize":None},
    "files":{},
}

out = "examples/renders/gepa_slime_mold_tree.excalidraw"
with open(out, "w") as f:
    json.dump(doc, f, indent=2)

import subprocess
r = subprocess.run(
    ["uv","run","python","skills/plotcraft-diagram/references/render_excalidraw.py",out],
    capture_output=True, text=True, timeout=30)
print(f"-> {r.stdout.strip()}" if r.returncode==0 else f"ERROR: {r.stderr}")
