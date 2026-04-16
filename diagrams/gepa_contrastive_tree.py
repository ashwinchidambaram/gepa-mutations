"""Contrastive Reflection: GEPA + historical evidence injection.

Key mechanism to convey:
1. Same GEPA evolutionary tree
2. Every evaluation records per-example scores into a persistent index
3. Before reflection, mine pairs: "on example X, past candidate scored higher"
4. Inject these as concrete evidence into the reflection prompt
5. LLM sees WHAT WORKED BETTER, not just its own failures
6. Zero extra LLM cost — pure CPU memory lookup
"""

import json
import math
import zlib


def _s(n): return zlib.adler32(n.encode())

def circ(id, x, y, r, fill, stroke, opacity=100, sw=2, dash="solid"):
    s = _s(id)
    return {"type":"ellipse","id":id,"x":x-r,"y":y-r,"width":r*2,"height":r*2,
        "strokeColor":stroke,"backgroundColor":fill,
        "fillStyle":"solid","strokeWidth":sw,"strokeStyle":dash,
        "roughness":1,"opacity":opacity,"angle":0,
        "seed":s,"version":1,"versionNonce":s^0xFFFF,
        "isDeleted":False,"groupIds":[],"boundElements":None,
        "link":None,"locked":False}

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

def ln(id, pts, stroke, width=2, opacity=100, dash="solid"):
    s = _s(id)
    mx = min(p[0] for p in pts); my = min(p[1] for p in pts)
    rel = [[p[0]-mx,p[1]-my] for p in pts]
    return {"type":"line","id":id,"x":mx,"y":my,
        "width":max(p[0] for p in rel) or 1,"height":max(p[1] for p in rel) or 1,
        "strokeColor":stroke,"backgroundColor":"transparent",
        "fillStyle":"solid","strokeWidth":width,"strokeStyle":dash,
        "roughness":1,"opacity":opacity,"angle":0,
        "seed":s,"version":1,"versionNonce":s^0xFFFF,
        "isDeleted":False,"groupIds":[],"boundElements":None,
        "link":None,"locked":False,"points":rel}

def arr(id, x, y, pts, stroke="#3B82F6", width=1.5, opacity=70):
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

def rect(id, x, y, w, h, fill, stroke, opacity=100, sw=1, dash="solid"):
    s = _s(id)
    return {"type":"rectangle","id":id,"x":x,"y":y,"width":w,"height":h,
        "strokeColor":stroke,"backgroundColor":fill,
        "fillStyle":"solid","strokeWidth":sw,"strokeStyle":dash,
        "roughness":1,"opacity":opacity,"angle":0,
        "seed":s,"version":1,"versionNonce":s^0xFFFF,
        "isDeleted":False,"groupIds":[],"boundElements":None,
        "link":None,"locked":False,"roundness":{"type":3}}


elements = []

# Colors
GREEN = "#16A34A"; GREEN_S = "#15803D"
BLUE_F = "#DBEAFE"; BLUE_S = "#3B82F6"
MERGED_F = "#93C5FD"; MERGED_S = "#2563EB"
GRAY_F = "#F3F4F6"; GRAY_S = "#D1D5DB"
PURPLE = "#7C3AED"; PURPLE_LIGHT = "#EDE9FE"; PURPLE_DARK = "#1E1B4B"

# Layout
COLS = {"seed": 100, "gen1": 370, "gen2": 640, "genN": 930, "final": 1220}
CY = 310  # center y for nodes
TL_Y = 490  # timeline y
MEM_Y = 560  # memory box y


# ── Title ────────────────────────────────────────────────────────

elements.append(txt("title", 280, 20, 900, 42,
    "Contrastive Reflection", size=38, color="#1F2937", align="center"))
elements.append(txt("sub", 200, 65, 1060, 20,
    "Same GEPA loop, but reflection gets evidence from historical success/failure pairs",
    size=14, color="#3B82F6", align="center"))


# ── Column headers ───────────────────────────────────────────────

for key, label in [("seed","Seed"),("gen1","Gen 1"),("gen2","Gen 2"),("genN","Gen N"),("final","Final")]:
    elements.append(txt(f"h_{key}", COLS[key]-20, 100, 120, 18, label, size=15, color="#374151"))
    if key != "seed":
        elements.append(ln(f"div_{key}", [[COLS[key]-55, 120],[COLS[key]-55, TL_Y-10]],
            "#E5E7EB", 1, 35, "dashed"))


# ── Seed ─────────────────────────────────────────────────────────

sx, sy = COLS["seed"], CY + 50
elements.append(circ("seed", sx, sy, 22, GREEN, GREEN_S))
elements.append(txt("seed_l", sx-40, sy+28, 100, 16, "seed prompt", size=12, color=GREEN))


# ── Gen 1 ────────────────────────────────────────────────────────

g1 = [(COLS["gen1"], CY-30), (COLS["gen1"], CY+30), (COLS["gen1"], CY+75)]
for i,(x,y) in enumerate(g1):
    elements.append(circ(f"g1_{i}", x, y, 13, BLUE_F, BLUE_S))

# Pruned
elements.append(circ("g1_p", COLS["gen1"]+10, CY+135, 10, GRAY_F, GRAY_S, 45, 1.5, "dashed"))
elements.append(txt("g1_pl", COLS["gen1"]-10, CY+150, 50, 12, "pruned", size=9, color="#9CA3AF"))

# Arrows seed → gen1
for i,(x,y) in enumerate(g1):
    elements.append(arr(f"as_{i}", sx+24, sy, [[0,0],[x-sx-37, y-sy]], BLUE_S, 1.5, 55))
elements.append(arr("as_p", sx+24, sy+5, [[0,0],[COLS["gen1"]+10-sx-34, CY+135-sy-5]], GRAY_S, 1, 25))

# Annotation
elements.append(txt("ann_reflect", sx+55, CY-70, 150, 30,
    "LLM reflects on failures\n+ contrastive evidence", size=11, color=PURPLE))


# ── Gen 2 ────────────────────────────────────────────────────────

g2 = [(COLS["gen2"], CY-55), (COLS["gen2"], CY+5, True), (COLS["gen2"], CY+55)]
for i, node in enumerate(g2):
    x, y = node[0], node[1]
    merged = len(node) > 2
    r = 16 if merged else 13
    f = MERGED_F if merged else BLUE_F
    s = MERGED_S if merged else BLUE_S
    elements.append(circ(f"g2_{i}", x, y, r, f, s))

# Pruned
elements.append(circ("g2_p", COLS["gen2"]+15, CY+135, 10, GRAY_F, GRAY_S, 45, 1.5, "dashed"))

# Arrows gen1 → gen2
elements.append(arr("a_g1_g2_0", g1[0][0]+15, g1[0][1],
    [[0,0],[COLS["gen2"]-g1[0][0]-28, CY-55-g1[0][1]]], BLUE_S, 1.3, 50))
elements.append(arr("a_g1_g2_1", g1[0][0]+15, g1[0][1],
    [[0,0],[COLS["gen2"]-g1[0][0]-29, CY+5-g1[0][1]]], BLUE_S, 1.3, 50))
elements.append(arr("a_g1_g2_2", g1[1][0]+15, g1[1][1],
    [[0,0],[COLS["gen2"]-g1[1][0]-29, CY+5-g1[1][1]]], BLUE_S, 1.3, 50))
elements.append(arr("a_g1_g2_3", g1[1][0]+15, g1[1][1],
    [[0,0],[COLS["gen2"]-g1[1][0]-28, CY+55-g1[1][1]]], BLUE_S, 1.3, 50))
elements.append(arr("a_g1_g2_4", g1[2][0]+15, g1[2][1],
    [[0,0],[COLS["gen2"]-g1[2][0]-28, CY+55-g1[2][1]]], BLUE_S, 1.2, 40))

# Annotation: inject evidence
elements.append(txt("ann_inject", COLS["gen2"]-65, CY+80, 150, 25,
    "inject contrastive\nevidence into reflection", size=11, color=PURPLE))


# ── Gen N ────────────────────────────────────────────────────────

gn = [(COLS["genN"], CY-60), (COLS["genN"], CY-10), (COLS["genN"], CY+40), (COLS["genN"], CY+90)]
for i,(x,y) in enumerate(gn):
    elements.append(circ(f"gn_{i}", x, y, 13, BLUE_F, BLUE_S))

# Dots between gen2 and genN
elements.append(txt("dots", (COLS["gen2"]+COLS["genN"])//2 - 10, CY-5, 30, 20,
    "···", size=22, color="#9CA3AF"))

# Simple arrows gen2 → genN
for i in range(min(len(g2), 3)):
    gx, gy = g2[i][0], g2[i][1]
    nx, ny = gn[i][0], gn[i][1]
    elements.append(arr(f"a_g2_gn_{i}", gx+16, gy,
        [[0,0],[nx-gx-29, ny-gy]], BLUE_S, 1, 30))


# ── Final ────────────────────────────────────────────────────────

fx, fy = COLS["final"], CY - 30
elements.append(circ("best", fx, fy, 26, GREEN, GREEN_S))
elements.append(txt("best_l", fx-30, fy+32, 100, 16, "best prompt", size=13, color=GREEN))

# Green merge arrows
elements.append(arr("a_best1", gn[0][0]+15, gn[0][1],
    [[0,0],[fx-gn[0][0]-41, fy-gn[0][1]]], GREEN_S, 2.5, 75))
elements.append(arr("a_best2", gn[1][0]+15, gn[1][1],
    [[0,0],[fx-gn[1][0]-41, fy-gn[1][1]]], GREEN_S, 2.5, 75))


# ── Timeline axis ────────────────────────────────────────────────

elements.append(ln("tl", [[60, TL_Y],[1340, TL_Y]], "#B0B0B0", 1.5, 50))
elements.append(arr("tl_arr", 1330, TL_Y, [[0,0],[30,0]], "#B0B0B0", 1.5, 50))
elements.append(txt("tl_label", 1340, TL_Y+10, 60, 14, "rollouts", size=11, color="#9CA3AF"))

for tx, label in [(COLS["seed"],"0"),(COLS["gen1"],"~1500"),(COLS["gen2"],"~3500"),
                   (COLS["genN"],"~5500"),(COLS["final"],"~7000")]:
    elements.append(ln(f"tk_{label}", [[tx,TL_Y-5],[tx,TL_Y+5]], "#B0B0B0", 1.5, 50))
    elements.append(txt(f"tkl_{label}", tx-18, TL_Y+8, 45, 12, label, size=10, color="#9CA3AF", align="center"))


# ══════════════════════════════════════════════════════════════════
# THE KEY DIFFERENCE: Contrastive Score Memory
# ══════════════════════════════════════════════════════════════════

# Memory box — dark, prominent
elements.append(rect("mem_box", 80, MEM_Y, 900, 95, PURPLE_DARK, PURPLE, 95, 2))

elements.append(txt("mem_title", 100, MEM_Y+8, 400, 20,
    "Contrastive Score Memory (CPU-only, grows each iteration)",
    size=14, color="#C4B5FD"))

elements.append(txt("mem_content", 100, MEM_Y+30, 860, 50,
    "Cand 0: ex17=0.3, ex42=0.8    Cand 1: ex17=0.9, ex42=0.1    Cand 2: ex17=0.5, ex42=1.0\n"
    "...per-example scores for every candidate ever evaluated — mined for contrastive pairs...",
    size=11, color="#A78BFA"))

# ── Purple arrows: scores flow DOWN, evidence flows UP ───────────

# DOWN arrows: recording scores after each generation
for gx, label in [(COLS["gen1"], "record"), (COLS["gen2"], "record"), (COLS["genN"], "record")]:
    elements.append(arr(f"down_{gx}", gx, TL_Y+5,
        [[0,0],[0, MEM_Y-TL_Y-10]], PURPLE, 1.5, 50))

# UP arrows: evidence injected into reflection
for gx in [COLS["gen2"], COLS["genN"]]:
    elements.append(arr(f"up_{gx}", gx+30, MEM_Y-5,
        [[0,0],[0, -(MEM_Y-TL_Y-10)]], PURPLE, 1.5, 50))

# Labels on arrows
elements.append(txt("down_l", COLS["gen1"]-55, TL_Y+15, 60, 12,
    "record\nscores", size=9, color=PURPLE))
elements.append(txt("up_l", COLS["gen2"]+35, TL_Y+15, 70, 12,
    "inject\nevidence", size=9, color=PURPLE))


# ── Zoomed callout: what a contrastive pair looks like ───────────

cx_call = 1040; cy_call = MEM_Y + 15
elements.append(rect("callout_bg", cx_call, cy_call, 320, 75,
    "#FAFAFA", "#E5E7EB", 90, 1))

elements.append(txt("call_title", cx_call+8, cy_call+5, 300, 14,
    "What the LLM sees (contrastive pair):", size=11, color="#374151"))
elements.append(txt("call_ex", cx_call+8, cy_call+22, 300, 14,
    '"On example 17, Candidate 1 scored 0.9', size=10, color=PURPLE))
elements.append(txt("call_ex2", cx_call+8, cy_call+36, 300, 14,
    ' vs your 0.3 (gap = 0.6).', size=10, color=PURPLE))
elements.append(txt("call_ex3", cx_call+8, cy_call+52, 300, 14,
    ' That candidate used: Be specific about..."', size=10, color="#6B7280"))


# ── Benefits callout ─────────────────────────────────────────────

elements.append(txt("b1", 1100, CY+100, 200, 16, "Zero extra LLM cost", size=13, color=GREEN))
elements.append(txt("b2", 1100, CY+118, 200, 16, "Pure CPU memory lookup", size=13, color=GREEN))
elements.append(txt("b3", 1100, CY+136, 200, 16, "Same budget as GEPA", size=13, color=GREEN))


# ── Legend ────────────────────────────────────────────────────────

ly = MEM_Y + 110
elements.append(rect("leg_bg", 80, ly, 700, 50, "#FAFAFA", "#E5E7EB", 70, 1))
elements.append(txt("leg_t", 90, ly+5, 50, 14, "Legend:", size=11, color="#374151"))

elements.append(circ("lc1", 160, ly+14, 7, BLUE_F, BLUE_S))
elements.append(txt("lt1", 172, ly+6, 100, 12, "Pareto candidate", size=10, color="#6B7280"))

elements.append(circ("lc2", 300, ly+14, 7, GRAY_F, GRAY_S, 50, 1.5, "dashed"))
elements.append(txt("lt2", 312, ly+6, 80, 12, "Pruned", size=10, color="#6B7280"))

elements.append(circ("lc3", 410, ly+14, 8, MERGED_F, MERGED_S))
elements.append(txt("lt3", 422, ly+6, 100, 12, "Merged offspring", size=10, color="#6B7280"))

elements.append(ln("lc4", [[510, ly+14],[530, ly+14]], PURPLE, 2, 70))
elements.append(txt("lt4", 535, ly+6, 130, 12, "Contrastive flow", size=10, color="#6B7280"))

elements.append(txt("leg_foot", 90, ly+28, 650, 12,
    "Blue arrows = LLM reflection (mutate)  |  Purple arrows = contrastive memory read/write",
    size=9, color="#9CA3AF"))


# ── Render ───────────────────────────────────────────────────────

doc = {
    "type":"excalidraw","version":2,"source":"plotcraft",
    "elements":elements,
    "appState":{"viewBackgroundColor":"#FFFFFF","gridSize":None},
    "files":{},
}

out = "examples/renders/gepa_contrastive_tree.excalidraw"
with open(out, "w") as f:
    json.dump(doc, f, indent=2)

import subprocess
r = subprocess.run(
    ["uv","run","python","skills/plotcraft-diagram/references/render_excalidraw.py",out],
    capture_output=True, text=True, timeout=30)
print(f"-> {r.stdout.strip()}" if r.returncode==0 else f"ERROR: {r.stderr}")
