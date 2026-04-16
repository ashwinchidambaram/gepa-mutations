"""Synaptic Pruning: one-shot compression pipeline.

Visual story: a fat prompt gets dissected, measured, cut, and strengthened.
Each section shown as a colored block that transforms across the pipeline.
NOT iterative — a linear surgical pipeline.
"""

import json
import zlib


def _s(n): return zlib.adler32(n.encode())

def rect(id, x, y, w, h, fill, stroke, opacity=100, sw=1.5, dash="solid"):
    s = _s(id)
    return {"type":"rectangle","id":id,"x":x,"y":y,"width":w,"height":h,
        "strokeColor":stroke,"backgroundColor":fill,
        "fillStyle":"solid","strokeWidth":sw,"strokeStyle":dash,
        "roughness":1,"opacity":opacity,"angle":0,
        "seed":s,"version":1,"versionNonce":s^0xFFFF,
        "isDeleted":False,"groupIds":[],"boundElements":None,
        "link":None,"locked":False,"roundness":{"type":3}}

def circ(id, x, y, r, fill, stroke, opacity=100, sw=2):
    s = _s(id)
    return {"type":"ellipse","id":id,"x":x-r,"y":y-r,"width":r*2,"height":r*2,
        "strokeColor":stroke,"backgroundColor":fill,
        "fillStyle":"solid","strokeWidth":sw,"strokeStyle":"solid",
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

# Colors
RED_F = "#FEE2E2"; RED_S = "#EF4444"      # critical / load-bearing
GRAY_F = "#F3F4F6"; GRAY_S = "#D1D5DB"    # prunable
GREEN_F = "#DCFCE7"; GREEN_S = "#16A34A"   # strengthened / final
BLUE_F = "#DBEAFE"; BLUE_S = "#3B82F6"    # initial / neutral
ORANGE_F = "#FEF3C7"; ORANGE_S = "#F59E0B" # harmful
PURPLE = "#7C3AED"

# Layout — 5 stages horizontal
STAGE_X = [80, 330, 610, 900, 1180]
CY = 240  # center y — close to headers
BLOCK_W = 130  # prompt block width


# ── Title ────────────────────────────────────────────────────────

elements.append(txt("title", 280, 20, 900, 42,
    "Synaptic Pruning", size=38, color="#1F2937", align="center"))
elements.append(txt("sub", 180, 72, 1100, 18,
    "One-shot compression: generate overspecified, measure each section's impact, cut dead weight, strengthen what matters",
    size=13, color="#6B7280", align="center"))


# ── Stage headers ────────────────────────────────────────────────

stages = [
    ("1. Generate (x3)", BLUE_S),
    ("2. Ablate Each", ORANGE_S),
    ("3. Prune", RED_S),
    ("4. Strengthen", GREEN_S),
    ("5. Final Prompt", GREEN_S),
]
for i, (label, color) in enumerate(stages):
    elements.append(txt(f"sh_{i}", STAGE_X[i], 120, 180, 20,
        label, size=15, color=color))


# ══════════════════════════════════════════════════════════════════
# STAGE 1: Generate 3 over-specified prompts
# ══════════════════════════════════════════════════════════════════

sx = STAGE_X[0]
# Three tall prompt blocks (over-specified = tall)
prompts = [
    (sx, CY-80, 0.65, "best"),
    (sx+52, CY-65, 0.58, ""),
    (sx+104, CY-70, 0.62, ""),
]
for i, (px, py, score, note) in enumerate(prompts):
    h = 160 + i*5  # tall = overspecified
    elements.append(rect(f"p{i}", px, py, 42, h, BLUE_F, BLUE_S, 80))
    elements.append(txt(f"ps{i}", px+2, py+h+5, 42, 14,
        f"{score}", size=10, color=BLUE_S, align="center"))

# Best indicator
elements.append(txt("best_ind", sx-5, CY+95, 50, 14,
    "best", size=10, color=GREEN_S))
elements.append(txt("s1_note", sx-5, CY+120, 160, 30,
    "~500 words each\n4 labeled examples", size=10, color="#6B7280"))

# Score label
elements.append(txt("score_l", sx+30, CY-100, 80, 14,
    "score:", size=9, color="#9CA3AF"))


# ══════════════════════════════════════════════════════════════════
# STAGE 2: Ablate — show sections with delta scores
# ══════════════════════════════════════════════════════════════════

ax = STAGE_X[1]
sections = [
    ("Section A", -0.08, "critical", RED_F, RED_S),
    ("Section B", -0.12, "critical", RED_F, RED_S),
    ("Section C", -0.002, "prunable", GRAY_F, GRAY_S),
    ("Section D", -0.03, "neutral", "#F9FAFB", "#9CA3AF"),
    ("Section E", +0.01, "harmful", ORANGE_F, ORANGE_S),
]

sy = CY - 90
for i, (name, delta, kind, fill, stroke) in enumerate(sections):
    bh = 38
    elements.append(rect(f"sec_{i}", ax, sy, BLOCK_W, bh, fill, stroke, 90 if kind != "prunable" else 55))
    elements.append(txt(f"secn_{i}", ax+5, sy+3, 80, 14,
        name, size=10, color="#374151"))
    # Delta score — inside block, right-aligned
    delta_str = f"{delta:+.3f}" if abs(delta) < 0.01 else f"{delta:+.2f}"
    elements.append(txt(f"secd_{i}", ax+BLOCK_W-48, sy+3, 44, 14,
        delta_str, size=9, color=stroke, align="right"))
    # Kind label — inside block, under the name
    elements.append(txt(f"seck_{i}", ax+5, sy+20, 70, 12,
        kind, size=8, color="#9CA3AF"))
    sy += bh + 4

elements.append(txt("abl_note", ax, sy+10, 180, 25,
    "Remove each section,\nmeasure score drop on 40 examples",
    size=10, color="#6B7280"))


# ══════════════════════════════════════════════════════════════════
# STAGE 3: Prune — remove prunable sections
# ══════════════════════════════════════════════════════════════════

px = STAGE_X[2]
# Show remaining sections (A, B, D)
kept = [
    ("Section A", RED_F, RED_S),
    ("Section B", RED_F, RED_S),
    ("Section D", "#F9FAFB", "#9CA3AF"),
]
py = CY - 50
for i, (name, fill, stroke) in enumerate(kept):
    bh = 32
    elements.append(rect(f"kept_{i}", px, py, BLOCK_W, bh, fill, stroke, 90))
    elements.append(txt(f"keptn_{i}", px+5, py+5, 80, 14,
        name, size=10, color="#374151"))
    py += bh + 6

# Crossed-out sections (C and E)
elements.append(txt("cut_label", px, py+15, 140, 25,
    "C, E removed\n(prunable + harmful)", size=10, color="#9CA3AF"))

# Visual: strikethrough blocks for removed sections
for i, name in enumerate(["C", "E"]):
    ry = py + 55 + i*25
    elements.append(rect(f"cut_{i}", px+20, ry, 80, 18, GRAY_F, GRAY_S, 30))
    elements.append(txt(f"cutn_{i}", px+30, ry+2, 60, 14,
        f"Section {name}", size=9, color="#D1D5DB"))
    # Strikethrough line
    elements.append(ln(f"strike_{i}", [[px+22, ry+9],[px+98, ry+9]], RED_S, 1.5, 50))

elements.append(txt("prune_note", px, py+115, 150, 14,
    "~2000 → ~1200 chars", size=10, color=RED_S))


# ══════════════════════════════════════════════════════════════════
# STAGE 4: Strengthen load-bearing sections
# ══════════════════════════════════════════════════════════════════

stx = STAGE_X[3]
strengthened = [
    ("Section A", GREEN_F, GREEN_S, True),
    ("Section B", GREEN_F, GREEN_S, True),
    ("Section D", "#F9FAFB", "#9CA3AF", False),
]
sty = CY - 50
for i, (name, fill, stroke, is_strong) in enumerate(strengthened):
    bh = 32
    sw = 2 if is_strong else 1
    elements.append(rect(f"str_{i}", stx, sty, BLOCK_W, bh, fill, stroke, 90, sw))
    label = f"{name} ✓" if is_strong else name
    elements.append(txt(f"strn_{i}", stx+5, sty+5, 120, 14,
        label, size=10, color="#374151"))
    sty += bh + 6

elements.append(txt("str_note", stx, sty+10, 170, 25,
    "LLM rewrites critical sections\nfor clarity and specificity",
    size=10, color="#6B7280"))


# ══════════════════════════════════════════════════════════════════
# STAGE 5: Final prompt
# ══════════════════════════════════════════════════════════════════

fx = STAGE_X[4]
# Single clean block — shorter than the originals
elements.append(rect("final", fx, CY-35, BLOCK_W+20, 80, GREEN_F, GREEN_S, 100, 2))
elements.append(txt("final_t", fx+10, CY-25, 130, 30,
    "Minimal,\nEffective\nPrompt", size=14, color=GREEN_S))
elements.append(txt("final_chars", fx, CY+55, 160, 14,
    "~800 chars (from ~2000)", size=11, color=GREEN_S))

# Efficiency callout
elements.append(txt("eff1", fx, CY+85, 200, 16,
    "25-460 rollouts", size=14, color=GREEN_S))
elements.append(txt("eff2", fx, CY+105, 200, 16,
    "15-100x fewer than GEPA", size=12, color="#6B7280"))


# ── Arrows between stages ────────────────────────────────────────

# Arrows between stages — positioned to avoid text
elements.append(arr("sa_0", STAGE_X[0]+155, CY, [[0,0],[STAGE_X[1]-STAGE_X[0]-165, 0]], BLUE_S, 2, 50))
elements.append(arr("sa_1", STAGE_X[1]+BLOCK_W+10, CY, [[0,0],[STAGE_X[2]-STAGE_X[1]-BLOCK_W-20, 0]], ORANGE_S, 2, 50))
elements.append(arr("sa_2", STAGE_X[2]+BLOCK_W+10, CY, [[0,0],[STAGE_X[3]-STAGE_X[2]-BLOCK_W-20, 0]], RED_S, 2, 50))
# Stage 4→5: same height as other arrows
elements.append(arr("sa_3", STAGE_X[3]+BLOCK_W+10, CY, [[0,0],[STAGE_X[4]-STAGE_X[3]-BLOCK_W-20, 0]], GREEN_S, 2, 50))


# ── Pipeline label at bottom ─────────────────────────────────────

elements.append(ln("pipeline_line", [[60, CY+200],[1400, CY+200]],
    "#E5E7EB", 1, 30))
elements.append(txt("pipeline_l", 500, CY+208, 400, 16,
    "pipeline (no iteration)", size=13, color="#9CA3AF", align="center"))


# ── Legend ────────────────────────────────────────────────────────

ly = CY + 240
elements.append(rect("leg_bg", 80, ly, 750, 45, "#FAFAFA", "#E5E7EB", 70, 1))
elements.append(txt("leg_t", 90, ly+5, 50, 14, "Legend:", size=11, color="#374151"))

items = [
    (155, RED_F, RED_S, "Load-bearing (Δ > 0.05)"),
    (350, GRAY_F, GRAY_S, "Prunable (Δ < 0.01)"),
    (530, ORANGE_F, ORANGE_S, "Harmful (Δ > 0)"),
    (700, GREEN_F, GREEN_S, "Strengthened"),
]
for lx, fill, stroke, label in items:
    elements.append(rect(f"leg_r{lx}", lx, ly+8, 16, 16, fill, stroke, 90, 1.5))
    elements.append(txt(f"leg_lt{lx}", lx+22, ly+8, 150, 14, label, size=10, color="#6B7280"))

elements.append(txt("leg_foot", 90, ly+28, 700, 12,
    "Thresholds: load-bearing > 0.05 | prunable < 0.01 | interaction check > 0.03 | max prune = 50% of sections",
    size=9, color="#9CA3AF"))


# ── Render ───────────────────────────────────────────────────────

doc = {
    "type":"excalidraw","version":2,"source":"plotcraft",
    "elements":elements,
    "appState":{"viewBackgroundColor":"#FFFFFF","gridSize":None},
    "files":{},
}

out = "examples/renders/gepa_synaptic_pruning_tree.excalidraw"
with open(out, "w") as f:
    json.dump(doc, f, indent=2)

import subprocess
r = subprocess.run(
    ["uv","run","python","skills/plotcraft-diagram/references/render_excalidraw.py",out],
    capture_output=True, text=True, timeout=30)
print(f"-> {r.stdout.strip()}" if r.returncode==0 else f"ERROR: {r.stderr}")
