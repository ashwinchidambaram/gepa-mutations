"""GEPA evolutionary tree: seed prompt evolves across generations.

Spatial composition with:
- Horizontal rollout timeline axis
- Candidate nodes at different heights (diversity)
- Branching arrows (reflection/mutation)
- Pruned nodes (gray dashed)
- Merged offspring (filled blue)
- Legend box
"""

import json
import math
import random
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

# ── Layout constants ─────────────────────────────────────────────

# Column x positions (generations)
COLS = {
    "seed": 100,
    "gen1": 380,
    "gen2": 660,
    "genN": 960,
    "final": 1260,
}

# Vertical center for the "main flow"
CENTER_Y = 340
TIMELINE_Y = 530

# Node sizes
SEED_R = 22
CANDIDATE_R = 14
MERGED_R = 17
PRUNED_R = 12
BEST_R = 28

# Colors
GREEN_FILL = "#16A34A"
GREEN_STROKE = "#15803D"
BLUE_FILL = "#DBEAFE"
BLUE_STROKE = "#3B82F6"
MERGED_FILL = "#93C5FD"
MERGED_STROKE = "#2563EB"
GRAY_FILL = "#F3F4F6"
GRAY_STROKE = "#D1D5DB"
ANNOTATION = "#3B82F6"


# ── Title ────────────────────────────────────────────────────────

elements.append(txt("title", 300, 25, 800, 45,
    "GEPA: Evolutionary Prompt Optimization",
    size=36, color="#1F2937", align="center"))
elements.append(txt("sub", 420, 72, 550, 20,
    "ICLR 2026 Oral  ·  arXiv:2507.19457",
    size=14, color="#6B7280", align="center"))


# ── Column headers ───────────────────────────────────────────────

headers = [
    ("seed", "Seed"), ("gen1", "Generation 1"), ("gen2", "Generation 2"),
    ("genN", "Generation N"), ("final", "Final"),
]
for key, label in headers:
    elements.append(txt(f"h_{key}", COLS[key]-30, 110, 180, 20,
        label, size=16, color="#374151"))

# Vertical dashed dividers between generations
for key in ["gen1", "gen2", "genN", "final"]:
    dx = COLS[key] - 60
    elements.append(ln(f"div_{key}", [[dx, 130], [dx, TIMELINE_Y-10]],
        "#E5E7EB", 1, 40, "dashed"))


# ── Seed node ────────────────────────────────────────────────────

sx, sy = COLS["seed"], CENTER_Y + 60
elements.append(circ("seed", sx, sy, SEED_R, GREEN_FILL, GREEN_STROKE))
elements.append(txt("seed_label", sx-40, sy+30, 100, 18,
    "seed prompt", size=13, color=GREEN_FILL))


# ── Generation 1 nodes ───────────────────────────────────────────

g1_nodes = [
    ("g1_a", COLS["gen1"], CENTER_Y - 40, CANDIDATE_R, BLUE_FILL, BLUE_STROKE),
    ("g1_b", COLS["gen1"], CENTER_Y + 30, CANDIDATE_R, BLUE_FILL, BLUE_STROKE),
    ("g1_c", COLS["gen1"], CENTER_Y + 80, CANDIDATE_R, BLUE_FILL, BLUE_STROKE),
]
for id, x, y, r, fill, stroke in g1_nodes:
    elements.append(circ(id, x, y, r, fill, stroke))

# Pruned node
elements.append(circ("g1_pruned", COLS["gen1"]+10, CENTER_Y+150,
    PRUNED_R, GRAY_FILL, GRAY_STROKE, opacity=50, dash="dashed"))
elements.append(txt("g1_pruned_l", COLS["gen1"]-15, CENTER_Y+168, 60, 14,
    "pruned", size=10, color="#9CA3AF"))

# Arrows from seed to gen1
for id, x, y, r, _, _ in g1_nodes:
    elements.append(arr(f"a_seed_{id}", sx+SEED_R+4, sy,
        [[0, 0], [x-sx-SEED_R-r-8, y-sy]], BLUE_STROKE, 1.5, 60))
# Arrow to pruned
elements.append(arr("a_seed_pruned", sx+SEED_R+4, sy+5,
    [[0, 0], [COLS["gen1"]+10-sx-SEED_R-8, CENTER_Y+150-sy-5]],
    GRAY_STROKE, 1, 30))

# Annotation: LLM reflects
elements.append(txt("ann_reflect", COLS["seed"]+60, CENTER_Y-80, 130, 30,
    "LLM reflects\non failures", size=12, color=ANNOTATION))


# ── Generation 2 nodes ───────────────────────────────────────────

g2_nodes = [
    ("g2_a", COLS["gen2"], CENTER_Y - 60, CANDIDATE_R, BLUE_FILL, BLUE_STROKE),
    ("g2_b", COLS["gen2"], CENTER_Y + 10, MERGED_R, MERGED_FILL, MERGED_STROKE),  # merged
    ("g2_c", COLS["gen2"], CENTER_Y + 70, CANDIDATE_R, BLUE_FILL, BLUE_STROKE),
]
for id, x, y, r, fill, stroke in g2_nodes:
    elements.append(circ(id, x, y, r, fill, stroke))

# Pruned node
elements.append(circ("g2_pruned", COLS["gen2"]+20, CENTER_Y+150,
    PRUNED_R, GRAY_FILL, GRAY_STROKE, opacity=50, dash="dashed"))
elements.append(txt("g2_pruned_l", COLS["gen2"]-5, CENTER_Y+168, 60, 14,
    "pruned", size=10, color="#9CA3AF"))

# Arrows from gen1 to gen2
# g1_a fans to g2_a and g2_b
elements.append(arr("a_g1a_g2a", COLS["gen1"]+CANDIDATE_R+4, CENTER_Y-40,
    [[0, 0], [COLS["gen2"]-COLS["gen1"]-CANDIDATE_R*2-8, -20]], BLUE_STROKE, 1.5, 55))
elements.append(arr("a_g1a_g2b", COLS["gen1"]+CANDIDATE_R+4, CENTER_Y-40,
    [[0, 0], [COLS["gen2"]-COLS["gen1"]-CANDIDATE_R-MERGED_R-8, 50]], BLUE_STROKE, 1.5, 55))

# g1_b to g2_b (merge arrow) and g2_c
elements.append(arr("a_g1b_g2b", COLS["gen1"]+CANDIDATE_R+4, CENTER_Y+30,
    [[0, 0], [COLS["gen2"]-COLS["gen1"]-CANDIDATE_R-MERGED_R-8, -20]], BLUE_STROKE, 1.5, 55))
elements.append(arr("a_g1b_g2c", COLS["gen1"]+CANDIDATE_R+4, CENTER_Y+30,
    [[0, 0], [COLS["gen2"]-COLS["gen1"]-CANDIDATE_R*2-8, 40]], BLUE_STROKE, 1.5, 55))

# g1_c carry forward
elements.append(arr("a_g1c_g2c", COLS["gen1"]+CANDIDATE_R+4, CENTER_Y+80,
    [[0, 0], [COLS["gen2"]-COLS["gen1"]-CANDIDATE_R*2-8, -10]], BLUE_STROKE, 1.2, 40))

# Pruned arrow
elements.append(arr("a_g1c_pruned2", COLS["gen1"]+CANDIDATE_R+4, CENTER_Y+85,
    [[0, 0], [COLS["gen2"]+20-COLS["gen1"]-CANDIDATE_R-8, 65]], GRAY_STROKE, 1, 25))

# Annotation: merge
elements.append(txt("ann_merge", COLS["gen2"]-50, CENTER_Y-100, 120, 25,
    "merge two\nparents", size=12, color=ANNOTATION))

# Annotation: Pareto
elements.append(txt("ann_pareto", COLS["gen1"]+60, CENTER_Y+120, 150, 25,
    "Pareto selects\ndiverse frontier", size=12, color=ANNOTATION))


# ── Generation N nodes ───────────────────────────────────────────

gn_nodes = [
    ("gn_a", COLS["genN"], CENTER_Y - 70, CANDIDATE_R, BLUE_FILL, BLUE_STROKE),
    ("gn_b", COLS["genN"], CENTER_Y - 10, CANDIDATE_R, BLUE_FILL, BLUE_STROKE),
    ("gn_c", COLS["genN"], CENTER_Y + 50, CANDIDATE_R, BLUE_FILL, BLUE_STROKE),
    ("gn_d", COLS["genN"], CENTER_Y + 110, CANDIDATE_R, BLUE_FILL, BLUE_STROKE),
]
for id, x, y, r, fill, stroke in gn_nodes:
    elements.append(circ(id, x, y, r, fill, stroke))

# Arrows from gen2 to genN (dashed to indicate "..." skip)
for g2id, g2x, g2y, g2r, _, _ in g2_nodes:
    # Each gen2 node sends to 1-2 genN nodes
    for gnid, gnx, gny, gnr, _, _ in gn_nodes[:2]:
        elements.append(arr(f"a_{g2id}_{gnid}",
            g2x+g2r+4, g2y,
            [[0, 0], [gnx-g2x-g2r-gnr-8, gny-g2y]],
            BLUE_STROKE, 1, 35))
        break  # just one arrow per gen2 node for clarity

# Dotted "..." indicators between gen2 and genN
elements.append(txt("dots", (COLS["gen2"]+COLS["genN"])//2 - 10, CENTER_Y-5, 30, 20,
    "···", size=24, color="#9CA3AF"))


# ── Final / Best node ────────────────────────────────────────────

fx, fy = COLS["final"], CENTER_Y - 30
elements.append(circ("best", fx, fy, BEST_R, GREEN_FILL, GREEN_STROKE))
elements.append(txt("best_label", fx-30, fy+35, 100, 18,
    "best prompt", size=14, color=GREEN_FILL))

# Merge arrow from top genN nodes to best (thick green)
# Two parents merge into the best
elements.append(arr("a_gn_best1",
    COLS["genN"]+CANDIDATE_R+4, CENTER_Y-70,
    [[0, 0], [fx-COLS["genN"]-CANDIDATE_R-BEST_R-8, fy-(CENTER_Y-70)]],
    GREEN_STROKE, 2.5, 80))
elements.append(arr("a_gn_best2",
    COLS["genN"]+CANDIDATE_R+4, CENTER_Y-10,
    [[0, 0], [fx-COLS["genN"]-CANDIDATE_R-BEST_R-8, fy-(CENTER_Y-10)]],
    GREEN_STROKE, 2.5, 80))


# ── Timeline axis ────────────────────────────────────────────────

# Horizontal line
elements.append(ln("timeline", [[60, TIMELINE_Y], [1350, TIMELINE_Y]],
    "#9CA3AF", 1.5, 60))
# Arrow at end
elements.append(arr("timeline_arr", 1340, TIMELINE_Y, [[0,0],[30,0]], "#9CA3AF", 1.5, 60))
elements.append(txt("timeline_label", 1360, TIMELINE_Y-10, 60, 16,
    "rollouts", size=12, color="#9CA3AF"))

# Tick marks and labels
ticks = [
    (COLS["seed"], "0"),
    (COLS["gen1"], "~1500"),
    (COLS["gen2"], "~3500"),
    (COLS["genN"], "~5500"),
    (COLS["final"], "~7000"),
]
for tx, label in ticks:
    elements.append(ln(f"tick_{label}", [[tx, TIMELINE_Y-6], [tx, TIMELINE_Y+6]],
        "#9CA3AF", 1.5, 60))
    elements.append(txt(f"tickl_{label}", tx-20, TIMELINE_Y+10, 50, 14,
        label, size=11, color="#9CA3AF", align="center"))


# ── Legend box ───────────────────────────────────────────────────

lx, ly = 70, TIMELINE_Y + 45
elements.append(rect("legend_bg", lx, ly, 650, 55,
    "#FAFAFA", "#E5E7EB", 80, 1))

elements.append(txt("leg_title", lx+10, ly+6, 55, 14,
    "Legend:", size=12, color="#374151"))

# Pareto candidate
elements.append(circ("leg_pareto", lx+100, ly+16, 8, BLUE_FILL, BLUE_STROKE))
elements.append(txt("leg_pareto_t", lx+114, ly+8, 120, 14,
    "Pareto candidate", size=11, color="#6B7280"))

# Pruned
elements.append(circ("leg_pruned", lx+260, ly+16, 8, GRAY_FILL, GRAY_STROKE, 50, 1.5, "dashed"))
elements.append(txt("leg_pruned_t", lx+274, ly+8, 100, 14,
    "Pruned / weak", size=11, color="#6B7280"))

# Merged
elements.append(circ("leg_merged", lx+400, ly+16, 9, MERGED_FILL, MERGED_STROKE))
elements.append(txt("leg_merged_t", lx+415, ly+8, 120, 14,
    "Merged offspring", size=11, color="#6B7280"))

# Footer line
elements.append(txt("leg_footer", lx+10, ly+30, 620, 14,
    "Arrows = LLM reflection (mutate) or Pareto selection (carry forward)",
    size=10, color="#9CA3AF"))


# ── Render ───────────────────────────────────────────────────────

doc = {
    "type": "excalidraw", "version": 2, "source": "plotcraft",
    "elements": elements,
    "appState": {"viewBackgroundColor": "#FFFFFF", "gridSize": None},
    "files": {},
}

out = "examples/renders/gepa_evolution_tree.excalidraw"
with open(out, "w") as f:
    json.dump(doc, f, indent=2)

import subprocess
r = subprocess.run(
    ["uv", "run", "python", "skills/plotcraft-diagram/references/render_excalidraw.py", out],
    capture_output=True, text=True, timeout=30)
print(f"-> {r.stdout.strip()}" if r.returncode == 0 else f"ERROR: {r.stderr}")
