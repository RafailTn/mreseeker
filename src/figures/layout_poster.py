#!/usr/bin/env python3
"""
Lay out the figure panels and introduction of results/mreseeker.svg, in place.

The poster is edited by hand in Inkscape between runs, so this script treats the
current file as the source of truth and only replaces what it generates itself:
the panel cards, the panel groups and the introduction card. Everything else -
the background gradient, the title and its glow, and any artwork added by hand -
is kept. Hand-added artwork that sits beside a panel moves with that panel.

Panels 2 and 3 keep the groups already in the file. Panels 1, 4 and 5 are
(re)imported from results/figures, with their ids prefixed so matplotlib's
repeated names (figure_1, axes_1, clip paths) cannot collide.

The page is A0, declared on top of an A4 viewBox: the proportions are identical,
so every coordinate, font size and stroke simply scales by 4.

    python3 src/figures/layout_poster.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from lxml import etree
from PIL import ImageFont

REPO = Path(__file__).resolve().parents[2]
POSTER = REPO / "results" / "mreseeker.svg"
INTRO = Path(__file__).resolve().parent / "poster_intro.txt"
SVG = "http://www.w3.org/2000/svg"
N = lambda t: f"{{{SVG}}}{t}"

PAGE_W = 210.0                     # viewBox units (mm on A4, x4 on A0)
TOP, BOTTOM = 38.5, 291.5          # the title glow ends ~35.7
PAD, GAP_MIN, COL_GAP, RADIUS = 2.2, 4.2, 8.0, 2.4
FULL_W = 194.0                     # width of the full-page blocks

KEPT: dict = {}
IMPORTED = {1: REPO / "results/figures/panel1_icons.svg",
            2: REPO / "results/figures/panel2_stacked.svg",
            4: REPO / "results/figures/panel4_narrow8.svg",
            5: REPO / "results/figures/panel5_narrow8.svg"}
# superseded panel groups from earlier layouts
STALE = ["figure_1-96", "figure_1-2", "figure_1-8", "figure_1", "figure_1-9"]
OLD_GENERATED_TEXT = True
BLOCKS = [[1], ["intro"], [2, 3], [4, 5]]
FULL_WIDTH = {1, "intro"}          # blocks that span the page, not the column

FONT_REG = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
BODY, LEAD, HEAD = 1.95, 1.40, 3.4
TPAD, TCOL_GAP, TEXT_INK = 4.6, 7.0, "#1c1b19"

GENERATED = re.compile(r"^(card\w*|intro\w*|order\d|p\dn_figure_1|cap\d|methods\w*|results\w*)$")
OVERFLOW_OK = False
PREFIXED = re.compile(r"^p\dn_")
URL = re.compile(r"url\(#([^)]+)\)")


def bbox_mm(gid: str) -> tuple[float, float, float, float] | None:
    """Rendered bounding box of an element, via Inkscape, in viewBox units."""
    out = subprocess.run(["inkscape", str(POSTER), f"--query-id={gid}", "--query-x",
                          "--query-y", "--query-width", "--query-height"],
                         capture_output=True, text=True).stdout.split()
    if len(out) != 4:
        return None
    return tuple(float(v) * 25.4 / 96 for v in out)   # px at 96 dpi -> mm


def main() -> None:
    tree = etree.parse(str(POSTER))
    root = tree.getroot()
    root.set("width", "210mm"); root.set("height", "297mm")   # query in A4 units
    tree.write(str(POSTER), xml_declaration=True, encoding="UTF-8", standalone=False)

    layer, defs = root.find(N("g")), root.find(N("defs"))
    old_cards = {c.get("id"): float(c.get("y")) for c in layer
                 if (c.get("id") or "").startswith("card")}
    old_x0 = next((float(c.get("x")) for c in layer if c.get("id") == "card2"), None)

    # Hand-added artwork beside a panel: remember which card it belongs to.
    anchored = []
    for el in list(layer):
        gid = el.get("id") or ""
        if etree.QName(el).localname != "g" or GENERATED.match(gid) or \
                gid in STALE or gid in {v[0] for v in KEPT.values()}:
            continue
        bb = bbox_mm(gid)
        if bb is None or bb[1] + bb[3] / 2 < TOP:        # top decorations stay put
            continue
        cy = bb[1] + bb[3] / 2
        near = min(old_cards, key=lambda k: abs(old_cards[k] - cy)) if old_cards else None
        if near:
            anchored.append((el, near, bb))

    # Remove everything this script generates, plus defs it imported last time.
    kept = {k: layer.find(f"{{{SVG}}}g[@id='{gid}']") for k, (gid, _, _) in KEPT.items()}
    for el in list(layer):
        gid = el.get("id") or ""
        if GENERATED.match(gid) or gid in STALE or el in kept.values():
            layer.remove(el)
    for el in list(defs):
        gid = el.get("id") or ""
        if PREFIXED.match(gid) or gid == "cardShadow":
            defs.remove(el)
    art = [a for a, _, _ in anchored]
    for a in art:
        layer.remove(a)

    panels = {k: (g, w, h) for k, ((_, w, h), g) in
              ((k, (KEPT[k], kept[k])) for k in KEPT)}

    def import_panel(path: Path, prefix: str):
        src = etree.parse(str(path)).getroot()
        _, _, w, h = (float(v) for v in src.get("viewBox").split())
        ren = {e.get("id"): prefix + e.get("id") for e in src.iter()
               if isinstance(e.tag, str) and e.get("id")}
        for e in src.iter():
            if not isinstance(e.tag, str):
                continue
            if e.get("id"):
                e.set("id", ren[e.get("id")])
            for key, val in list(e.attrib.items()):
                if "url(#" in val:
                    e.set(key, URL.sub(lambda m: f"url(#{ren.get(m.group(1), m.group(1))})", val))
                if key.endswith("href") and val.startswith("#") and val[1:] in ren:
                    e.set(key, "#" + ren[val[1:]])
        for d in src.findall(N("defs")):
            for c in list(d):
                # matplotlib's <style> is a global `*` rule: inside the poster it
                # would restyle every stroke on the page.
                if etree.QName(c).localname != "style":
                    defs.append(c)
        return src.find(N("g")), w, h

    for key, path in IMPORTED.items():
        panels[key] = import_panel(path, f"p{key}n_")

    # -- text helpers -----------------------------------------------------------
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import poster_text as pt

    font = ImageFont.truetype(FONT_REG, 1000)
    font_b = ImageFont.truetype(FONT_REG.replace("Regular", "Bold"), 1000)

    def rich_words(text):
        """Split on spaces; words inside **...** are flagged bold."""
        out, bold = [], False
        for w in text.split():
            start = w.startswith("**")
            if start:
                bold, w = True, w[2:]
            end = w.endswith("**")
            if end:
                w = w[:-2]
            out.append((w, bold))
            if end:
                bold = False
        return out

    def wrap_rich(text, width, size):
        lines, cur, cur_w = [], [], 0.0
        space = font.getlength(" ") * size / 1000
        for w, b in rich_words(text):
            ww = (font_b if b else font).getlength(w) * size / 1000
            add = ww + (space if cur else 0)
            if cur and cur_w + add > width:
                lines.append(cur); cur, cur_w = [(w, b)], ww
            else:
                cur.append((w, b)); cur_w += add
        return lines + ([cur] if cur else [])

    def wrap(text, width, size):
        lines, cur = [], ""
        for word in text.split():
            trial = f"{cur} {word}".strip()
            if cur and font.getlength(trial) * size / 1000 > width:
                lines.append(cur); cur = word
            else:
                cur = trial
        return lines + ([cur] if cur else [])

    def put_lines(tid, lines, x, y_first, size, colour=TEXT_INK, italic=False):
        style = f"font-size:{size}px;font-family:'Noto Sans',sans-serif;fill:{colour}"
        if italic:
            style += ";font-style:italic"
        t = etree.SubElement(layer, N("text"), id=tid, style=style)
        for i, line in enumerate(lines):
            ts = etree.SubElement(t, N("tspan"), x=f"{x:.3f}",
                                  y=f"{y_first + i * size * LEAD:.3f}")
            ts.text = line

    def heading(tid, text, x, y):
        h = etree.SubElement(layer, N("text"), id=tid, x=f"{x:.3f}", y=f"{y:.3f}",
                             style=(f"font-weight:bold;font-size:{HEAD}px;"
                                    f"font-family:'Noto Sans',sans-serif;fill:{TEXT_INK}"))
        h.text = text

    CAP = 1.70                              # caption size, ~19 pt at A0
    CAP_GAP = 1.8                           # figure-to-caption gap

    # -- introduction: one column across the full card --------------------------
    text = INTRO.read_text(encoding="utf-8").strip()
    intro_lines = wrap(text, FULL_W - 2 * TPAD, BODY)
    intro_h = TPAD + HEAD + 2.6 + len(intro_lines) * BODY * LEAD + TPAD - 0.6

    # -- shadow -----------------------------------------------------------------
    f = etree.SubElement(defs, N("filter"), id="cardShadow", x="-10%", y="-10%",
                         width="120%", height="130%",
                         style="color-interpolation-filters:sRGB")
    etree.SubElement(f, N("feGaussianBlur"), {"in": "SourceAlpha", "stdDeviation": "1.1",
                                              "result": "b"})
    etree.SubElement(f, N("feOffset"), {"in": "b", "dy": "0.7", "result": "o"})
    ct = etree.SubElement(f, N("feComponentTransfer"), {"in": "o", "result": "s"})
    etree.SubElement(ct, N("feFuncA"), type="linear", slope="0.45")
    mg = etree.SubElement(f, N("feMerge"))
    etree.SubElement(mg, N("feMergeNode"), {"in": "s"})
    etree.SubElement(mg, N("feMergeNode"), {"in": "SourceGraphic"})

    def card(cid, x, y, w, h):
        etree.SubElement(layer, N("rect"), id=cid, x=f"{x:.3f}", y=f"{y:.3f}",
                         width=f"{w:.3f}", height=f"{h:.3f}", rx=str(RADIUS),
                         ry=str(RADIUS),
                         style="fill:#ffffff;stroke:none;filter:url(#cardShadow)")

    def caption_lines(key, width):
        return wrap(pt.CAPTIONS[key], width - 2 * TPAD, CAP)

    def fig_card_h(key, width, scale=1.0):
        """Natural card height: figure at `scale` of the inner width, a fixed
        gap, then the caption, all inside the card padding."""
        _, pw, ph = panels[key]
        inner = width - 2 * PAD
        fig_h = inner * scale * ph / pw
        cap = caption_lines(key, width)
        return PAD + fig_h + CAP_GAP + len(cap) * CAP * LEAD + TPAD * 0.7

    def place_fig(key, x, y, width, height, scale):
        """Card of exactly `height`. The caption sits on the card's bottom inset,
        so captions in cards whose bottoms align also align; the figure is
        centred in the space left above it."""
        card(f"card{key}", x, y, width, height)
        g, pw, ph = panels[key]
        inner = width - 2 * PAD
        fig_w = inner * scale
        fig_h = fig_w * ph / pw
        cap = caption_lines(key, width)
        cap_top = y + height - TPAD * 0.7 - len(cap) * CAP * LEAD
        free_top, free_bot = y + PAD, cap_top - CAP_GAP
        fy = free_top + (free_bot - free_top - fig_h) / 2
        fx = x + PAD + (inner - fig_w) / 2
        s_ = fig_w / pw
        g.set("transform", f"matrix({s_:.8f},0,0,{s_:.8f},{fx:.4f},{fy:.4f})")
        layer.append(g)
        put_lines(f"cap{key}", cap, x + TPAD, cap_top + CAP * 0.95, CAP,
                  colour="#3d3b37")
        return fy

    def text_card_h(body, width):
        n = len(wrap_rich(body, width - 2 * TPAD, BODY))
        return TPAD + HEAD + 2.6 + n * BODY * LEAD + TPAD - 0.6

    def place_text(tid, title, body, x, y, width, h, show=True):
        card(f"card{tid}", x, y, width, h)
        hb = y + TPAD + HEAD * 0.76
        heading(f"{tid}Heading", title, x + TPAD, hb)
        first = hb + 2.6 + BODY * 0.95
        if show:
            t = etree.SubElement(layer, N("text"), id=f"{tid}Body",
                                 style=(f"font-size:{BODY}px;font-family:'Noto Sans',"
                                        f"sans-serif;fill:{TEXT_INK}"))
            for i, line in enumerate(wrap_rich(body, width - 2 * TPAD, BODY)):
                row = etree.SubElement(t, N("tspan"), x=f"{x + TPAD:.3f}",
                                       y=f"{first + i * BODY * LEAD:.3f}")
                # group consecutive words of the same weight into one tspan
                runs = []
                for w, b in line:
                    if runs and runs[-1][1] == b:
                        runs[-1][0].append(w)
                    else:
                        runs.append(([w], b))
                for k, (ws, b) in enumerate(runs):
                    sp = etree.SubElement(row, N("tspan"))
                    if b:
                        sp.set("style", "font-weight:bold")
                    sp.text = (" " if k else "") + " ".join(ws)
        else:
            put_lines(f"{tid}Body", ["[text awaiting approval]"], x + TPAD, first,
                      BODY, colour="#8a8880", italic=True)

    # -- grid -----------------------------------------------------------------------
    # Every card edge sits on one of four vertical lines (the page frame and the
    # two column edges), and every vertical gap between cards is exactly G. The
    # spare height goes into figure cards, never into gaps, so gaps stay equal.
    G = GAP_MIN
    ix = (PAGE_W - FULL_W) / 2
    colw = (FULL_W - COL_GAP) / 2
    xl, xr = ix, ix + colw + COL_GAP
    p1_h = fig_card_h(1, FULL_W)
    text_h = max(text_card_h(pt.METHODS, colw), text_card_h(pt.RESULTS, colw))
    y_text = TOP + p1_h + G + intro_h + G
    y_figs = y_text + text_h + G
    avail = BOTTOM - y_figs
    if avail <= 0:
        raise SystemExit("no room for the figure row")

    def best_scale(fits):
        if fits(1.0):
            return 1.0
        lo, hi = 0.2, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if fits(mid) else (lo, mid)
        if not fits(lo):
            raise SystemExit("figures cannot fit")
        return lo

    s_left = best_scale(lambda sc: fig_card_h(2, colw, sc) <= avail)
    s_right = best_scale(lambda sc: fig_card_h(4, colw, sc) + G
                         + fig_card_h(5, colw, sc) <= avail)
    n4, n5 = fig_card_h(4, colw, s_right), fig_card_h(5, colw, s_right)
    h4 = n4 + (avail - G - n4 - n5) * n4 / (n4 + n5)
    h5 = avail - G - h4

    # -- place ------------------------------------------------------------------------
    place_fig(1, ix, TOP, FULL_W, p1_h, 1.0)
    y_intro = TOP + p1_h + G
    card("cardIntro", ix, y_intro, FULL_W, intro_h)
    head = y_intro + TPAD + HEAD * 0.76
    heading("introHeading", "Introduction", ix + TPAD, head)
    put_lines("introBody", intro_lines, ix + TPAD, head + 2.6 + BODY * 0.95, BODY)

    place_text("methods", "Methods", pt.METHODS, xl, y_text, colw, text_h,
               show=pt.METHODS_APPROVED)
    place_text("results", "Results", pt.RESULTS, xr, y_text, colw, text_h)
    place_fig(2, xl, y_figs, colw, avail, s_left)
    place_fig(4, xr, y_figs, colw, h4, s_right)
    place_fig(5, xr, y_figs + h4 + G, colw, h5, s_right)
    new_cards = {"card2": y_figs, "card4": y_figs, "card5": y_figs + h4 + G}
    scales = {"left": round(s_left, 3), "right": round(s_right, 3)}
    y_cols = y_text
    W, x0 = colw, ix
    hs = [p1_h, intro_h]

    # Hand-added artwork: same offset from its card as before, recentred in
    # the margin, which widens or narrows with the figure column.
    for el, near, bb in anchored:
        dy = new_cards.get(near, old_cards[near]) - old_cards[near]
        dx = ((x0 - (old_x0 or x0)) / 2) if old_x0 is not None else 0.0
        el.set("transform", f"translate({dx:.4f},{dy:.4f}) " + (el.get("transform") or ""))
        layer.append(el)

    root.set("width", "841mm"); root.set("height", "1189mm")
    tree.write(str(POSTER), xml_declaration=True, encoding="UTF-8", standalone=False)
    print(f"A0 written. column width {W:.1f} mm; figure scale per column {scales}")
    print(f"panel 1 card {hs[0]:.1f} mm, intro {hs[1]:.1f} mm, columns start at {y_cols:.1f}")
    for el, near, _ in anchored:
        print(f"moved hand-added {el.get('id')} with {near}")


if __name__ == "__main__":
    main()
