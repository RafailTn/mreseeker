#!/usr/bin/env python3
"""
Export results/mreseeker.svg to results/mreseeker.pdf without losing the title glow.

The glow ellipse behind the title uses `mix-blend-mode: overlay` plus a blur.
PDF export has no equivalent for SVG blend modes, so Inkscape drops the blend and
paints the ellipse as flat blue - on top of the title text, since the ellipse
follows it in the file. Here the background gradient and the glow are rendered
together to a high-resolution image (the blend is computed against exactly the
background it sits on), that image replaces the ellipse in a temporary copy, and
the title text is kept as vector on top. The SVG itself is never modified.

    python3 src/figures/export_poster_pdf.py [--dpi 300]
"""
from __future__ import annotations

import argparse
import base64
import copy
import subprocess
import tempfile
from pathlib import Path

from lxml import etree

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "mreseeker.svg"
DST = REPO / "results" / "mreseeker.pdf"
SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
BACKGROUND, GLOW = "rect275", "path4406"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300,
                    help="Resolution of the baked title glow at A0 print size.")
    a = ap.parse_args()

    tree = etree.parse(str(SRC))
    root = tree.getroot()
    glow = root.find(f".//{{{SVG}}}ellipse[@id='{GLOW}']")
    if glow is None:
        raise SystemExit(f"no #{GLOW} in {SRC}; nothing to bake")
    cx, cy = float(glow.get("cx")), float(glow.get("cy"))
    rx, ry = float(glow.get("rx")), float(glow.get("ry"))
    # Region to bake: the ellipse plus room for its blur, clamped to the page.
    vb = [float(v) for v in root.get("viewBox").split()]
    pad = 4.0
    x0, y0 = max(vb[0], cx - rx - pad), max(vb[1], cy - ry - pad)
    x1, y1 = min(vb[2], cx + rx + pad), min(vb[3], cy + ry + pad)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 1. background + glow only
        only = copy.deepcopy(tree)
        layer = only.getroot().find(f"{{{SVG}}}g")
        for el in list(layer):
            if el.get("id") not in (BACKGROUND, GLOW):
                layer.remove(el)
        bg_svg = tmp / "bg.svg"
        only.write(str(bg_svg))
        png = tmp / "glow.png"
        # The page is A0 on an A4 viewBox, so export dpi scales by 4 in user units.
        # --export-area is in document px: one viewBox unit is `scale` mm.
        scale = float(root.get("width").rstrip("m")) / (vb[2] - vb[0])
        px = lambda v: v * scale * 96 / 25.4
        r = subprocess.run(["inkscape", str(bg_svg), "--export-type=png",
                            f"--export-filename={png}",
                            # whole numbers: Inkscape parses this with the
                            # system locale, which may use a decimal comma
                            f"--export-area={int(px(x0))}:{int(px(y0))}:"
                            f"{int(px(x1)) + 1}:{int(px(y1)) + 1}",
                            f"--export-dpi={a.dpi}"],
                           capture_output=True, text=True)
        if not png.exists():
            raise SystemExit(f"glow render failed:\n{r.stderr[-800:]}")

        # 2. copy with the ellipse replaced by the baked image, in the same slot
        out = copy.deepcopy(tree)
        g = out.getroot().find(f".//{{{SVG}}}ellipse[@id='{GLOW}']")
        parent = g.getparent()
        bg = out.getroot().find(f".//{{{SVG}}}rect[@id='{BACKGROUND}']")
        img = etree.Element(f"{{{SVG}}}image", id="bakedTitleGlow",
                            x=f"{x0:.4f}", y=f"{y0:.4f}",
                            width=f"{x1 - x0:.4f}", height=f"{y1 - y0:.4f}",
                            preserveAspectRatio="none")
        img.set(f"{{{XLINK}}}href", "data:image/png;base64," +
                base64.b64encode(png.read_bytes()).decode())
        parent.remove(g)
        # Directly above the background, so the title text stays on top of it.
        bg.addnext(img)
        pdf_svg = tmp / "for_pdf.svg"
        out.write(str(pdf_svg), xml_declaration=True, encoding="UTF-8")
        subprocess.run(["inkscape", str(pdf_svg), "--export-type=pdf",
                        f"--export-filename={DST}"], check=True, capture_output=True)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
