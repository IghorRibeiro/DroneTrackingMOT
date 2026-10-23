"""Minimal PDF checks without external tools: page size in inches, font types and whether every font is embedded.

Matplotlib writes font dictionaries uncompressed, so a byte-level scan is enough:
  * /MediaBox [0 0 w h] in PostScript points (72 per inch);
  * /Subtype /Type3 would be a bitmap-like Type 3 font (forbidden by the publisher);
  * every /FontDescriptor must reference an embedded font file (/FontFile2 for TrueType, /FontFile3 for CFF).
"""
from __future__ import annotations

import re
from pathlib import Path


def inspect_pdf(path: Path) -> dict:
    raw = Path(path).read_bytes()
    boxes = re.findall(rb"/MediaBox\s*\[\s*([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s*\]", raw)
    x0, y0, x1, y1 = (float(v) for v in boxes[0])
    width_in, height_in = (x1 - x0) / 72.0, (y1 - y0) / 72.0
    subtypes = re.findall(rb"/Subtype\s*/(Type3|TrueType|Type1|Type0|CIDFontType2|CIDFontType0)", raw)
    descriptors = re.findall(rb"/Type\s*/FontDescriptor.*?>>", raw, flags=re.S)
    embedded = sum(1 for d in descriptors if b"/FontFile" in d)
    names = sorted({m.decode() for m in re.findall(rb"/FontName\s*/([A-Za-z0-9+\-_]+)", raw)})
    return {"file": Path(path).name, "pages": len(boxes), "width_in": width_in, "height_in": height_in,
            "type3_fonts": sum(1 for s in subtypes if s == b"Type3"), "font_subtypes": sorted({s.decode() for s in subtypes}),
            "font_descriptors": len(descriptors), "embedded_font_files": embedded, "font_names": names}
