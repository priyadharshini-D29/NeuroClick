import sys, fitz
pdf = sys.argv[1]
doc = fitz.open(pdf)
print("pages", doc.page_count)
fonts = {}
for page in doc:
    for f in page.get_fonts(full=True):
        xref, ext, ftype, name, ref, enc = f[:6]
        fonts.setdefault((name, ftype, ext), set()).add(page.number + 1)
print("\nFONTS (name, type, ext) -> pages")
for k, v in sorted(fonts.items()):
    emb = "EMBEDDED" if k[2] != "n/a" else "NOT embedded"
    print(f"  {k[0]:40s} {k[1]:10s} {emb:14s} pages {sorted(v)}")
print("\nIMAGES")
for page in doc:
    for info in page.get_image_info(xrefs=True):
        xref = info["xref"]
        w, h = info["width"], info["height"]
        bbox = fitz.Rect(info["bbox"])
        dpi_x = w / (bbox.width / 72) if bbox.width else 0
        dpi_y = h / (bbox.height / 72) if bbox.height else 0
        print(f"  p{page.number+1} xref {xref} {w}x{h}px placed {bbox.width:.0f}x{bbox.height:.0f}pt -> {dpi_x:.0f}x{dpi_y:.0f} dpi")
