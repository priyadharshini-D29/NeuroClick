import sys, pymupdf, os
pdf, outdir, dpi = sys.argv[1], sys.argv[2], int(sys.argv[3])
pages = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else None
os.makedirs(outdir, exist_ok=True)
doc = pymupdf.open(pdf)
print('pages', len(doc))
for pno in range(len(doc)):
    if pages and (pno + 1) not in pages:
        continue
    pix = doc[pno].get_pixmap(dpi=dpi)
    out = os.path.join(outdir, 'page%02d.png' % (pno + 1))
    pix.save(out)
    print(out)
