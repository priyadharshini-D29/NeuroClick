"""Replace Word's 200-dpi downsampled figure bitmaps in the exported PDF with the 600-dpi PNGs
from the docx media folder, in document order; then drop the PDF/A identification from the XMP
metadata (images were swapped after export, so PDF/A conformance is no longer claimed)."""
import sys, os, re
import fitz
from lxml import etree
pdf_in, pdf_out, unpacked = sys.argv[1], sys.argv[2], sys.argv[3]
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
root = etree.parse(os.path.join(unpacked, 'word', 'document.xml')).getroot()
rels = etree.parse(os.path.join(unpacked, 'word', '_rels', 'document.xml.rels')).getroot()
relmap = {r.get('Id'): r.get('Target') for r in rels}
media = []
for blip in root.iter('{%s}blip' % A):
    media.append(os.path.join(unpacked, 'word', relmap[blip.get('{%s}embed' % R)]))
doc = fitz.open(pdf_in)
pdf_imgs = []
for page in doc:
    for info in page.get_image_info(xrefs=True):
        pdf_imgs.append((page, info))
assert len(media) == len(pdf_imgs), (len(media), len(pdf_imgs))
for (page, info), src in zip(pdf_imgs, media):
    bbox = fitz.Rect(info['bbox'])
    page.replace_image(info['xref'], filename=src)
    print(f'p{page.number+1}: {os.path.basename(src)} -> xref {info["xref"]} ({bbox.width:.0f}x{bbox.height:.0f} pt)')
xmp = doc.get_xml_metadata()
if 'pdfaid' in xmp:
    xmp = re.sub(r'<rdf:Description[^>]*pdfaid[^>]*>.*?</rdf:Description>', '', xmp, flags=re.S)
    doc.set_xml_metadata(xmp)
    print('PDF/A identification removed from XMP')
doc.save(pdf_out, garbage=4, deflate=True)
print('saved', pdf_out, os.path.getsize(pdf_out))
