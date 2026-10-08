from lxml import etree
from PIL import Image
import os
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
ns = {'w': W, 'wp': WP, 'a': A, 'r': R}
def w(tag): return '{%s}%s' % (W, tag)
DOC = 'unpacked/word/document.xml'
tree = etree.parse(DOC); root = tree.getroot()
rels = etree.parse('unpacked/word/_rels/document.xml.rels').getroot()
relmap = {r.get('Id'): r.get('Target') for r in rels}
# figure captions: keepNext explicitly off so the figure+caption pair can end a page
n = 0
for p in root.iter(w('p')):
    pPr = p.find(w('pPr'))
    if pPr is None: continue
    ps = pPr.find(w('pStyle'))
    if ps is not None and ps.get(w('val')) == 'figurecaption':
        kn = pPr.find(w('keepNext'))
        if kn is None:
            kn = etree.Element(w('keepNext')); pPr.insert(list(pPr).index(ps) + 1, kn)
        kn.set(w('val'), '0'); n += 1
print('figure captions with keepNext=0:', n)
for d in root.iter(w('drawing')):
    blip = d.find('.//a:blip', ns); target = relmap[blip.get('{%s}embed' % R)]
    if target != 'media/image4.png': continue
    img = Image.open(os.path.join('unpacked/word', target)); ratio = img.size[0] / img.size[1]
    cx = 4140000; cy = int(round(cx / ratio))
    d.find('.//wp:extent', ns).set('cx', str(cx)); d.find('.//wp:extent', ns).set('cy', str(cy))
    xf = d.find('.//a:xfrm/a:ext', ns); xf.set('cx', str(cx)); xf.set('cy', str(cy))
    print(target, f'{cx/360000:.2f} x {cy/360000:.2f} cm')
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
