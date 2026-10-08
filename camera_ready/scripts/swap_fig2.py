from lxml import etree
from PIL import Image
import os, shutil
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
ns = {'w': W, 'wp': WP, 'a': A, 'r': R}
def w(tag): return '{%s}%s' % (W, tag)
shutil.copy('newfigs/fig2_singlerow.png', 'unpacked/word/media/image2.png')
DOC = 'unpacked/word/document.xml'
tree = etree.parse(DOC); root = tree.getroot()
rels = etree.parse('unpacked/word/_rels/document.xml.rels').getroot()
relmap = {r.get('Id'): r.get('Target') for r in rels}
for d in root.iter(w('drawing')):
    blip = d.find('.//a:blip', ns); target = relmap[blip.get('{%s}embed' % R)]
    if target != 'media/image2.png': continue
    img = Image.open('unpacked/word/media/image2.png'); ratio = img.size[0] / img.size[1]
    cx = 4392000; cy = int(round(cx / ratio))   # 12.2 cm wide
    d.find('.//wp:extent', ns).set('cx', str(cx)); d.find('.//wp:extent', ns).set('cy', str(cy))
    xf = d.find('.//a:xfrm/a:ext', ns); xf.set('cx', str(cx)); xf.set('cy', str(cy))
    print('Fig 2 ->', f'{cx/360000:.2f} x {cy/360000:.2f} cm')
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
