"""Keep a figure's placed width, set its height from the PNG's pixel aspect: set_fig_aspect.py <media name> [width_pt]"""
import os, sys
from lxml import etree
from PIL import Image
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'; WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'; W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
BASE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(BASE, 'unpacked', 'word'); name = sys.argv[1]
width_pt = float(sys.argv[2]) if len(sys.argv) > 2 else None
im = Image.open(os.path.join(U, 'media', name)); ratio = im.size[1] / im.size[0]
rels = etree.parse(os.path.join(U, '_rels', 'document.xml.rels')).getroot()
rid = [r.get('Id') for r in rels if r.get('Target') == 'media/' + name][0]
tree = etree.parse(os.path.join(U, 'document.xml')); root = tree.getroot(); n = 0
for d in root.iter('{%s}drawing' % W):
    blip = d.find('.//{%s}blip' % A)
    if blip is None or blip.get('{%s}embed' % R) != rid: continue
    exts = [d.find('.//{%s}extent' % WP)] + d.findall('.//{%s}xfrm/{%s}ext' % (A, A))
    for ext in exts:
        cx = int(width_pt * 12700) if width_pt else int(ext.get('cx'))
        ext.set('cx', str(cx)); ext.set('cy', str(int(cx * ratio))); n += 1
    print(name, 'placed', cx // 12700, 'x', int(cx * ratio) // 12700, 'pt', f'({n} extents)')
assert n == 2, n
tree.write(os.path.join(U, 'document.xml'), xml_declaration=True, encoding='UTF-8', standalone=True)
