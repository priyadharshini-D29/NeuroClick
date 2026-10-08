"""Scale the placed size of one figure: scale_fig.py <media name> <factor>"""
import os, sys
from lxml import etree
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
BASE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(BASE, 'unpacked', 'word')
name, f = sys.argv[1], float(sys.argv[2])
rels = etree.parse(os.path.join(U, '_rels', 'document.xml.rels')).getroot()
rid = [r.get('Id') for r in rels if r.get('Target') == 'media/' + name][0]
tree = etree.parse(os.path.join(U, 'document.xml')); root = tree.getroot(); n = 0
for d in root.iter('{%s}drawing' % W):
    blip = d.find('.//{%s}blip' % A)
    if blip is None or blip.get('{%s}embed' % R) != rid: continue
    for ext in [d.find('.//{%s}extent' % WP), d.find('.//{%s}ext' % A)]:
        cx, cy = int(ext.get('cx')), int(ext.get('cy'))
        ext.set('cx', str(int(cx * f))); ext.set('cy', str(int(cy * f))); n += 1
        print(name, 'extent', cx // 12700, 'x', cy // 12700, 'pt ->', int(cx * f) // 12700, 'x', int(cy * f) // 12700, 'pt')
assert n == 2, n
tree.write(os.path.join(U, 'document.xml'), xml_declaration=True, encoding='UTF-8', standalone=True)
