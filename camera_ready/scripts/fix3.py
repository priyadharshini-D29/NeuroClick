from lxml import etree
from PIL import Image
import os, copy
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
body = root.find('w:body', ns); ch = list(body)
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))

# figure sizes
newcx = {'media/image2.png': 4320000, 'media/image6.png': 4180000}
for d in root.iter(w('drawing')):
    blip = d.find('.//a:blip', ns); target = relmap[blip.get('{%s}embed' % R)]
    if target not in newcx: continue
    img = Image.open(os.path.join('unpacked/word', target)); ratio = img.size[0] / img.size[1]
    cx = newcx[target]; cy = int(round(cx / ratio))
    ext = d.find('.//wp:extent', ns); ext.set('cx', str(cx)); ext.set('cy', str(cy))
    xf = d.find('.//a:xfrm/a:ext', ns); xf.set('cx', str(cx)); xf.set('cy', str(cy))
    print(target, f'{cx/360000:.2f} x {cy/360000:.2f} cm')

# Table 2 widths: wider Condition column
tables = root.findall('.//w:tbl', ns)
widths = [1224, 2200, 1100, 900, 1180, 1036, 900]
tbl = tables[1]
grid = tbl.find(w('tblGrid'))
for g in list(grid): grid.remove(g)
for wd in widths: etree.SubElement(grid, w('gridCol')).set(w('w'), str(wd))
tbl.find(w('tblPr')).find(w('tblW')).set(w('w'), str(sum(widths)))
for tr in tbl.findall(w('tr')):
    for k, tc in enumerate(tr.findall(w('tc'))):
        tc.find(w('tcPr')).find(w('tcW')).set(w('w'), str(widths[k]))

# shorter Fig 2 caption
for p in ch:
    if p.tag == w('p') and ptext(p).startswith('Fig. 2.'):
        ts = list(p.iter(w('t')))
        ts[1].text = (' NeuroClick architecture, unrolled per completed visit. Browsing behaviour (with the cross-fitted '
                      'propensity logit, Sect. 3.3) is always present; EEG and eye tracking enter through optional '
                      'validity-gated residual branches. The fused representation z(j) feeds a unidirectional GRU whose '
                      'sigmoid head yields the visit score h(j); Eq. (1) aggregates these into the nondecreasing risk p(t). '
                      'The timeline marks the audited cutoff.')
        print('Fig 2 caption:', ptext(p)[:80])
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
