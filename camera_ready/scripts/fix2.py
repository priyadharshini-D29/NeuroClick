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

# 1. table cell paragraphs: no first-line indent, left-aligned unless already centred
n = 0
for tc in root.iter(w('tc')):
    for p in tc.iter(w('p')):
        pPr = p.find(w('pPr'))
        if pPr is None:
            pPr = etree.Element(w('pPr')); p.insert(0, pPr)
        jc = pPr.find(w('jc'))
        if jc is None:
            jc = etree.SubElement(pPr, w('jc')); jc.set(w('val'), 'left')
        ind = pPr.find(w('ind'))
        if ind is None:
            ind = etree.Element(w('ind'))
            pPr.insert(list(pPr).index(jc), ind)  # ind precedes jc in schema order
        ind.set(w('firstLine'), '0')
        for k in (w('hanging'),):
            if k in ind.attrib: del ind.attrib[k]
        n += 1
print('cell paragraphs fixed:', n)

# 2. Table 2 widths
tables = root.findall('.//w:tbl', ns)
def set_grid(tbl, widths):
    grid = tbl.find(w('tblGrid'))
    for g in list(grid): grid.remove(g)
    for wd in widths: etree.SubElement(grid, w('gridCol')).set(w('w'), str(wd))
    tbl.find(w('tblPr')).find(w('tblW')).set(w('w'), str(sum(widths)))
    for tr in tbl.findall(w('tr')):
        for k, tc in enumerate(tr.findall(w('tc'))):
            tc.find(w('tcPr')).find(w('tcW')).set(w('w'), str(widths[k]))
set_grid(tables[1], [1224, 1906, 1150, 1000, 1180, 1080, 1000])

# 3. shrink Fig 4 and Fig 5 (and Fig 7 slightly)
newcx = {'media/image4.png': 4140000, 'media/image5.png': 4250000, 'media/image7.png': 4320000}
for d in root.iter(w('drawing')):
    blip = d.find('.//a:blip', ns); target = relmap[blip.get('{%s}embed' % R)]
    if target not in newcx: continue
    img = Image.open(os.path.join('unpacked/word', target)); ratio = img.size[0] / img.size[1]
    cx = newcx[target]; cy = int(round(cx / ratio))
    ext = d.find('.//wp:extent', ns); ext.set('cx', str(cx)); ext.set('cy', str(cy))
    xf = d.find('.//a:xfrm/a:ext', ns); xf.set('cx', str(cx)); xf.set('cy', str(cy))
    print(target, f'{cx/360000:.2f} x {cy/360000:.2f} cm')
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
