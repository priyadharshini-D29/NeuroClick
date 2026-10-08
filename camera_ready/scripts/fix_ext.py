from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
ns = {'w': W, 'wp': WP, 'a': A}
DOC = 'unpacked/word/document.xml'
tree = etree.parse(DOC); root = tree.getroot()
for d in root.iter('{%s}drawing' % W):
    ext = d.find('.//wp:extent', ns)
    cx, cy = ext.get('cx'), ext.get('cy')
    for e in d.iter('{%s}ext' % A):
        if e.get('uri') is not None:
            for k in ('cx', 'cy'):
                if k in e.attrib: del e.attrib[k]
        elif e.getparent().tag == '{%s}xfrm' % A:
            e.set('cx', cx); e.set('cy', cy)
    xf = d.find('.//a:xfrm/a:ext', ns)
    print('fixed', cx, cy, 'xfrm ext ->', xf.get('cx') if xf is not None else None)
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
