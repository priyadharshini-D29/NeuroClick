"""Keep Table 7 (and its caption) on one page: keepNext on the caption and on every row but the last."""
import os
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body')); ch = list(body)
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
cap, tbl = ch[82], ch[83]
assert ptext(cap).startswith('Table 7.') and tbl.tag == w('tbl')
rows = tbl.findall(w('tr')); assert len(rows) == 8
def add_keep(p):
    pPr = p.find(w('pPr'))
    if pPr is None: pPr = etree.Element(w('pPr')); p.insert(0, pPr)
    if pPr.find(w('keepNext')) is None:
        ps = pPr.find(w('pStyle')); idx = list(pPr).index(ps) + 1 if ps is not None else 0
        pPr.insert(idx, etree.Element(w('keepNext'))); return 1
    return 0
n = add_keep(cap)
for r in rows[:-1]:
    for p in r.iter(w('p')): n += add_keep(p)
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print('keepNext added to', n, 'paragraphs of Table 7')
