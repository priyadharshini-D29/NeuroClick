"""Split Table 7 after row SPLIT_AT (0 = header) and insert a 'Table 7. (continued)' caption that starts page 11."""
import os, copy, sys
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
SPLIT_AT = int(sys.argv[1]) if len(sys.argv) > 1 else 5      # first row index that goes to the continuation
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body')); ch = list(body)
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
cap = ch[82]; tbl = ch[83]
assert ptext(cap).startswith('Table 7.') and tbl.tag == w('tbl')
rows = tbl.findall(w('tr')); assert len(rows) == 8, len(rows)
assert ptext(rows[SPLIT_AT]).startswith('Propensity smoothing'), ptext(rows[SPLIT_AT])[:40]
# continuation table: tblPr + tblGrid + header row + remaining rows
tbl2 = etree.Element(w('tbl'))
for child in tbl:
    if child.tag in (w('tblPr'), w('tblGrid')): tbl2.append(copy.deepcopy(child))
tbl2.append(copy.deepcopy(rows[0]))
for r in rows[SPLIT_AT:]:
    tbl.remove(r); tbl2.append(r)
# keep the remaining first-part rows from splitting onto the next page: last row of part 1 must not keepNext
for p in rows[SPLIT_AT - 1].iter(w('p')):
    pPr = p.find(w('pPr'))
    if pPr is not None:
        kn = pPr.find(w('keepNext'))
        if kn is not None: pPr.remove(kn)
# continuation caption: copy caption paragraph formatting, text 'Table 7. (continued)', page break before
cap2 = copy.deepcopy(cap)
runs = cap2.findall(w('r'))
for r in runs[1:]: cap2.remove(r)
t = runs[0].find(w('t')); t.text = 'Table 7. '; t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
r2 = etree.SubElement(cap2, w('r')); rp = copy.deepcopy(runs[0].find(w('rPr')))
if rp is not None:
    b = rp.find(w('b'))
    if b is not None: rp.remove(b)
    r2.append(rp)
t2 = etree.SubElement(r2, w('t')); t2.text = '(continued)'
pPr = cap2.find(w('pPr'))
if pPr is None: pPr = etree.Element(w('pPr')); cap2.insert(0, pPr)
if pPr.find(w('pageBreakBefore')) is None:
    ps = pPr.find(w('pStyle')); idx = list(pPr).index(ps) + 1 if ps is not None else 0
    kn = pPr.find(w('keepNext'))
    if kn is not None: idx = list(pPr).index(kn) + 1
    pPr.insert(idx, etree.Element(w('pageBreakBefore')))
i = list(body).index(tbl)
body.insert(i + 1, cap2); body.insert(i + 2, tbl2)
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print('Table 7 split: part 1 rows', len(tbl.findall(w('tr'))), '| part 2 rows', len(tbl2.findall(w('tr'))), '| caption:', ptext(cap2))
