"""Undo split_table7.py: move the continuation rows back into Table 7 and drop the '(continued)' caption."""
import os
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body')); ch = list(body)
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
tbl, cap2, tbl2 = ch[83], ch[84], ch[85]
assert tbl.tag == w('tbl') and ptext(cap2) == 'Table 7. (continued)' and tbl2.tag == w('tbl'), (ptext(cap2))
rows2 = tbl2.findall(w('tr'))
for r in rows2[1:]:
    tbl2.remove(r); tbl.append(r)
body.remove(cap2); body.remove(tbl2)
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print('Table 7 merged: rows', len(tbl.findall(w('tr'))))
