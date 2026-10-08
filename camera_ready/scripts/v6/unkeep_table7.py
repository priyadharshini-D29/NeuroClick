"""Let Table 7 break across pages again: remove keepNext from its rows (header still repeats); keep caption with first row."""
import os
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__)); DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body')); ch = list(body)
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
cap, tbl = ch[82], ch[83]; assert ptext(cap).startswith('Table 7.') and tbl.tag == w('tbl')
n = 0
for r in tbl.findall(w('tr'))[1:]:
    for p in r.iter(w('p')):
        pPr = p.find(w('pPr'))
        if pPr is not None and pPr.find(w('keepNext')) is not None: pPr.remove(pPr.find(w('keepNext'))); n += 1
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True); print('keepNext removed from', n, 'Table 7 paragraphs (header row keeps with first body row)')
