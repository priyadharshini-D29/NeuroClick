"""Capitalise the first letter of every table cell that starts lowercase (all columns, all tables)."""
import os, sys
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__)); DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
apply = '--apply' in sys.argv
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body'))
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
n = 0
for ti, tbl in enumerate(body.findall(w('tbl')), 1):
    for ri, tr in enumerate(tbl.findall(w('tr'))):
        for ci, tc in enumerate(tr.findall(w('tc'))):
            txt = ptext(tc).strip()
            if txt and txt[0].islower() and txt != 'n' and not txt.startswith('p(product)'):   # keep symbols/formulas
                first = next((t for t in tc.iter(w('t')) if t.text and t.text.strip()), None)
                s = first.text; i = len(s) - len(s.lstrip())
                print(f'Table {ti} row {ri} col {ci}: "{txt[:40]}" -> "{s[:i] + s[i].upper() + s[i+1:]}"[...]')
                if apply: first.text = s[:i] + s[i].upper() + s[i+1:]
                n += 1
if apply: tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print(('fixed' if apply else 'found'), n, 'cells')
