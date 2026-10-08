"""Move the Fig. 7 drawing paragraph and its caption to directly after Table 6, so the figure no
longer spills onto the next page and leaves page 9 half empty."""
import os
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body')); ch = list(body)
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
tbl6, after_tbl, text_p, fig_p, cap_p = ch[75], ch[76], ch[77], ch[78], ch[79]
assert tbl6.tag == w('tbl') and ptext(cap_p).startswith('Fig. 7.') and fig_p.find('.//' + w('drawing')) is not None
assert ptext(text_p).startswith('Neither ensemble strategy') and ptext(after_tbl) == ''
body.remove(fig_p); body.remove(cap_p)
idx = list(body).index(after_tbl) + 1
body.insert(idx, fig_p); body.insert(idx + 1, cap_p)
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print('Fig. 7 moved after Table 6')
