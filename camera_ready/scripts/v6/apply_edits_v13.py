"""v13: template-conformance pass against splnproc1703s.docx (the file on the ICAIN Downloads page).
(1) Table 7 caption: only 'Table 7.' bold, like every other caption. (2) Declarations: bold run-in labels
(template: 'Run-in Heading in Bold. Text follows'). The 'moment, The MCC' typo was fixed in the source already."""
import os, copy
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body')); ch = list(body)
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
XMLSP = '{http://www.w3.org/XML/1998/namespace}space'

# (1) Table 7 caption
cap = ch[82]; assert ptext(cap).startswith('Table 7. Reproducibility checklist'), ptext(cap)[:50]
runs = cap.findall(w('r')); first = runs[0]; t0 = first.find(w('t'))
assert t0.text.startswith('Table 7. Reproducibility checklist'), t0.text
rest = t0.text[len('Table 7. '):]; t0.text = 'Table 7. '; t0.set(XMLSP, 'preserve')
nr = etree.Element(w('r')); rp = copy.deepcopy(first.find(w('rPr')))
if rp is not None:
    b = rp.find(w('b'))
    if b is not None: rp.remove(b)
    bcs = rp.find(w('bCs'))
    if bcs is not None: rp.remove(bcs)
    if len(rp): nr.append(rp)
nt = etree.SubElement(nr, w('t')); nt.text = rest; nt.set(XMLSP, 'preserve')
first.addnext(nr); print('Table 7 caption: bold limited to "Table 7."')

# (2) Declarations run-in labels
labels = ['Data availability.', 'Code availability.', 'Acknowledgements.', 'Author contributions.', 'Ethics statement.', 'Disclosure of Interests.']
done = 0
for p in ch:
    if p.tag != w('p'): continue
    txt = ptext(p)
    for lab in labels:
        if txt.startswith(lab + ' '):
            rs = p.findall(w('r')); assert len(rs) == 1, (lab, len(rs))
            r = rs[0]; t = r.find(w('t')); assert t.text.startswith(lab + ' ')
            t.text = t.text[len(lab) + 1:]; t.set(XMLSP, 'preserve')
            br = etree.Element(w('r')); brp = copy.deepcopy(r.find(w('rPr'))) if r.find(w('rPr')) is not None else etree.Element(w('rPr'))
            if brp.find(w('b')) is None: brp.insert(0, etree.Element(w('b')))
            br.append(brp); bt = etree.SubElement(br, w('t')); bt.text = lab + ' '; bt.set(XMLSP, 'preserve')
            r.addprevious(br); done += 1
assert done == 6, done; print('Declarations: 6 run-in labels bolded')
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
