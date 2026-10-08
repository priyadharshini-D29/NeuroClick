"""Insert a 'Use of generative AI.' declaration after 'Disclosure of Interests.' (same wording as the INS-HDGS-CMT paper)."""
import os, copy
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__)); DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body'))
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
src = next(p for p in body.findall(w('p')) if ptext(p).startswith('Disclosure of Interests.'))
assert not any(ptext(p).startswith('Use of generative AI') for p in body.findall(w('p')))
new = copy.deepcopy(src); runs = new.findall(w('r')); assert len(runs) == 2, len(runs)
runs[0].find(w('t')).text = 'Use of generative AI. '
runs[1].find(w('t')).text = ('Generative AI was used solely for language editing and grammar improvement during manuscript preparation. '
                             'All content was reviewed and approved by the authors, who are fully responsible for the final manuscript.')
src.addnext(new); tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print('inserted:', ptext(new)[:80])
