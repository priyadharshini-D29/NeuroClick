import os
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__)); DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body'))
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
fund = next(p for p in body.findall(w('p')) if ptext(p).startswith('Funding.'))
fund.findall(w('r'))[1].find(w('t')).text = ('No specific funding was received for this work. GPU computing time was provided in kind through the '
    'NVIDIA Academic Hardware Grant Program (see Acknowledgements).')
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True); print(ptext(fund))
