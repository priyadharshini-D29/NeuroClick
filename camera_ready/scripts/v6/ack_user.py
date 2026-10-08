import os
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__)); DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body'))
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
ack = next(p for p in body.findall(w('p')) if ptext(p).startswith('Acknowledgements.'))
ack.findall(w('r'))[1].find(w('t')).text = ('The authors, Priyadharshini D and Shridevi S, thank Vellore Institute of Technology, Chennai, for research '
    'support. GPU compute for all primary experiments was provided through the NVIDIA Academic Hardware Grant via Brev.dev '
    '(NVIDIA A100-SXM4-80 GB), awarded to Shridevi S.')
fund = next(p for p in body.findall(w('p')) if ptext(p).startswith('Funding.'))
fund.findall(w('r'))[1].find(w('t')).text = ('No specific funding was received for this work; GPU computing time was provided in kind through the NVIDIA '
    'Academic Hardware Grant (see Acknowledgements).')
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print(ptext(ack)); print(ptext(fund))
