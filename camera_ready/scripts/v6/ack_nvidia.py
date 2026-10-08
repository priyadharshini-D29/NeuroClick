"""Acknowledgements and Funding worded as in the INS-HDGS-CMT manuscript."""
import os, copy
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__)); DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body'))
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
ack = next(p for p in body.findall(w('p')) if ptext(p).startswith('Acknowledgements.'))
runs = ack.findall(w('r')); assert len(runs) == 2 and runs[0].find(w('t')).text == 'Acknowledgements. '
runs[1].find(w('t')).text = 'All experiments were run on NVIDIA A100 GPUs provided through the NVIDIA Academic Grant Program, awarded to the second author (Shridevi S.).'
assert not any(ptext(p).startswith('Funding.') for p in body.findall(w('p')))
fund = copy.deepcopy(ack); fr = fund.findall(w('r'))
fr[0].find(w('t')).text = 'Funding. '
fr[1].find(w('t')).text = 'No specific funding was received for this work; GPU computing time was provided in kind through the NVIDIA Academic Grant Program (see Acknowledgements).'
ack.addnext(fund)
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print('Acknowledgements:', ptext(ack)); print('Funding:', ptext(fund))
