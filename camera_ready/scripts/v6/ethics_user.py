import os
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__)); DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body'))
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
# confirm [15] is the NeuMa dataset paper
refs = [p for p in body.findall(w('p')) if 'NeuMa' in ptext(p) and 'Scientific Data' in ptext(p)]
assert len(refs) == 1
allrefs = [p for p in body.findall(w('p')) if p.find(w('pPr')) is not None and p.find(w('pPr')).find(w('pStyle')) is not None and p.find(w('pPr')).find(w('pStyle')).get(w('val')) == 'referenceitem']
num = allrefs.index(refs[0]) + 1; print('NeuMa reference number:', num)
eth = next(p for p in body.findall(w('p')) if ptext(p).startswith('Ethics statement.'))
runs = eth.findall(w('r')); assert len(runs) == 2
runs[0].find(w('t')).text = 'Ethics approval and consent to participate. '
runs[1].find(w('t')).text = (f'The NeuMa data were collected in accordance with the Declaration of Helsinki under the ethical approval '
                             f'reported in the original dataset publication [{num}]; the present study constitutes a secondary analysis.')
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True); print(ptext(eth))
