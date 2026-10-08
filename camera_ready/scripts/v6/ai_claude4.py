import os
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__)); DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body'))
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
p = next(p for p in body.findall(w('p')) if ptext(p).startswith('Use of generative AI.'))
p.findall(w('r'))[1].find(w('t')).text = (
    'Claude (Anthropic) was used during this work as a coding assistant for writing and debugging the analysis, '
    'evaluation, figure-rendering and document-formatting scripts released with the code, and during manuscript '
    'preparation for language editing and for revising the wording of author-written text. It was not used to '
    'conceive the study, select the methods, process the data, run the experiments, interpret the findings, '
    'or generate references. The authors reviewed and verified all code, text and results and take full '
    'responsibility for the content of this paper.')
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True); print(ptext(p))
