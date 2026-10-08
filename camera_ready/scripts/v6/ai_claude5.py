import os
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__)); DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body'))
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
p = next(p for p in body.findall(w('p')) if ptext(p).startswith('Use of generative AI.'))
r = p.findall(w('r')); r[0].find(w('t')).text = 'Use of Generative AI. '
r[1].find(w('t')).text = (
    'Claude (Anthropic) was used to assist with code writing and debugging for the analysis and figure-generation '
    'scripts, as well as language editing of author-prepared manuscript text. The tool was not used for study design, '
    'scientific interpretation, or reference selection. All AI-assisted outputs, analyses, and reported results were '
    'independently reviewed and verified by the authors, who take full responsibility for the content and integrity '
    'of this work.')
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True); print(ptext(p))
