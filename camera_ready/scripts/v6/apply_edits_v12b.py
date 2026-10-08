"""v12 trims: abstract back under 250 words; drop one redundant Limitations sentence; tighten the new Sect. 3.2 sentence."""
import os
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body')); ch = list(body)
def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))
def rep(idx, old, new):
    p = ch[idx]; hits = [t for t in p.iter(w('t')) if t.text and old in t.text]
    assert len(hits) == 1 and hits[0].text.count(old) == 1, (idx, old[:60], len(hits), ptext(p)[:90])
    hits[0].text = hits[0].text.replace(old, new); print(f'[{idx}] ok: {old[:55]}')
rep(5, '(a visit that has ended or reached the audited pre-decision cutoff)', '(one that has ended or reached the pre-decision cutoff)')
rep(5, 'Evaluation uses leave-one-participant-out testing with participant-disjoint validation and paired significance testing throughout.',
       'Evaluation is leave-one-participant-out with participant-disjoint validation and paired significance testing.')
rep(28, 'it is not a deployment-style prospective test in which the boundary is fixed without knowledge of the outcome (Sect. 6).',
        'it is not a prospective test in which the boundary is fixed without knowledge of the outcome (Sect. 6).')
assert ptext(ch[92]).startswith('This is a single-dataset study')
rep(92, 'The propensity cross-fitting and the two information-matched first1 baselines are specified in Sect. 3.3 and evaluated in Sects. 4.1 and 4.4. ', '')
rep(92, 'and the two classes may differ in how their final visits end. ', '')
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print('abstract words:', len(ptext(ch[5]).replace('Abstract.', '').split()))
