"""v12: fourth external audit. Outcome-dependent cutoff made explicit (Sect. 3.2 + Limitations), 'completed visit'
qualified in the abstract, threshold-transfer caveat, code pointer names both scripts. Reference 8 unchanged (publisher's title)."""
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

# Issue 2: abstract
rep(5, 'After the first completed visit, NeuroClick reached',
       'After the first completed visit (a visit that has ended or reached the audited pre-decision cutoff), NeuroClick reached')
# Issue 1: Sect. 3.2, right after the two cutoff definitions
rep(28, 'A visit is retained only if it begins before its cutoff.',
        'The observation boundary therefore depends on the eventual outcome: Buy sequences are cut 500 ms before the decision, NoBuy '
        'sequences when browsing of the page ends. The evaluation asks how well the recorded history up to each boundary ranks purchases; '
        'it is not a deployment-style prospective test in which the boundary is fixed without knowledge of the outcome (Sect. 6). '
        'A visit is retained only if it begins before its cutoff.')
# Issue 3: code pointer
rep(33, 'are specified in the accompanying cache-construction code (see Code availability).',
        'are specified in the accompanying code: the causal filters in the cache-construction script and the descriptor computations in '
        'the visit-feature extraction of the benchmark script (see Code availability).')
# Issue 1 + threshold transfer: Limitations
assert ptext(ch[94]).startswith('This is a single-dataset study'), ptext(ch[94])[:60]
rep(94, 'the effect of these truncated visits on the first1 result has not been isolated.',
        'the effect of these truncated visits on the first1 result has not been isolated. More generally, because the Buy cutoff is '
        'anchored to the click and the NoBuy cutoff to the end of page browsing, the observation boundary is partly defined by the '
        'outcome (Sect. 3.2); the results characterise ranking from outcome-anchored histories, not prospective prediction at an '
        'outcome-blind moment, and the two classes may differ in how their final visits end. The MCC thresholds are chosen on the '
        'inner-validation predictions and then applied to a network retrained on all 41 training participants, so any calibration '
        'shift after retraining affects the threshold-dependent metrics; PR AUC and ROC AUC are unaffected.')
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print('abstract words:', len(ptext(ch[5]).replace('Abstract.', '').split()))
