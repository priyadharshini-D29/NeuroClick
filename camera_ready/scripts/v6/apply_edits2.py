"""Second pass: results-file-backed values and the technical-check qualifications."""
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
    assert len(hits) == 1 and hits[0].text.count(old) == 1, (idx, old[:40], len(hits))
    hits[0].text = hits[0].text.replace(old, new); print(f'[{idx}] ok: {old[:50]}')

# Table 6, complete history, delta vs baseline (ensemble_pairwise_tests.csv: +0.0019062, Holm p = 1.0)
tbl = ch[75]; cell = tbl.findall(w('tr'))[4].findall(w('tc'))[3]
ts = [t for t in cell.iter(w('t')) if t.text]; assert ''.join(t.text for t in ts) == 'n.s.'
ts[0].text = '+.0019'; [setattr(t, 'text', '') for t in ts[1:]]; print('Table 6 full delta vs baseline -> +.0019')

# Sect. 3.2: NeuMa release used (raw XDF, not the supplied zero-phase-filtered arrays)
rep(29, 'EEG channels are reconstructed to a 19-channel montage and processed only in the forward direction.',
        'All signals are read from the raw NeuMa XDF recordings rather than from the dataset\'s supplied preprocessed arrays, whose zero-phase filtering and gaze interpolation would not be past-only. EEG channels are reconstructed to a 19-channel montage and filtered only in the forward direction (causal common-average reference, 1–45 Hz band-pass, and 50 Hz notch).')

# Sect. 4.5: the stacked variant and the threshold rule reuse the single outer LOPO prediction set
rep(73, 'using two leakage-free strategies: an unweighted average in logit space (no free parameters), and a stacked logistic regression on the two logits, refit separately for each held-out participant using only the other 41 participants’ out-of-fold predictions, so that no participant’s own label contributes to their own combined prediction.',
        'using two strategies: an unweighted average in logit space, which has no free parameters and therefore introduces no fitting step beyond the two leave-one-participant-out components, and a stacked logistic regression on the two logits, refit separately for each held-out participant using only the other 41 participants’ out-of-fold predictions, so that no participant’s own label contributes to their own combined prediction. Those out-of-fold predictions are taken from the single outer leave-one-participant-out run, in which the component models that scored the other 41 participants were trained on partitions containing the held-out participant; the stacker and the threshold rule below therefore see no held-out label directly, but are not re-nested within each outer training partition. All headline PR AUC results in this section use the parameter-free average.')

# Limitations: name this and the clipped-visit point
rep(92, 'One item remains open: confidence intervals describe variation across participants from a single training seed per fold, with multi-seed variance not yet quantified despite several close horizon-level comparisons.',
        'Three items remain open. Confidence intervals describe variation across participants from a single training seed per fold, with multi-seed variance not yet quantified despite several close horizon-level comparisons. The stacked ensemble and the threshold-matched MCC analysis (Sect. 4.5) reuse the outer leave-one-participant-out predictions rather than re-nesting them within each training partition; the parameter-free average ensemble, on which the reported PR AUC gains rest, is unaffected. For Buy sequences, a visit in progress at the 500 ms pre-click cutoff is clipped there rather than ending naturally, so a "completed" visit immediately before a purchase can be shorter than its unclipped counterpart; the share of clipped visits and its effect on the first1 result have not been quantified.')

tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print('abstract words:', len(ptext(ch[5]).replace('Abstract.', '').split()))
