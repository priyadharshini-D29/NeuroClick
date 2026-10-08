"""v11: third external audit. (1) matched-baseline sensitivity check (scripts/neuroclick_crc_matched_baseline.py);
(2) 'confirmatory' -> secondary; (3) per-model thresholds, 'closely approximates'; plus two wording nits."""
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

# nit: Sect. 3.2 wording
rep(28, 'every other Buy first visit ended naturally because a later visit followed.',
        'every other Buy first visit was not truncated by the cutoff, because a later visit followed.')

# nit: Sect. 3.5 ordering - move the retraining sentence after the inner-validation sentence
RETRAIN = (' The early-stopped network serves only to choose the epoch count and the MCC thresholds: the network scored on the '
           'test participant is retrained from scratch on all 41 training participants for exactly that number of epochs, with the '
           'feature scaler and the training propensity refit on those 41 participants, and the inner-validation thresholds are applied '
           'unchanged to its test predictions.')
rep(43, RETRAIN, '')
rep(43, 'chooses horizon-specific Matthews correlation coefficient (MCC) [19] thresholds.',
        'chooses horizon-specific Matthews correlation coefficient (MCC) [19] thresholds.' + RETRAIN)

# (1) matched baseline: the published summary carries two validity fractions that behaviour-only NeuroClick never sees
rep(43, 'At first1 this reduces to the initial dwell and the validity fractions.',
        'At first1 this reduces to the initial dwell and the validity fractions. Behaviour-only NeuroClick does not receive the two '
        'validity fractions, so the published baselines carry slightly more information than the sequential model. Re-evaluating both '
        'logistic baselines without those two values, under the identical protocol, left every conclusion unchanged: for dwell + '
        'propensity at first1, pooled PR AUC .2932 (published .2938), participant mean .3511, and NeuroClick minus baseline +0.0021 '
        '(95% CI −0.0096 to 0.0138, uncorrected p = 0.66); at the later horizons the participant-mean differences moved by at most '
        '0.004. "Information-matched" below refers to this check.')
rep(46, 'is against a baseline given the same two sources of information: the browsing-behaviour summary plus the identical cross-fitted propensity.',
        'is against a baseline given the same two sources of information: the browsing-behaviour summary plus the identical cross-fitted '
        'propensity (matched up to the two validity fractions, whose removal does not change the result; Sect. 3.5).')

# (2) confirmatory -> secondary
rep(73, 'The stacked-ensemble and threshold-matched MCC results are therefore reported as exploratory; only the parameter-free average\'s PR AUC analysis, which involves no second-stage fitting, is treated as confirmatory, and all headline results in this section use it.',
        'The stacked-ensemble and threshold-matched MCC results are therefore reported as exploratory. The parameter-free average requires '
        'no second-stage fitting; its PR AUC results are reported as secondary analyses (first1 remains the prespecified primary endpoint, '
        'Sect. 3.5), and all headline results in this section use it.')

# (3) threshold rule
rep(79, 'each held-out participant receives the MCC-maximising threshold from the remaining 41 participants’ out-of-fold predictions, applied identically to all models, so that all models are threshold-matched (under this rule NeuroClick’s first1 MCC reproduces the inner-validated value, 0.2337 vs 0.2336).',
        'each model receives its own threshold chosen by one shared rule: for every held-out participant, the threshold that maximises MCC '
        'on that model’s out-of-fold predictions for the remaining 41 participants. The rule, not a common numerical threshold, is what '
        'all models share (under this rule NeuroClick’s first1 MCC closely approximates the inner-validated value, 0.2337 vs 0.2336).')
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print('abstract words:', len(ptext(ch[5]).replace('Abstract.', '').split()))
