"""v6 edits to unpacked/word/document.xml (audit of v5, 8 Oct 2026)."""
import sys, os
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot(); body = root.find(w('body')); ch = list(body)
FULL_DELTA = sys.argv[1] if len(sys.argv) > 1 else None   # Table 6, complete-history delta vs baseline

def ptext(p): return ''.join(t.text or '' for t in p.iter(w('t')))

def replace_in_para(idx, old, new):
    p = ch[idx]
    hits = [t for t in p.iter(w('t')) if t.text and old in t.text]
    assert len(hits) == 1, (idx, old[:40], len(hits), ptext(p)[:80])
    assert hits[0].text.count(old) == 1
    hits[0].text = hits[0].text.replace(old, new)
    print(f'[{idx}] ok: ...{old[:50]}...')

# 1. Sect. 4.1: "every pooled metric" is wrong (Table 2: NeuroClick higher on Bal. acc. and MCC)
replace_in_para(46,
    'numerically above NeuroClick on every pooled metric, and statistically indistinguishable from it at the participant level',
    'numerically above NeuroClick on pooled PR AUC and ROC AUC but below it on balanced accuracy and MCC (Table 2), and statistically indistinguishable from it at the participant level')

# 3. Abstract: the full-history ensemble result only matches the static baseline; the gain over the baseline is at first2
replace_in_para(5,
    'A leakage-free ensemble of NeuroClick and this baseline, however, significantly outperformed NeuroClick alone at complete history (+0.0637 PR AUC, Holm p < 0.0001) while matching the static baseline, recovering the sequential model\'s full-history deficit.',
    'A leakage-free ensemble of NeuroClick and this baseline significantly outperformed NeuroClick alone at complete history (+0.0637 PR AUC, Holm p < 0.0001) but only matched the static baseline there; its sole significant gain over the static baseline occurred after the second visit (+0.0187 PR AUC, Holm p = 0.022).')
replace_in_para(5,
    'Early purchase ranking is supported by behaviour and propensity; NeuroClick\'s demonstrated value lies later in the visit history.',
    'Early purchase ranking is supported by behaviour and propensity; the sequential model\'s demonstrated contribution is confined to this second-visit ensemble gain.')

# 3b. Sect. 4.2 closing sentence
replace_in_para(56,
    'which changes this picture substantially.',
    'which recovers this deficit without exceeding the static baseline.')

# 3c. Discussion closing sentence
replace_in_para(86,
    'across every horizon tested, NeuroClick alone never significantly exceeds the information-matched baseline; its demonstrated value arises through the ensemble (Sect. 4.5).',
    'across every horizon tested, NeuroClick alone never significantly exceeds the information-matched baseline, and the only significant gain over that baseline is the ensemble\'s at first2 (+0.0187 PR AUC, Holm p = 0.022; Sect. 4.5).')

# 3d. Conclusion
replace_in_para(94,
    'The sequential model\'s contribution is better supported later in the visit history: combined with the dwell-plus-propensity baseline in a leakage-free ensemble, it significantly outperforms NeuroClick alone at complete history (+0.0637 PR AUC, Holm p < 0.0001) while matching the static baseline, recovering the sequential model\'s deficit at that horizon even though neither model improves on the other after a single visit.',
    'The sequential model\'s contribution appears only in combination with the static signal: the leakage-free ensemble of NeuroClick and the dwell-plus-propensity baseline significantly exceeded the static baseline after the second visit (+0.0187 participant-mean PR AUC, Holm p = 0.022), and at complete history it significantly exceeded NeuroClick alone (+0.0637 PR AUC, Holm p < 0.0001) while only matching the static baseline, so the full-history result recovers NeuroClick\'s deficit rather than demonstrating improvement over the static model.')

# 4. "pooling artefact" overstates the evidence
replace_in_para(60,
    'so we treat it as a pooling artefact rather than evidence that eye tracking helps.',
    'so we report it as a numerical pooled advantage that participant-paired testing did not support, not as evidence that eye tracking helps.')
replace_in_para(87,
    'Eye tracking\'s picture is now resolved: pooled, it exceeds plain behaviour at first1 and first2, but this advantage does not survive participant-level correction (Sect. 4.3), so we treat it as a pooling artefact rather than a positive finding.',
    'Eye tracking added to behaviour exceeds plain behaviour in the pooled curves at first1 and first2, but this numerical advantage was not supported by participant-paired testing (Sect. 4.3), so we do not treat it as a positive finding.')

# 5. Code availability wording: body text said "released"; the declaration says "will be released"
replace_in_para(33, 'are specified in the released cache-construction code.',
                    'are specified in the accompanying cache-construction code (see Code availability).')
replace_in_para(82, 'values taken directly from the released audit and cache-construction scripts.',
                    'values taken directly from the accompanying audit and cache-construction scripts (see Code availability).')
replace_in_para(90, 'Finally, the released leakage-audit and cross-fitting pipeline offers',
                    'Finally, the accompanying leakage-audit and cross-fitting pipeline offers')
replace_in_para(97,
    'The audit, cache construction, evaluation, statistical analysis, and figure-generation scripts will be released at https://github.com/medpriya/ICAIN. The NeuMa data are not redistributed.',
    'The audit, cache construction, evaluation, statistical analysis, and figure-generation scripts accompanying this paper will be released at https://github.com/medpriya/ICAIN on publication. The NeuMa data are not redistributed.')

# 2. Table 6: report every effect size; caption must match
replace_in_para(74,
    'Differences are given only where they survive Holm correction; n.s. = not significant. The stacked variant is reported in the text.',
    'All participant-mean differences are given with their Holm-corrected p-values; only first2 versus the baseline and complete history versus NeuroClick survive correction. The stacked variant is reported in the text.')
tbl = ch[75]; assert tbl.tag == w('tbl')
rows = tbl.findall(w('tr'))
# mean(E - B) = mean(E - N) + mean(N - B); N - B participant means from Table 5 (first1 -0.0025, first2 +0.0176, first3 +0.0150)
derived = {1: '+.0055', 3: '+.0170'}
if FULL_DELTA: derived[4] = FULL_DELTA
for ri, val in derived.items():
    cell = rows[ri].findall(w('tc'))[3]
    ts = [t for t in cell.iter(w('t')) if t.text]
    assert ''.join(t.text for t in ts) == 'n.s.', (ri, [t.text for t in ts])
    ts[0].text = val
    for t in ts[1:]: t.text = ''
    print(f'Table 6 row {ri} delta vs baseline -> {val}')

tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
abs_words = len(ptext(ch[5]).replace('Abstract.', '').split())
print('abstract words:', abs_words)
