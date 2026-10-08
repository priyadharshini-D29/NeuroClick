"""v10: second external audit. Every methods statement below was read from the released scripts
(04_run_neuroclick_hazard.py, 03_run_classical_losocv_benchmarks.py, 02_build_..._causal_v1.py,
01_audit_buy_nobuy_events.py) and the audit manifest (results/audit_full_42_v2)."""
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

# ---- 5. overstated wording + 1. ensemble qualification (abstract, intro, contribution 4)
rep(5, 'but a logistic regression given the identical behaviour-plus-propensity inputs matched it (participant-mean difference −0.0025, Holm p = 0.76): at the first visit, browsing behaviour and propensity carry the measurable signal and the sequential architecture adds nothing measurable beyond them.',
       'but a logistic regression given the same browsing-behaviour information plus the identical cross-fitted propensity matched it (participant-mean difference −0.0025, Holm p = 0.76): no statistically detectable advantage of the sequential architecture was observed at the first visit.')
rep(5, 'A leakage-free ensemble of NeuroClick and this baseline', 'A parameter-free average ensemble of NeuroClick and this baseline')
rep(9, 'Unlike conventional propensity-based purchase prediction, which produces a single retrospective score per user–item pair once the interaction is complete, this formulation estimates risk while browsing is still in progress,',
       'Unlike purchase-prediction models that score a user–item pair only once the interaction is complete, this formulation estimates risk after each completed visit while browsing is still in progress,')
rep(13, 'A horizon-wise, leakage-free ensemble analysis,', 'A horizon-wise ensemble analysis,')

# ---- 4. sample retention (Sect. 3.1) and completed-visit definition + window end + truncation counts (Sect. 3.2)
rep(21, 'The final cache contains 5,715 sequences: 746 Buy and 4,969 NoBuy sequences (Table 1);',
        'Of the 6,048 participant-product pairs, 333 (10 of them with a recorded purchase click) received no gaze visit of at least 100 ms before their cutoff (Sect. 3.2) and are excluded; the final cache contains the remaining 5,715 sequences: 746 Buy and 4,969 NoBuy sequences (Table 1);')
rep(28, 'A visit is retained only if it begins before its cutoff. Its end is clipped to the earliest of the observed end, the page-interval end, and the cutoff. One second of preceding signal history is sampled for the EEG and eye-tracking representations; missing history is marked by a validity mask.',
        'A visit is retained only if it begins before its cutoff. Its end is the earliest of the observed end, the page-interval end, and the cutoff, so a retained visit may end naturally or be truncated at the cutoff; a "completed visit" in this paper is a visit that has reached this end, not necessarily one that gaze has left. Truncation is common for Buy sequences, because participants were usually still looking at the product 500 ms before clicking: the last pre-cutoff visit was still in progress at the cutoff in 490 of the 746 Buy sequences. The first visit itself is truncated only when it is the sole visit, which happened in 67 of the 93 single-visit Buy sequences (9.0% of all Buy first visits); every other Buy first visit ended naturally because a later visit followed. For each retained visit, the one second of signal immediately preceding its end is resampled for the EEG and eye-tracking representations, using only samples before that end; missing history is marked by a validity mask.')

# ---- 2. final outer-fold fit (Sect. 3.5) and 3. baseline inputs, PR AUC implementation, pooled vs participant-level
rep(43, 'the validation subset holds out 20% of training participants by grouped split (seed 42, deterministic).',
        'the validation subset holds out 20% of training participants by grouped split (seed 42, deterministic). The early-stopped network serves only to choose the epoch count and the MCC thresholds: the network scored on the test participant is retrained from scratch on all 41 training participants for exactly that number of epochs, with the feature scaler and the training propensity refit on those 41 participants, and the inner-validation thresholds are applied unchanged to its test predictions.')
rep(43, 'a classical logistic-regression baseline given the identical behaviour features plus the identical cross-fitted propensity (Sect. 4.1; labelled dwell + propensity in the tables and figures)',
        'a classical logistic-regression baseline given a summary of the same visit history plus the identical cross-fitted propensity logit (Sect. 4.1; labelled dwell + propensity in the tables and figures)')
rep(43, 'Every classical baseline standardises its inputs on the training partition',
        'The classical behaviour baselines (logistic behaviour and dwell + propensity) do not receive NeuroClick\'s per-visit sequence; at every horizon they receive a ten-value summary of the visits observed up to that horizon: visit count; total, mean, standard-deviation and maximum dwell; elapsed time from the first visit\'s start to the current visit\'s end; mean and maximum inter-visit gap; and the mean EEG and eye-tracking validity fractions. At first1 this reduces to the initial dwell and the validity fractions. The total-dwell heuristic ranks by summed dwell alone and the product-propensity heuristic by the training-partition propensity alone. Every classical baseline standardises its inputs on the training partition')
rep(43, 'Every participant-paired comparison is a two-sided Wilcoxon signed-rank test [20]',
        'PR AUC is average precision as implemented in scikit-learn (non-interpolated). Pooled values (Tables 2–4) are computed once over all held-out predictions concatenated across the 42 folds; participant-level values are computed within each held-out participant and then averaged (Tables 5–6, Figs. 6–7), which is why pooled and participant-mean figures differ (for example NeuroClick\'s first1 MCC is 0.2220 pooled and 0.2336 as a participant mean). Every participant-paired comparison is a two-sided Wilcoxon signed-rank test [20]')

# ---- Sect. 4.1 / 4.2 / 4.4 wording
rep(46, 'This is significantly higher than logistic regression on the same five behaviour features alone',
        'This is significantly higher than logistic regression on the browsing-behaviour summary alone')
rep(46, 'is against a baseline given exactly the same two inputs: dwell features plus the identical cross-fitted propensity.',
        'is against a baseline given the same two sources of information: the browsing-behaviour summary plus the identical cross-fitted propensity.')
rep(56, 'but under the same correction that removed the first1 advantage, no horizon shows',
        'but after Holm correction within the secondary comparison family, no horizon shows')
rep(67, 'and that propensity itself is not a leakage artefact; they do not support that NeuroClick\'s recurrent architecture adds value beyond a linear combination of the same two inputs at this horizon.',
        'while cross-fitting (Sect. 3.3) ensures that no training sequence\'s own label enters its propensity estimate; they do not support that NeuroClick\'s recurrent architecture adds value beyond a linear combination of the same two information sources at this horizon.')

# ---- 1. Sect. 4.5: exploratory qualification
rep(73, 'but are not re-nested within each outer training partition. All headline PR AUC results in this section use the parameter-free average.',
        'but are not re-nested within each outer training partition. The stacked-ensemble and threshold-matched MCC results are therefore reported as exploratory; only the parameter-free average\'s PR AUC analysis, which involves no second-stage fitting, is treated as confirmatory, and all headline results in this section use it.')
rep(79, 'For MCC, each held-out participant receives', 'For MCC (exploratory, see above), each held-out participant receives')
rep(78, 'with Holm-corrected ensemble vs NeuroClick p-values.',
        'with Holm-corrected p-values for the average ensemble against NeuroClick and against the dwell + propensity baseline.')

# ---- Discussion, 5.1, Limitations, Conclusion
rep(86, 'disappears against a baseline given the identical inputs – dwell features and the same cross-fitted propensity – combined via plain logistic regression',
        'disappears against a baseline given the same information – a summary of the same visit history and the same cross-fitted propensity – combined via plain logistic regression')
rep(86, 'combined in a leakage-free ensemble, the two significantly exceed', 'combined in the parameter-free average ensemble, the two significantly exceed')
rep(88, 'subtle EEG effects would be indistinguishable from noise under participant-disjoint testing.',
        'the cohort may have limited power to detect subtle EEG effects under participant-disjoint testing.')
rep(90, 'Ranking purchase intent after a single completed visit enables lighter-touch, better-timed assistance – surfacing relevant support early rather than retargeting exhaustively – which can reduce intrusive advertising volume.',
        'Ranking purchase intent after a single completed visit could enable lighter-touch, better-timed assistance – surfacing relevant support early rather than retargeting exhaustively – and might thereby reduce intrusive advertising volume; this is a potential application, not an outcome evaluated here.')
rep(92, 'a visit in progress at the 500 ms pre-click cutoff is clipped there rather than ending naturally, so a "completed" visit immediately before a purchase can be shorter than its unclipped counterpart; the share of clipped visits and its effect on the first1 result have not been quantified.',
        'a visit in progress at the 500 ms pre-click cutoff is truncated there rather than ending naturally (67 of the 746 Buy first visits, Sect. 3.2), so a "completed" visit immediately before a purchase can be shorter than its untruncated counterpart; the effect of these truncated visits on the first1 result has not been isolated.')
rep(94, 'does not demonstrate an advantage over a linear model given the same two inputs at that horizon',
        'does not demonstrate an advantage over a linear model given the same two information sources at that horizon')
rep(94, 'the leakage-free ensemble of NeuroClick and the dwell-plus-propensity baseline', 'the parameter-free average ensemble of NeuroClick and the dwell-plus-propensity baseline')

# ---- Table 7: give the implemented propensity rule
tbl7 = ch[83]; assert tbl7.tag == w('tbl')
cell = tbl7.findall(w('tr'))[5].findall(w('tc'))[1]
ts = [t for t in cell.iter(w('t')) if t.text]; full = ''.join(t.text for t in ts)
assert full.startswith('Beta shrinkage toward training prevalence'), full[:60]
ts[0].text = ('p(product) = (nBuy + m × π) / (n + m), where n is the number of training sequences for the product, nBuy its Buy count, '
              'π the training-partition Buy prevalence and m = 10 (beta shrinkage toward prevalence); cross-fitted with 5-fold GroupKFold on '
              'participant (Sect. 3.3) so no bag\'s own label enters its own propensity value.')
for t in ts[1:]: t.text = ''
print('Table 7 propensity rule written')

# ---- Table 2: keep the whole table on one page (keepNext on every row but the last)
tbl2 = ch[48]; assert tbl2.tag == w('tbl'); rows = tbl2.findall(w('tr')); n = 0
for r in rows[:-1]:
    for p in r.iter(w('p')):
        pPr = p.find(w('pPr'))
        if pPr is None: pPr = etree.Element(w('pPr')); p.insert(0, pPr)
        if pPr.find(w('keepNext')) is None:
            ps = pPr.find(w('pStyle')); idx = list(pPr).index(ps) + 1 if ps is not None else 0
            pPr.insert(idx, etree.Element(w('keepNext'))); n += 1
print('Table 2 keepNext paragraphs:', n)
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print('abstract words:', len(ptext(ch[5]).replace('Abstract.', '').split()))
