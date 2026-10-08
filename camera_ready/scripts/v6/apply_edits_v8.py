"""v8: (1) DOI links -> doi: form in the reference list; (2) cite Eq. (2) by number; (3) classical/CatBoost hyperparameters."""
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
    assert len(hits) == 1 and hits[0].text.count(old) == 1, (idx, old[:50], len(hits), ptext(p)[:80])
    hits[0].text = hits[0].text.replace(old, new); print(f'[{idx}] ok: {old[:60]}')

# (1) reference list: https://doi.org/... -> doi:...
n = 0
for p in ch:
    if p.tag != w('p'): continue
    for t in p.iter(w('t')):
        if t.text and 'https://doi.org/' in t.text:
            n += t.text.count('https://doi.org/'); t.text = t.text.replace('https://doi.org/', 'doi:')
print('DOI links converted:', n)
assert n == 20, n

# (2) Eq. (2) cited by number
assert ptext(ch[37]).startswith('Eq. (1) has the form')
rep(37, "final Buy/NoBuy label (Sect. 3.5) rather than visit-specific event targets",
        "final Buy/NoBuy label (Eq. (2), Sect. 3.5) rather than visit-specific event targets")
assert ptext(ch[43]).startswith('a class-weighted binary cross-entropy')
rep(43, "a class-weighted binary cross-entropy on the cumulative risk of Eq. (1), where ",
        "Eq. (2) is a class-weighted binary cross-entropy on the cumulative risk of Eq. (1), in which ")

# (3) classical baseline and CatBoost hyperparameters (values from results/*/run_config.json and script 03)
rep(43, "which differs from the full model only in that its behaviour encoder omits the propensity term.",
        "which differs from the full model only in that its behaviour encoder omits the propensity term. "
        "Every classical baseline standardises its inputs on the training partition and uses L2-regularised logistic "
        "regression (C = 1, balanced class weights); for the EEG, eye-tracking and fused inputs an ANOVA F-score filter, "
        "fitted on the training partition, keeps the top 25% of features. CatBoost uses 300 iterations, depth 5, "
        "learning rate 0.05 and balanced class weights behind the same filter. MCC thresholds for the classical models "
        "come from 3-fold participant-grouped inner cross-validation.")
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print('remaining https://doi.org in document:', sum((t.text or '').count('https://doi.org') for t in root.iter(w('t'))))
