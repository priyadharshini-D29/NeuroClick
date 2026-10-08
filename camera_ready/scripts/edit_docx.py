"""Apply all camera-ready edits to unpacked/word/document.xml and media."""
import re, shutil, copy, os
from lxml import etree
from PIL import Image

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
ns = {'w': W, 'wp': WP, 'a': A, 'r': R}
def w(tag): return '{%s}%s' % (W, tag)

DOC = 'unpacked/word/document.xml'
tree = etree.parse(DOC)
root = tree.getroot()
body = root.find('w:body', ns)
ch = list(body)
rels = etree.parse('unpacked/word/_rels/document.xml.rels').getroot()
relmap = {r.get('Id'): r.get('Target') for r in rels}

def ptext(p):
    return ''.join(t.text or '' for t in p.iter(w('t')))

def set_text(p, text, keep_first_bold_prefix=None):
    """Replace all runs of paragraph p with a single run (optionally a bold lead run)."""
    pPr = p.find(w('pPr'))
    first_r = p.find(w('r'))
    rPr = copy.deepcopy(first_r.find(w('rPr'))) if first_r is not None and first_r.find(w('rPr')) is not None else None
    for c in list(p):
        if c.tag != w('pPr'):
            p.remove(c)
    if keep_first_bold_prefix:
        r = etree.SubElement(p, w('r'))
        rp = etree.SubElement(r, w('rPr')); etree.SubElement(rp, w('b'))
        t = etree.SubElement(r, w('t')); t.text = keep_first_bold_prefix
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r = etree.SubElement(p, w('r'))
    if rPr is not None and not (rPr.find(w('b')) is not None and keep_first_bold_prefix):
        rp = copy.deepcopy(rPr)
        b = rp.find(w('b'))
        if b is not None: rp.remove(b)
        if len(rp): r.append(rp)
    t = etree.SubElement(r, w('t')); t.text = text
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')

def ensure_pPr(p):
    pPr = p.find(w('pPr'))
    if pPr is None:
        pPr = etree.Element(w('pPr')); p.insert(0, pPr)
    return pPr

def add_pPr_flag(p, name):
    pPr = ensure_pPr(p)
    if pPr.find(w(name)) is None:
        el = etree.Element(w(name))
        # schema order: pStyle, keepNext, keepLines, ... jc comes later -> insert after pStyle
        ps = pPr.find(w('pStyle'))
        idx = list(pPr).index(ps) + 1 if ps is not None else 0
        if name == 'keepLines' and pPr.find(w('keepNext')) is not None:
            idx = list(pPr).index(pPr.find(w('keepNext'))) + 1
        pPr.insert(idx, el)

# ---------------------------------------------------------------- sanity: locate by text
def find_idx(prefix, style=None):
    for i, c in enumerate(ch):
        if c.tag == w('p') and ptext(c).startswith(prefix):
            return i
    raise KeyError(prefix)

# ---------------------------------------------------------------- A. figures: swap images + extents + keepNext
newfigs = {'media/image2.png': 'newfigs/fig2.png', 'media/image3.png': 'newfigs/fig3.png',
           'media/image4.png': 'newfigs/fig4.png', 'media/image5.png': 'newfigs/fig5.png',
           'media/image6.png': 'newfigs/fig6.png', 'media/image7.png': 'newfigs/fig7.png'}
for target, src in newfigs.items():
    shutil.copy(src, os.path.join('unpacked/word', target))
for c in ch:
    for d in c.iter(w('drawing')):
        blip = d.find('.//a:blip', ns)
        target = relmap[blip.get('{%s}embed' % R)]
        img = Image.open(os.path.join('unpacked/word', target))
        ratio = img.size[0] / img.size[1]
        ext = d.find('.//wp:extent', ns)
        cx = int(ext.get('cx')); cy = int(round(cx / ratio))
        ext.set('cy', str(cy))
        aext = d.find('.//a:ext', ns)
        if aext is not None:
            aext.set('cx', str(cx)); aext.set('cy', str(cy))
        print('figure', target, 'cx', cx, 'cy', cy, f'({cx/360000:.2f} x {cy/360000:.2f} cm)')
        add_pPr_flag(c, 'keepNext'); add_pPr_flag(c, 'keepLines')

# table captions keep with table; figure captions keep lines
for c in ch:
    if c.tag == w('p'):
        pPr = c.find(w('pPr'))
        st = pPr.find(w('pStyle')).get(w('val')) if pPr is not None and pPr.find(w('pStyle')) is not None else None
        if st == 'tablecaption':
            add_pPr_flag(c, 'keepNext'); add_pPr_flag(c, 'keepLines')
        if st == 'figurecaption':
            add_pPr_flag(c, 'keepLines')

# ---------------------------------------------------------------- B. Table 2 caption style
i = find_idx('Table 2.')
pPr = ch[i].find(w('pPr'))
pPr.find(w('pStyle')).set(w('val'), 'tablecaption')
jc = pPr.find(w('jc'))
if jc is not None: pPr.remove(jc)
add_pPr_flag(ch[i], 'keepNext'); add_pPr_flag(ch[i], 'keepLines')
print('Table 2 caption restyled')

# ---------------------------------------------------------------- C. tables: widths, header repeat, no justified cells
tables = [c for c in ch if c.tag == w('tbl')]
def set_grid(tbl, widths):
    grid = tbl.find(w('tblGrid'))
    for g in list(grid): grid.remove(g)
    for wd in widths:
        etree.SubElement(grid, w('gridCol')).set(w('w'), str(wd))
    tblW = tbl.find(w('tblPr')).find(w('tblW'))
    tblW.set(w('w'), str(sum(widths))); tblW.set(w('type'), 'dxa')
    for tr in tbl.findall(w('tr')):
        for k, tc in enumerate(tr.findall(w('tc'))):
            tcW = tc.find(w('tcPr')).find(w('tcW'))
            tcW.set(w('w'), str(widths[k])); tcW.set(w('type'), 'dxa')
    lay = tbl.find(w('tblPr')).find(w('tblLayout'))
    if lay is None:
        lay = etree.SubElement(tbl.find(w('tblPr')), w('tblLayout'))
    lay.set(w('type'), 'fixed')

# Table 2 (index 1): widen PR AUC / ROC AUC columns
set_grid(tables[1], [1150, 1980, 1150, 1000, 1180, 1080, 1000])
for tbl in tables:
    rows = tbl.findall(w('tr'))
    for ri, tr in enumerate(rows):
        trPr = tr.find(w('trPr'))
        if trPr is None:
            trPr = etree.Element(w('trPr')); tr.insert(0, trPr)
        if trPr.find(w('cantSplit')) is None:
            trPr.insert(0, etree.Element(w('cantSplit')))
        if ri == 0 and trPr.find(w('tblHeader')) is None:
            trPr.append(etree.Element(w('tblHeader')))
        for p in tr.iter(w('p')):
            jc = p.find('w:pPr/w:jc', ns)
            if jc is not None and jc.get(w('val')) == 'both':
                jc.set(w('val'), 'left')
print('tables: header repeat + cantSplit set; Table 2 widths updated')

# ---------------------------------------------------------------- D. Table 6 restructure (drop redundant column 2)
t6 = tables[5]
for tr in t6.findall(w('tr')):
    tcs = tr.findall(w('tc'))
    tr.remove(tcs[1])
set_grid(t6, [1200, 1900, 1200, 1900, 1200])
newvals = [
    ['Horizon', 'Δ vs NeuroClick', 'Holm p', 'Δ vs baseline', 'Holm p'],
    ['first1', '+.0080', '1.00', 'n.s.', '0.69'],
    ['first2', '+.0010', '1.00', '+.0187', '0.022'],
    ['first3', '+.0020', '1.00', 'n.s.', '0.17'],
    ['full', '+.0637', '<0.0001', 'n.s.', '1.00'],
]
for tr, vals in zip(t6.findall(w('tr')), newvals):
    for tc, v in zip(tr.findall(w('tc')), vals):
        ts = list(tc.iter(w('t')))
        ts[0].text = v
        for extra in ts[1:]:
            extra.getparent().getparent().remove(extra.getparent())
i = find_idx('Table 6.')
set_text(ch[i], ' Average ensemble per horizon: participant-mean PR AUC difference of the ensemble relative to each '
               'component (Fig. 7). Differences are given only where they survive Holm correction; n.s. = not '
               'significant. The stacked variant is reported in the text.', keep_first_bold_prefix='Table 6.')
print('Table 6 restructured')

# ---------------------------------------------------------------- E. text edits
i = find_idx('Conflict of interest.')
set_text(ch[i], 'Disclosure of Interests. The authors have no competing interests to declare.')

i = find_idx('Fig. 2.')
set_text(ch[i], ' NeuroClick architecture, unrolled per completed visit. Browsing behaviour (with the cross-fitted '
               'propensity logit, Sect. 3.3) is always present; EEG and eye tracking enter through optional '
               'validity-gated residual branches. The fused visit representation z(j) feeds a unidirectional GRU whose '
               'sigmoid head yields the visit score h(j), and Eq. (1) aggregates these scores into the nondecreasing '
               'cumulative risk p(t). The timeline marks the audited cutoff: each completed visit updates the risk from '
               'its own history only.', keep_first_bold_prefix='Fig. 2.')

i = find_idx('a class-weighted binary cross-entropy')
p = ch[i]
for t in p.iter(w('t')):
    if t.text and 'in response to review' in t.text:
        t.text = t.text.replace('were added after the primary first1 family in response to review and are Holm-corrected',
                                'were specified after the primary first1 family and are Holm-corrected')
    if t.text and 'plus the identical cross-fitted propensity (Sect. 4.1)' in t.text:
        t.text = t.text.replace('plus the identical cross-fitted propensity (Sect. 4.1)',
                                'plus the identical cross-fitted propensity (Sect. 4.1; labelled dwell + propensity in the tables and figures)')
assert 'in response to review' not in ptext(p) and 'labelled dwell + propensity' in ptext(p)
print('Sect. 3.5 text updated')

# ---------------------------------------------------------------- F. references
refs = {
 'Hakim, A.': 'Hakim, A., Klorfeld, S., Sela, T., Friedman, D., Shabat-Simon, M., Levy, D.J.: Machines learn neuromarketing: improving preference prediction from self-reports using multiple EEG measures and machine learning. International Journal of Research in Marketing 38(3), 770–791 (2021). https://doi.org/10.1016/j.ijresmar.2020.10.005',
 'Khushaba, R.N.': 'Khushaba, R.N., Wise, C., Kodagoda, S., Louviere, J., Kahn, B.E., Townsend, C.: Consumer neuroscience: assessing the brain response to marketing stimuli using electroencephalogram (EEG) and eye tracking. Expert Systems with Applications 40(9), 3803–3812 (2013). https://doi.org/10.1016/j.eswa.2012.12.095',
 'Quiles Perez, M.': 'Quiles Pérez, M., Martínez Beltrán, E.T., López Bernal, S., Horna Prat, E., Montesano Del Campo, L., Fernández Maimó, L., Huertas Celdrán, A.: Data fusion in neuromarketing: multimodal analysis of biosignals, lifecycle stages, current advances, datasets, trends, and challenges. Information Fusion 105, 102231 (2024). https://doi.org/10.1016/j.inffus.2024.102231',
 'Lee, C., Zame': 'Lee, C., Zame, W.R., Yoon, J., van der Schaar, M.: DeepHit: a deep learning approach to survival analysis with competing risks. In: Proceedings of the 32nd AAAI Conference on Artificial Intelligence (AAAI-18), pp. 2314–2321. AAAI Press, New Orleans (2018). https://doi.org/10.1609/aaai.v32i1.11842',
 'Cho, K.': 'Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., Bengio, Y.: Learning phrase representations using RNN encoder–decoder for statistical machine translation. In: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1724–1734. Association for Computational Linguistics, Doha (2014). https://doi.org/10.3115/v1/D14-1179',
 'Georgiadis, K., et al.': 'Georgiadis, K., Kalaganis, F.P., Riskos, K., Matta, E., Oikonomou, V.P., Yfantidou, I., Chantziaras, D., Pantouvakis, K., Nikolopoulos, S., Laskaris, N.A., Kompatsiaris, I.: NeuMa – the absolute neuromarketing dataset en route to an holistic understanding of consumer behaviour. Scientific Data 10, 508 (2023). https://doi.org/10.1038/s41597-023-02392-9',
 'Loshchilov, I.': 'Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: 7th International Conference on Learning Representations (ICLR 2019), New Orleans (2019). https://doi.org/10.48550/arXiv.1711.05101',
 'Prokhorenkova, L.': 'Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A.: CatBoost: unbiased boosting with categorical features. In: Advances in Neural Information Processing Systems 31 (NeurIPS 2018), pp. 6638–6648. Curran Associates, Red Hook (2018).',
 'Afshar, M.P.': 'Afshar, M.P., Azimi, A.: EEG-based consumer behaviour prediction: an exploration from classical machine learning to graph neural networks. arXiv preprint arXiv:2509.21567 (2025). https://doi.org/10.48550/arXiv.2509.21567',
}
done = set()
for c in ch:
    if c.tag != w('p'): continue
    pPr = c.find(w('pPr'))
    st = pPr.find(w('pStyle')).get(w('val')) if pPr is not None and pPr.find(w('pStyle')) is not None else None
    if st != 'referenceitem': continue
    txt = ptext(c)
    for key, new in refs.items():
        if txt.startswith(key):
            set_text(c, new); done.add(key); txt = new
            break
    # en dash in page ranges "427-435 (" ; keep DOIs untouched
    for t in c.iter(w('t')):
        if t.text:
            t.text = re.sub(r'(\d)-(\d+) \(', lambda m: f'{m.group(1)}–{m.group(2)} (', t.text)
print('references updated:', len(done), 'of', len(refs), sorted(set(refs) - done))

tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
print('document.xml written')
