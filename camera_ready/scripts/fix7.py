from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(tag): return '{%s}%s' % (W, tag)
DOC = 'unpacked/word/document.xml'
tree = etree.parse(DOC); root = tree.getroot()
ORDER = ['pStyle', 'keepNext', 'keepLines', 'pageBreakBefore', 'framePr', 'widowControl', 'numPr', 'suppressLineNumbers',
         'pBdr', 'shd', 'tabs', 'suppressAutoHyphens', 'kinsoku', 'wordWrap', 'overflowPunct', 'topLinePunct',
         'autoSpaceDE', 'autoSpaceDN', 'bidi', 'adjustRightInd', 'snapToGrid', 'spacing', 'ind']
n = 0
for p in root.iter(w('p')):
    if p.find('.//' + w('drawing')) is None: continue
    pPr = p.find(w('pPr'))
    if pPr is None:
        pPr = etree.Element(w('pPr')); p.insert(0, pPr)
    sp = pPr.find(w('spacing'))
    if sp is None:
        sp = etree.Element(w('spacing'))
        # insert in schema order: after the last existing element that precedes 'spacing'
        idx = 0
        for i, ch in enumerate(list(pPr)):
            tag = ch.tag.split('}')[1]
            if tag in ORDER and ORDER.index(tag) < ORDER.index('spacing'):
                idx = i + 1
        pPr.insert(idx, sp)
    sp.set(w('before'), '160')
    n += 1
print('drawing paragraphs with 8 pt space before:', n)
m = 0
for t in root.iter(w('t')):
    if t.text and 'C1-C68' in t.text:
        t.text = t.text.replace('C1-C68', 'C1–C68'); m += 1
print('C1-C68 fixed:', m)
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
