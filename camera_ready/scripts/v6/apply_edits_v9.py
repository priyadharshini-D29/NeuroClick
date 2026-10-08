"""v9: add the three issue numbers CrossRef records (Yadava 76(18), Kalaganis 12(1), NeuMa 10(1))."""
import os
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(BASE, 'unpacked', 'word', 'document.xml')
tree = etree.parse(DOC); root = tree.getroot()
edits = [('Multimedia Tools and Applications 76, 19087', 'Multimedia Tools and Applications 76(18), 19087'),
         ('Brain Informatics 12, 23 (2025)', 'Brain Informatics 12(1), 23 (2025)'),
         ('Scientific Data 10, 508 (2023)', 'Scientific Data 10(1), 508 (2023)')]
for old, new in edits:
    hits = [t for t in root.iter(w('t')) if t.text and old in t.text]
    assert len(hits) == 1, (old, len(hits))
    hits[0].text = hits[0].text.replace(old, new); print('ok:', new)
tree.write(DOC, xml_declaration=True, encoding='UTF-8', standalone=True)
