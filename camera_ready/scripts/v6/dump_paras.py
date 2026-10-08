import sys
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
root = etree.parse(sys.argv[1]).getroot()
body = root.find(w('body'))
out = []
for i, c in enumerate(body):
    tag = c.tag.split('}')[1]
    if tag == 'p':
        txt = ''.join(t.text or '' for t in c.iter(w('t')))
        nruns = len(c.findall(w('r')))
        out.append(f"[{i}] p runs={nruns} | {txt}")
    elif tag == 'tbl':
        for ri, row in enumerate(c.findall(w('tr'))):
            cells = ['/'.join(''.join(t.text or '' for t in p.iter(w('t'))) for p in cell.findall(w('p'))) for cell in row.findall(w('tc'))]
            out.append(f"[{i}] tbl r{ri} | " + ' || '.join(cells))
    else:
        out.append(f"[{i}] {tag}")
open(sys.argv[2], 'w', encoding='utf-8').write('\n'.join(out))
print(len(out), 'lines')
