import sys, zipfile
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
z = zipfile.ZipFile(sys.argv[1]); d = etree.fromstring(z.read('word/document.xml'))
for p in d.iter(w('p')):
    ps = p.find(w('pPr') + '/' + w('pStyle')); sid = ps.get(w('val')) if ps is not None else 'Normal'
    runs = []
    for r in p.findall(w('r')):
        t = ''.join(x.text or '' for x in r.iter(w('t'))); rp = r.find(w('rPr'))
        b = rp is not None and rp.find(w('b')) is not None; i = rp is not None and rp.find(w('i')) is not None
        if t: runs.append(('**' if b else '') + ('_' if i else '') + t + ('_' if i else '') + ('**' if b else ''))
    txt = ''.join(runs)
    if txt.strip(): print(f'[{sid}] {txt[:300]}')
