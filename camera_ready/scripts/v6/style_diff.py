"""Diff style definitions (rPr/pPr) between the conference template and the paper, for the styles the paper uses."""
import sys, zipfile, re
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
def styles(path):
    z = zipfile.ZipFile(path); st = etree.fromstring(z.read('word/styles.xml')); doc = etree.fromstring(z.read('word/document.xml'))
    out = {}
    for s in st.findall(w('style')):
        sid = s.get(w('styleId')); props = {}
        for part in ('pPr', 'rPr'):
            el = s.find(w(part))
            if el is None: continue
            for ch in el:
                tag = ch.tag.split('}')[1]
                attrs = {k.split('}')[1]: v for k, v in ch.attrib.items()}
                if tag == 'rFonts': attrs = {k: v for k, v in attrs.items() if k in ('ascii', 'hAnsi')}
                props[f'{part}.{tag}'] = attrs if attrs else True
        out[sid] = props
    dd = st.find(w('docDefaults')); props = {}
    for ch in dd.iter():
        tag = ch.tag.split('}')[1]
        if ch.attrib: props[tag] = {k.split('}')[1]: v for k, v in ch.attrib.items()}
    out['__docDefaults__'] = props
    used = set()
    for p in doc.iter(w('p')):
        ps = p.find(w('pPr') + '/' + w('pStyle')); used.add(ps.get(w('val')) if ps is not None else 'Normal')
    return out, used
tpl, _ = styles(sys.argv[1]); paper, used = styles(sys.argv[2])
print('styles used in paper:', sorted(used))
for sid in sorted(used) + ['__docDefaults__']:
    a, b = tpl.get(sid), paper.get(sid)
    if a is None: print(f'\n{sid}: NOT IN TEMPLATE'); continue
    diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    print(f'\n{sid}: ' + ('identical' if not diff else 'DIFFERS'))
    for k, (x, y) in sorted(diff.items()): print(f'   {k}: template={x}  paper={y}')
