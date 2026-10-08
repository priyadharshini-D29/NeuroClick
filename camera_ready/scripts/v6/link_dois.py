"""Wrap each 'doi:10....' token (and the GitHub URL) in a Word hyperlink whose target is https://doi.org/...,
keeping the visible text and run formatting unchanged. Word's PDF export then emits link annotations."""
import os, re, copy
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'; R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PR = 'http://schemas.openxmlformats.org/package/2006/relationships'
def w(t): return '{%s}%s' % (W, t)
BASE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(BASE, 'unpacked', 'word')
tree = etree.parse(os.path.join(U, 'document.xml')); root = tree.getroot()
rels = etree.parse(os.path.join(U, '_rels', 'document.xml.rels')); rroot = rels.getroot()
nid = max(int(r.get('Id')[3:]) for r in rroot if r.get('Id', '').startswith('rId')) + 1
PAT = re.compile(r'(doi:10\.\S+?)(?=[\s]|$)|(https://github\.com/[^\s.]+(?:\.[^\s.]+)*?)(?=\.?(?:\s|$))')
n = 0
for p in list(root.iter(w('p'))):
    for r in list(p.findall(w('r'))):
        t = r.find(w('t'))
        if t is None or not t.text: continue
        m = PAT.search(t.text)
        if not m: continue
        token = m.group(0); url = 'https://doi.org/' + token[4:] if token.startswith('doi:') else token
        before, after = t.text[:m.start()], t.text[m.end():]
        rPr = r.find(w('rPr'))
        def mkrun(text):
            nr = etree.Element(w('r'))
            if rPr is not None: nr.append(copy.deepcopy(rPr))
            nt = etree.SubElement(nr, w('t')); nt.text = text; nt.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve'); return nr
        rid = f'rId{nid}'; nid += 1
        etree.SubElement(rroot, '{%s}Relationship' % PR, Id=rid, Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', Target=url, TargetMode='External')
        h = etree.Element(w('hyperlink')); h.set('{%s}id' % R, rid); h.append(mkrun(token))
        idx = list(p).index(r); p.remove(r)
        parts = ([mkrun(before)] if before else []) + [h] + ([mkrun(after)] if after else [])
        for k, el in enumerate(parts): p.insert(idx + k, el)
        n += 1; print(f'{token}  ->  {url}')
tree.write(os.path.join(U, 'document.xml'), xml_declaration=True, encoding='UTF-8', standalone=True)
rels.write(os.path.join(U, '_rels', 'document.xml.rels'), xml_declaration=True, encoding='UTF-8', standalone=True)
print('hyperlinks added:', n)
