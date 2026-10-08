import re, sys, zipfile, io
from lxml import etree
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(t): return '{%s}%s' % (W, t)
tw = lambda v: round(int(v) / 567, 2)
for path in sys.argv[1:]:
    z = zipfile.ZipFile(path)
    doc = etree.fromstring(z.read('word/document.xml'))
    sect = doc.find('.//' + w('sectPr')); pg = sect.find(w('pgSz')); mar = sect.find(w('pgMar'))
    print(path.split('\\')[-1], '| page', tw(pg.get(w('w'))), 'x', tw(pg.get(w('h'))), '| margins T/B/L/R', tw(mar.get(w('top'))), tw(mar.get(w('bottom'))), tw(mar.get(w('left'))), tw(mar.get(w('right'))),
          '| text area', round(tw(pg.get(w('w'))) - tw(mar.get(w('left'))) - tw(mar.get(w('right'))), 2), 'x', round(tw(pg.get(w('h'))) - tw(mar.get(w('top'))) - tw(mar.get(w('bottom'))), 2))
    s = z.read('word/settings.xml').decode('utf-8', 'replace'); print('  attachedTemplate:', re.findall(r'attachedTemplate[^>]*', s)[:1])
    try:
        a = z.read('docProps/app.xml').decode('utf-8', 'replace'); print('  app.xml Template:', re.findall(r'<Template>(.*?)</Template>', a), 'Application:', re.findall(r'<Application>(.*?)</Application>', a))
    except KeyError: print('  no app.xml')
    st = z.read('word/styles.xml').decode('utf-8', 'replace'); ids = re.findall(r'w:styleId="([^"]+)"', st)
    print('  style ids:', [i for i in ids if not i.startswith(('Table', 'List', 'Toc', 'toc', 'Header', 'Footer', 'Balloon', 'Normal', 'Default'))][:40])
    # body text style: Normal rPr
    m = re.search(r'<w:style [^>]*w:styleId="Normal".*?</w:style>', st, re.S); print('  Normal:', re.findall(r'<w:(rFonts|sz|spacing|jc)[^>]*/>', m.group(0)) if m else None)
