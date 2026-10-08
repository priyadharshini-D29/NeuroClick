import sys, os, zipfile
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
    # [Content_Types].xml first, as Word wrote it
    z.write(os.path.join(src, '[Content_Types].xml'), '[Content_Types].xml')
    for dp, dn, fn in os.walk(src):
        for f in fn:
            full = os.path.join(dp, f)
            arc = os.path.relpath(full, src).replace('\\', '/')
            if arc == '[Content_Types].xml': continue
            z.write(full, arc)
print('packed', dst, os.path.getsize(dst))
