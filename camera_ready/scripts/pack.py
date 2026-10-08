import zipfile, os, sys
src, out = 'unpacked', sys.argv[1]
if os.path.exists(out): os.remove(out)
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(src, '[Content_Types].xml'), '[Content_Types].xml')
    for dp, dn, fn in os.walk(src):
        for f in fn:
            full = os.path.join(dp, f)
            arc = os.path.relpath(full, src).replace('\\', '/')
            if arc == '[Content_Types].xml': continue
            z.write(full, arc)
print('packed', out, os.path.getsize(out))
