"""Fig 2: the ORIGINAL single-row layout (make_fig2_singlerow.py, unchanged geometry), rendered at the full
text width (6.27 in = 15.92 cm) instead of 4.8 in, with every type size scaled by the same 1.306 factor so the
drawing is identical but 31% larger on the page. Output: newfigs/fig2.png at 600 dpi."""
import io, os, re, runpy
D = os.path.dirname(os.path.abspath(__file__))
src = io.open(r'D:\PRIYA\ICAIN\camera_ready_build\scripts\make_fig2_singlerow.py', encoding='utf-8').read()
K = 6.27 / 4.8
src = src.replace("fig = plt.figure(figsize=(4.8, 2.23), dpi=600)", f"fig = plt.figure(figsize=(6.27, {2.23 * K:.3f}), dpi=600)")
src = src.replace("H, T, S = 5.0, 4.4, 3.9", f"H, T, S = {5.0 * K:.2f}, {4.4 * K:.2f}, {3.9 * K:.2f}")
src = src.replace("'font.size': 5,", f"'font.size': {5 * K:.2f},")
src = re.sub(r"size=6,", f"size={6 * K:.2f},", src)                      # the × and + glyphs
src = src.replace("os.path.join(OUT, 'fig2_singlerow.png')", "os.path.join(OUT, 'fig2.png')")
assert src.count(f"{5.0 * K:.2f}") >= 1 and "fig2.png" in src
io.open(os.path.join(D, '_fig2_fullwidth_generated.py'), 'w', encoding='utf-8').write(src)
runpy.run_path(os.path.join(D, '_fig2_fullwidth_generated.py'), run_name='__main__')
print(f'type sizes now: header {5.0*K:.1f} pt, titles {4.4*K:.1f} pt, details {3.9*K:.1f} pt (min detail {(3.9-0.6)*K:.1f} pt)')
