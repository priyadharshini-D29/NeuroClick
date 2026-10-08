import fitz, sys
d = fitz.open(sys.argv[1])
for i, pg in enumerate(d):
    blocks = pg.get_text('dict')['blocks']
    bottom = max([b['bbox'][3] for b in blocks] + [0])
    keys = [k for k in ['Fig. 1.', 'Fig. 2.', 'Table 2.', 'Fig. 3.', 'Table 3.', 'Fig. 4.', 'Table 4.', 'Fig. 5.', 'Table 5.', 'Fig. 6.', 'Table 6.', 'Fig. 7.', 'Table 7.', '5 Discussion', '6 Limitations', '7 Conclusion', 'References', '22. Prokhorenkova'] if k in pg.get_text()]
    print(f'p{i+1}: content ends {bottom:4.0f} / {pg.rect.height - 68:4.0f} pt -> gap {pg.rect.height - 68 - bottom:4.0f} pt  {keys}')
print('pages', d.page_count)
