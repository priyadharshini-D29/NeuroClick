import fitz, sys
d = fitz.open(sys.argv[1]); t = ' '.join(' '.join(p.get_text().split()) for p in d)
print('typo left:', t.count('outcome-blind moment, The'), '| fixed:', t.count('outcome-blind moment. The'))
for i in range(d.page_count):
    pt = d[i].get_text()
    for key in ['4.6', 'Table 7.', 'Timestamp/boundary', '5 Discussion', '7 Conclusion', 'Declarations', 'References', '22. Prokhorenkova']:
        if key in pt: print(f'p{i+1}: {key}')
for pg in range(d.page_count):
    for b in d[pg].get_text('dict')['blocks']:
        for l in b.get('lines', []):
            for sp in l['spans']:
                for key in ['Data availability.', 'Code availability.', 'Acknowledgements.', 'Author contributions.', 'Ethics statement.', 'Disclosure of Interests.', 'Reproducibility checklist', 'Table 7.']:
                    if sp['text'].strip().startswith(key) or sp['text'].strip() == key:
                        print(f'p{pg+1} "{sp["text"].strip()[:40]}" font={sp["font"]} size={sp["size"]:.1f}')
print('pages', d.page_count)
