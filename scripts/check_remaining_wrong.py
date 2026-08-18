import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
with open('data/processed/figure_descriptions.json', encoding='utf-8') as f:
    figs = json.load(f)
# ch=13 (electronics) should also be exempt from circuit check
ELEC = {10, 11, 12, 13}
MARKERS = ['সার্কিট', 'ব্যাটারি', 'প্রতিরোধক', 'বৈদ্যুতিক সার্কিট', 'NEEDS_REDESCRIPTION']
remaining = []
for key, val in figs.items():
    ch = val.get('chapter_num', 0)
    desc = val.get('description', '')
    if ch not in ELEC and any(m in desc for m in MARKERS):
        remaining.append((ch, val.get('book_page'), key, desc[:120]))

print(f'Remaining wrong: {len(remaining)}')
for ch, pg, key, desc in remaining:
    print(f'  ch={ch} pg={pg} key={key}')
    print(f'  desc: {desc}')
    print()

# Also check what ch=13 ones look like (they should be ok now)
print('\nCh=13 circuit descriptions (should be valid - electronics chapter):')
for key, val in figs.items():
    if val.get('chapter_num') == 13 and any(m in val.get('description','') for m in MARKERS):
        print(f'  {key}: {val["description"][:80]}')
