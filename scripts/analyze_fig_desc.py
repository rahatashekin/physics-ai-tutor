"""Analyze and fix wrong figure descriptions for ch=8 optics chapter."""
import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('data/processed/figure_descriptions.json', encoding='utf-8') as f:
    figs = json.load(f)

CIRCUIT_MARKERS = ['সার্কিট', 'ব্যাটারি', 'প্রতিরোধক']

print('=== CH=8 figures ===')
wrong_keys = []
for key, val in figs.items():
    ch = val.get('chapter_num', 0)
    if ch == 8:
        desc = val.get('description', '')
        pg   = val.get('book_page', '?')
        is_wrong = any(m in desc for m in CIRCUIT_MARKERS)
        print(f'  key={key} pg={pg} wrong={is_wrong}')
        print(f'  desc: {desc[:150]}')
        print()
        if is_wrong:
            wrong_keys.append(key)

print(f'Total ch=8 wrong: {len(wrong_keys)}')
print('Wrong keys:', wrong_keys)
