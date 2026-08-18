"""
Fix wrong figure descriptions for ch=8 optics chapter.
1. Updates figure_descriptions.json
2. Updates corresponding parent_chunks in LanceDB
"""
import json, io, sys, pathlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import lancedb
import pandas as pd

# ── Load figure descriptions ────────────────────────────────────────────────
FIG_JSON = pathlib.Path('data/processed/figure_descriptions.json')
with open(FIG_JSON, encoding='utf-8') as f:
    figs = json.load(f)

# ── Corrections for ch=8 wrong/imprecise descriptions ──────────────────────
# Format: figure_key → correct description
# Source: what these figures actually show based on book page + context
CORRECTIONS = {
    # pg=236 (PDF 241): ৮.৮.৪ পাহাড়ি রাস্তার বাঁকে আয়নার ব্যবহার
    "fig_page241_0.png": (
        "এটি অধ্যায় ৮ (আলোর প্রতিফলন) এর ৮.৮.৪ ধারার চিত্র। "
        "পাহাড়ি রাস্তার বিপজ্জনক বাঁকে (প্রায় সমকোণ) একটি বড় উত্তল আয়না "
        "৪৫° কোণে বসানো হয়েছে। এই আয়নার সাহায্যে চালকেরা রাস্তার অন্যদিক থেকে "
        "আসা যানবাহন দেখতে পায় এবং দুর্ঘটনা এড়ানো সম্ভব হয়।"
    ),
    # pg=234 figures: section 8.8 আয়নার ব্যবহার — mirror reflection
    "fig_page239_0.png": (
        "অধ্যায় ৮ (আলোর প্রতিফলন) এর চিত্র। একটি মেয়ে সমতল আয়নার সামনে "
        "দাঁড়িয়ে নিজের প্রতিবিম্ব দেখছে। সমতল আয়না প্রতিফলনের নীতিতে কাজ করে — "
        "আপতন কোণ = প্রতিফলন কোণ। প্রতিবিম্বটি সোজা, সমান আকারের এবং আয়নার পেছনে অবস্থিত।"
    ),
    "fig_page239_1.png": (
        "অধ্যায় ৮ (আলোর প্রতিফলন) এর চিত্র। আয়নার প্রতিফলনের রশ্মি চিত্র। "
        "বস্তু থেকে আসা আলোকরশ্মি আয়নায় প্রতিফলিত হয়ে চোখে পৌঁছায় — "
        "এভাবে আয়নায় প্রতিবিম্ব দেখা যায়।"
    ),
    # pg=225 (ch=8): ছেলে চামচে প্রতিবিম্ব দেখছে (অবতল/উত্তল আয়নার উদাহরণ)
    "fig_page232_0.png": (
        "অধ্যায় ৮ (আলোর প্রতিফলন) এর চিত্র। একটি ছেলে চামচের (spoon) দুই পাশে "
        "নিজের প্রতিবিম্ব পরীক্ষা করছে। চামচের ভেতরের বাঁকা পৃষ্ঠ অবতল আয়না হিসেবে "
        "কাজ করে (উল্টো ছবি দেখায়) এবং বাইরের পৃষ্ঠ উত্তল আয়না হিসেবে কাজ করে "
        "(ছোট সোজা ছবি দেখায়)।"
    ),
}

# Also fix other circuit-described ch=8 figures by looking at their page context
# Check which wrong ones we haven't manually corrected
CIRCUIT_MARKERS = ['সার্কিট', 'ব্যাটারি', 'প্রতিরোধক']
unfixed = []
for key, val in figs.items():
    if val.get('chapter_num') == 8:
        desc = val.get('description', '')
        if any(m in desc for m in CIRCUIT_MARKERS) and key not in CORRECTIONS:
            unfixed.append((key, val.get('book_page'), desc[:100]))

print(f'Ch=8 circuit-described figures remaining after corrections: {len(unfixed)}')
for key, pg, d in unfixed:
    print(f'  {key} pg={pg}: {d[:80]}')

# ── Apply corrections to figure_descriptions.json ──────────────────────────
changed = 0
for key, new_desc in CORRECTIONS.items():
    if key in figs:
        old_desc = figs[key].get('description', '')
        figs[key]['description'] = new_desc
        print(f'\nCORRECTED {key}:')
        print(f'  OLD: {old_desc[:80]}')
        print(f'  NEW: {new_desc[:80]}')
        changed += 1
    else:
        print(f'WARNING: {key} not found in figure_descriptions.json')

with open(FIG_JSON, 'w', encoding='utf-8') as f:
    json.dump(figs, f, ensure_ascii=False, indent=2)
print(f'\nSaved {changed} corrections to figure_descriptions.json')

# ── Update LanceDB parent_chunks ────────────────────────────────────────────
db  = lancedb.connect('data/lancedb')
tbl = db.open_table('parent_chunks')
df  = tbl.to_pandas()

update_count = 0
for key, new_desc in CORRECTIONS.items():
    # Figure chunks have text like "[চিত্র | অধ্যায় N | পৃষ্ঠা P]\n<description>"
    # Match by looking for the figure key or matching old description prefix
    old_desc_prefix = figs.get(key, {}).get('description', '')[:30]
    if not old_desc_prefix:
        continue

    # Find matching chunk by content_type=figure and description overlap
    # The chunk text format: "[চিত্র | অধ্যায় 8 | পৃষ্ঠা 236]\n<description>"
    pg = figs.get(key, {}).get('book_page')
    ch = figs.get(key, {}).get('chapter_num')
    if pg is None or ch is None:
        continue

    mask = (
        (df['content_type'] == 'figure') &
        (df['chapter_num'] == ch) &
        (df['book_page_start'] == pg)
    )
    matched = df[mask]
    if matched.empty:
        print(f'No DB chunk found for {key} (ch={ch} pg={pg})')
        continue

    for idx in matched.index:
        old_text = df.at[idx, 'text']
        # Replace description part (after the header line)
        header_end = old_text.find('\n')
        if header_end >= 0:
            new_text = old_text[:header_end + 1] + new_desc
        else:
            new_text = new_desc
        df.at[idx, 'text'] = new_text
        update_count += 1
        print(f'Updated DB chunk for {key} (row idx={idx})')

if update_count > 0:
    # Overwrite the table with updated data
    tbl.delete('1=1')  # Clear all rows
    tbl.add(df)        # Re-add with corrections
    print(f'\nUpdated {update_count} DB chunks in parent_chunks')
else:
    print('\nNo DB chunks needed updating (or not found)')

print('\n[DONE] Figure description fix complete.')
