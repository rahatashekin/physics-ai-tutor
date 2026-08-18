"""
Fix section mislabeling in LanceDB parent_chunks table.

Problem: Some chunks have section_num values that are formula fragments
(e.g., "13.6" for chapter 5) because the pipeline parsed equation
numbers/values as section headers.

Fix: For each chunk where section_num major digit != chapter_num,
infer correct section from nearest valid neighbors in the same chapter.

Does NOT require re-embedding — only fixes metadata.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import lancedb
import pandas as pd
import re

db  = lancedb.connect('data/lancedb')
tbl = db.open_table('parent_chunks')
df  = tbl.to_pandas()

print(f'Total chunks: {len(df)}')

# ── Identify mislabeled chunks ──────────────────────────────────────────────
def is_mislabeled(row):
    """Return True if section_num major digit doesn't match chapter_num."""
    sec = str(row.get('section_num', ''))
    ch  = row.get('chapter_num', 0)
    if not sec or sec in ['', 'nan', 'None']:
        return False
    # Extract major number (before first dot)
    m = re.match(r'^(\d+)', sec)
    if not m:
        return False
    try:
        major = int(m.group(1))
        return major != ch and major > 0
    except:
        return False

# ── Fix: inherit nearest valid section ──────────────────────────────────────
# For each chapter, sort chunks by book_page_start.
# For mislabeled chunks, find the nearest valid-labeled neighbor.

def get_valid_section(chapter_num: int, page_start: int, df: pd.DataFrame) -> str:
    """Find nearest valid section for a given chapter and page."""
    # Note: mislabeled_mask needs to be calculated per-df in the loop below
    ch_df = df[(df['chapter_num'] == chapter_num) & (~df.apply(is_mislabeled, axis=1))].copy()
    ch_df = ch_df.sort_values('book_page_start')
    if ch_df.empty:
        return str(chapter_num)

    # Find closest chunk by page
    ch_df['dist'] = (ch_df['book_page_start'] - page_start).abs()
    nearest = ch_df.nsmallest(1, 'dist')
    return str(nearest.iloc[0]['section_num'])

for table_name in ['parent_chunks', 'child_chunks']:
    tbl = db.open_table(table_name)
    df  = tbl.to_pandas()
    print(f'\n[{table_name}] Total chunks: {len(df)}')

    mislabeled_mask = df.apply(is_mislabeled, axis=1)
    mislabeled = df[mislabeled_mask].copy()
    print(f'[{table_name}] Mislabeled chunks: {len(mislabeled)}')

    if len(mislabeled) == 0:
        print(f'[{table_name}] Nothing to fix.')
        continue

    # Show breakdown by chapter
    for ch in sorted(mislabeled['chapter_num'].unique()):
        ch_bad = mislabeled[mislabeled['chapter_num'] == ch]
        bad_secs = sorted(ch_bad['section_num'].unique())
        print(f'  ch={ch}: {len(ch_bad)} chunks with sections {bad_secs[:8]}')

    # Fix: inherit nearest valid section from same chapter
    fixed_rows = []
    for idx, row in mislabeled.iterrows():
        old_sec = row['section_num']
        new_sec = get_valid_section(row['chapter_num'], row['book_page_start'], df)
        fixed_rows.append({'index': idx, 'old': old_sec, 'new': new_sec})
        df.at[idx, 'section_num'] = new_sec

    print(f'Fixed {len(fixed_rows)} chunks. Examples:')
    for r in fixed_rows[:5]:
        print(f'  idx={r["index"]}: {r["old"]} → {r["new"]}')

    print(f'Writing fixed [{table_name}] back to LanceDB...')
    tbl.delete('1=1')
    tbl.add(df)

    # Verify
    df2 = db.open_table(table_name).to_pandas()
    still_bad = df2.apply(is_mislabeled, axis=1).sum()
    print(f'[{table_name}] Remaining mislabeled after fix: {still_bad}')

print('\n[DONE] Section mislabeling fix complete (parent + child).')
