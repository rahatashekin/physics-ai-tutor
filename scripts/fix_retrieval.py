"""
Q1 deep debug: স্ক্রু Ch1 তে কোথায়? FTS কেন Ch1 থেকে আনছে না?
"""
import io, sys, os, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()

import lancedb
db = lancedb.connect('data/lancedb')
pdf = db.open_table('parent_chunks').to_pandas()

print("=" * 60)
print("Q1: Ch1 এ স্ক্রু কোন chunks এ, কোথায়?")
print("=" * 60)
ch1 = pdf[pdf['chapter_num'] == 1]
for _, row in ch1.iterrows():
    text = str(row['text'])
    if 'স্ক্রু' in text:
        pos = text.find('স্ক্রু')
        in_2500 = pos < 2500
        print(f"✅ Ch1 {row['content_type']} p{row['book_page_start']}:")
        print(f"   Total len: {len(text)} | pos: {pos} | in_first_2500: {in_2500}")
        print(f"   Context: ...{text[max(0,pos-40):pos+80]}...")
        print()

print("=" * 60)
print("Q1: Ch1 এ 0.01 কোন chunks এ?")
print("=" * 60)
for _, row in ch1.iterrows():
    text = str(row['text'])
    if '0.01' in text:
        pos = text.find('0.01')
        in_2500 = pos < 2500
        print(f"✅ Ch1 {row['content_type']} p{row['book_page_start']}:")
        print(f"   Total len: {len(text)} | pos: {pos} | in_first_2500: {in_2500}")
        print(f"   Context: ...{text[max(0,pos-40):pos+80]}...")
        print()

# Also check what FTS returns for স্ক্রু
fts_t = db.open_table('parent_chunks')
fts_r = fts_t.search('স্ক্রু', query_type='fts').limit(8).to_pandas()
print("=" * 60)
print(f"FTS 'স্ক্রু': {len(fts_r)} results")
print("=" * 60)
for _, row in fts_r.iterrows():
    print(f"  Ch{row['chapter_num']} {row['content_type']} p{row['book_page_start']}: {row['text'][:80].replace(chr(10),' ')}")
