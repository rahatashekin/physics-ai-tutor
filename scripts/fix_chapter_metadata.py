"""
scripts/fix_chapter_metadata.py
================================
PURPOSE:
    LanceDB তে chapter_num metadata ভুল entries fix করা।
    section_num দেখে chapter_num ঠিক করব।
    
    যেমন: sec=2.8 কিন্তু ch=4 → ch=2 করব।
"""
import io, sys, lancedb, pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

db = lancedb.connect("data/lancedb")

def fix_table(table_name: str):
    print(f"\n{'='*50}")
    print(f"Fixing: {table_name}")
    t  = db.open_table(table_name)
    df = t.to_pandas()
    print(f"  Before: {len(df)} rows")

    # section_num থেকে correct chapter_num infer করি
    # sec="2.6" → chapter 2, sec="3.4" → chapter 3, etc.
    fixes = 0
    for i, row in df.iterrows():
        sec = str(row["section_num"]).strip()
        current_ch = row["chapter_num"]

        # sec="2.6", "2.09", "2.8" → chapter 2
        # sec="3.x" → chapter 3, etc.
        m = None
        import re
        m = re.match(r"^(\d+)\.", sec)
        if m:
            inferred_ch = int(m.group(1))
            if inferred_ch != current_ch and 1 <= inferred_ch <= 15:
                df.at[i, "chapter_num"] = inferred_ch
                fixes += 1
                print(f"  Fix: sec={sec} ch={current_ch} → ch={inferred_ch} pg={row['book_page_start']}")

    if fixes == 0:
        print("  No fixes needed!")
        return

    print(f"\n  Total fixes: {fixes}")

    # table drop করে re-create করি (vectors সহ)
    db.drop_table(table_name)
    db.create_table(table_name, data=df)
    print(f"  Table re-created with {len(df)} rows")
    print(f"  After verification: {db.open_table(table_name).count_rows()} rows")


fix_table("child_chunks")
fix_table("parent_chunks")

print("\nDone! Chapter metadata fixed.")
