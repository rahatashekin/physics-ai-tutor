"""
scripts/fix_chapter_by_page.py
================================
PURPOSE:
    বইয়ের TOC থেকে বের করা exact page range ব্যবহার করে
    child_chunks ও parent_chunks এর chapter_num সঠিক করা।

    কোনো heuristic নেই — শুধু book_page_start দেখে chapter assign।

CHAPTER PAGE RANGES (from TOC, extracted_book.md):
    ch=1  : pages   1 –  31   ভৌত রাশি ও পরিমাপ
    ch=2  : pages  32 –  61   গতি
    ch=3  : pages  62 –  97   বল
    ch=4  : pages  98 – 126   কাজ, ক্ষমতা ও শক্তি
    ch=5  : pages 127 – 158   পদার্থের অবস্থা ও চাপ
    ch=6  : pages 159 – 185   তাপের প্রভাব
    ch=7  : pages 186 – 209   তরঙ্গ ও শব্দ
    ch=8  : pages 210 – 240   আলোর প্রতিফলন
    ch=9  : pages 241 – 269   আলোর প্রতিসরণ
    ch=10 : pages 270 – 297   স্থির বিদ্যুৎ
    ch=11 : pages 298 – 328   চল বিদ্যুৎ
    ch=12 : pages 329 – 345   বিদ্যুতের চৌম্বক ক্রিয়া
    ch=13 : pages 346 – 366   তেজস্ক্রিয়তা ও ইলেকট্রনিকস
"""

import io, sys, lancedb, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────
# Chapter page ranges (inclusive)
# ─────────────────────────────────────────────
CHAPTER_RANGES = [
    (1,   1,   31),
    (2,   32,  61),
    (3,   62,  97),
    (4,   98,  126),
    (5,   127, 158),
    (6,   159, 185),
    (7,   186, 209),
    (8,   210, 240),
    (9,   241, 269),
    (10,  270, 297),
    (11,  298, 328),
    (12,  329, 345),
    (13,  346, 366),
]

def page_to_chapter(page: int) -> int:
    """book_page_start থেকে correct chapter_num বের করো।"""
    for ch, start, end in CHAPTER_RANGES:
        if start <= page <= end:
            return ch
    return -1   # unknown (front matter বা back matter)


def fix_table(db, table_name: str) -> None:
    print(f"\n{'='*55}")
    print(f"Table: {table_name}")
    print('='*55)

    t  = db.open_table(table_name)
    df = t.to_pandas()
    print(f"  Rows: {len(df)}")

    changed = 0
    unknown = 0
    for i, row in df.iterrows():
        pg         = int(row["book_page_start"])
        correct_ch = page_to_chapter(pg)

        if correct_ch == -1:
            unknown += 1
            continue

        if int(row["chapter_num"]) != correct_ch:
            old = int(row["chapter_num"])
            df.at[i, "chapter_num"] = correct_ch
            changed += 1
            if changed <= 10:   # প্রথম ১০টা print করি sample হিসেবে
                print(f"  Fix: pg={pg} ch {old}→{correct_ch}")

    if changed > 10:
        print(f"  ... (showing first 10 of {changed} fixes)")

    print(f"\n  Total changed : {changed}")
    print(f"  Unknown pages : {unknown}")

    if changed == 0:
        print("  Nothing to fix.")
        return

    # drop + recreate (vectors সহ — re-embed করতে হবে না)
    db.drop_table(table_name)
    db.create_table(table_name, data=df)
    verify = db.open_table(table_name).count_rows()
    print(f"  Recreated with {verify} rows. ✓")


def main():
    print("Chapter Metadata Fix — Page Range Based")
    print("=========================================")

    db = lancedb.connect("data/lancedb")

    fix_table(db, "child_chunks")
    fix_table(db, "parent_chunks")

    # ─── Verification ───────────────────────────────────
    print("\n\n=== Verification ===")
    ct = db.open_table("child_chunks")
    df = ct.search().select(["chapter_num", "book_page_start"]).limit(5000).to_pandas()

    print("\nChapter distribution after fix:")
    print(df["chapter_num"].value_counts().sort_index().to_string())

    print("\nSample ch=2 chunks (should be pages 32-61):")
    ch2 = df[df["chapter_num"] == 2].sort_values("book_page_start")
    print(f"  page range: {ch2['book_page_start'].min()} – {ch2['book_page_start'].max()}")

    print("\nSample ch=9 chunks (should be pages 241-269):")
    ch9 = df[df["chapter_num"] == 9].sort_values("book_page_start")
    print(f"  page range: {ch9['book_page_start'].min()} – {ch9['book_page_start'].max()}")

    print("\nDone! ✓")


if __name__ == "__main__":
    main()
