import io, sys, lancedb
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
db = lancedb.connect("data/lancedb")
pt = db.open_table("parent_chunks")
df = pt.to_pandas()

# Direct text search for chapter 2 MCQ content
keywords = ["ত্বরণের একক", "ঘড়ির কাঁটার গতি", "বহুনির্বাচনি"]
for kw in keywords:
    hits = df[df["text"].str.contains(kw, na=False)]
    if len(hits) > 0:
        print(f"'{kw}' found in {len(hits)} chunks:")
        for _, row in hits.iterrows():
            print(f"  ch={row['chapter_num']} pg={row['book_page_start']}-{row['book_page_end']} type={row['content_type']} text_start={str(row['text'])[:80]}")
    else:
        print(f"'{kw}' NOT FOUND in any chunk")
    print()

# Also: show ALL chunks at book_page 59-70 (might cross chapter boundary)
print("=== All chunks at book pages 59-70 ===")
cross = df[(df["book_page_start"]>=59) & (df["book_page_start"]<=70)].sort_values("book_page_start")
print(f"Total: {len(cross)}")
for _, row in cross.iterrows():
    print(f"  ch={row['chapter_num']} pg={row['book_page_start']}-{row['book_page_end']} type={row['content_type']} text={str(row['text'])[:70]}")
