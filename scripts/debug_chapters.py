import io, sys, lancedb, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

db = lancedb.connect("data/lancedb")
t  = db.open_table("child_chunks")
df = t.search().select(["chunk_id","chapter_num","section_num","book_page_start","content_type"]).limit(5000).to_pandas()

print("Chapter distribution in child_chunks:")
print(df["chapter_num"].value_counts().sort_index())
print()

# section starts with "2." → should be chapter 2
mask = df["section_num"].astype(str).str.match(r"^2\.\d")
sec2 = df[mask]
print(f"section_num starts with '2.': {len(sec2)} chunks")
print("Their chapter_num distribution:")
print(sec2["chapter_num"].value_counts())
print()
print("Sample sec=2.x rows (page range):")
sample = sec2.sort_values("book_page_start").drop_duplicates("section_num").head(10)
for _, row in sample.iterrows():
    ch  = row["chapter_num"]
    sec = row["section_num"]
    pg  = row["book_page_start"]
    ct  = row["content_type"]
    print(f"  ch={ch} sec={sec} pg={pg} type={ct}")
