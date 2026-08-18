import io, sys, lancedb, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

db = lancedb.connect("data/lancedb")
t  = db.open_table("child_chunks")
df = t.search().select(["chunk_id", "text", "chapter_num", "section_num", "token_count"]).limit(10000).to_pandas()

# ত্বরণ সম্পর্কিত content
mask  = df["text"].str.contains("ত্বরণ", na=False)
found = df[mask]
print(f"child_chunks এ 'ত্বরণ' আছে: {len(found)} chunks")
print()
for _, row in found.head(6).iterrows():
    ch  = row["chapter_num"]
    sec = row["section_num"]
    tok = row["token_count"]
    txt = repr(str(row["text"])[:120])
    print(f"  ch={ch} sec={sec} tok={tok}")
    print(f"  {txt}")
    print()

# তড়িৎ আধান সম্পর্কিত
print("=" * 60)
mask2  = df["text"].str.contains("তড়িৎ আধান", na=False)
found2 = df[mask2]
print(f"child_chunks এ 'তড়িৎ আধান' আছে: {len(found2)} chunks")
for _, row in found2.head(4).iterrows():
    ch  = row["chapter_num"]
    sec = row["section_num"]
    tok = row["token_count"]
    txt = repr(str(row["text"])[:120])
    print(f"  ch={ch} sec={sec} tok={tok}  {txt}")
