import io, sys, lancedb, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

db = lancedb.connect("data/lancedb")
t  = db.open_table("child_chunks")
df = t.search().select(["chunk_id","text","chapter_num","section_num","token_count"]).limit(10000).to_pandas()

# ch=1 sec=3.9 এর full text
bad = df[(df["chapter_num"]==1) & (df["section_num"]=="3.9")]
print(f"ch=1 sec=3.9 chunks: {len(bad)}")
for _, row in bad.head(2).iterrows():
    print(f"\n  tok={row['token_count']}")
    print(f"  full: {repr(str(row['text'])[:600])}")

print("\n" + "="*60)
# ch=2 ত্বরণ chunks - contains 'একক'?
good = df[(df["chapter_num"]==2) & (df["text"].str.contains("ত্বরণ", na=False))]
print(f"\nch=2 ত্বরণ chunks: {len(good)}")
for _, row in good.head(4).iterrows():
    has_ekk = "একক" in str(row["text"])
    print(f"  tok={row['token_count']}  has_একক={has_ekk}")
    print(f"  {repr(str(row['text'])[:200])}")
    print()
