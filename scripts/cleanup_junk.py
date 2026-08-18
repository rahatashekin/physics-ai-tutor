import io, sys, lancedb, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

db = lancedb.connect("data/lancedb")
t  = db.open_table("child_chunks")
before = t.count_rows()

# token_count < 30 olan junk chunks delete
t.delete("token_count < 30")
after = t.count_rows()
print(f"Before : {before}")
print(f"After  : {after}")
print(f"Deleted: {before - after} junk chunks")
print()

df = t.to_pandas(columns=["token_count", "content_type", "text"])
bins   = [0, 30, 50, 100, 200, 9999]
labels = ["30-50", "50-100", "100-200", "200-400", "400+"]
df["bucket"] = pd.cut(df["token_count"], bins=bins, labels=labels)
print("Token distribution after cleanup:")
print(df["bucket"].value_counts().sort_index())
print()
print("Smallest 5 remaining chunks:")
for _, row in df.nsmallest(5, "token_count").iterrows():
    tok  = row["token_count"]
    txt  = repr(str(row["text"])[:80])
    print(f"  tokens={tok}  {txt}")
