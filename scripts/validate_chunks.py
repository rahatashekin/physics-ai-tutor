import io, sys, lancedb, re, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

db = lancedb.connect("data/lancedb")
t  = db.open_table("child_chunks")
df = t.search().select(["chunk_id","chapter_num","book_page_start","section_num"]).limit(5000).to_pandas()

mismatches = 0
for _, row in df.iterrows():
    cid = str(row["chunk_id"])
    m   = re.match(r"ch(\d+)_", cid)
    if not m:
        print(f"NO MATCH: {cid}")
        continue
    inferred = int(m.group(1))
    stored   = row["chapter_num"]
    pg       = row["book_page_start"]
    sec      = row["section_num"]
    if inferred != stored:
        mismatches += 1
        print(f"MISMATCH: id_ch={inferred} stored={stored} pg={pg} sec={sec}  {cid[:40]}")

print(f"\nTotal mismatches: {mismatches} out of {len(df)}")
