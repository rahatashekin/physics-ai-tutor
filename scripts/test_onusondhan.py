import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
with open("data/processed/parent_chunks.jsonl", encoding="utf-8") as f:
    parents = [json.loads(l) for l in f]
on_chunks = [c for c in parents if c["onusondhan_id"]]
print(f"Total onusondhan chunks: {len(on_chunks)}")
for c in on_chunks[:8]:
    oid = c["onusondhan_id"]
    p0 = c["book_page_start"]
    p1 = c["book_page_end"]
    sp = c["spans_pages"]
    tok = c["token_count"]
    txt = c["text"][:120]
    print(f"  id={oid}  page={p0}-{p1}  spans={sp}  tokens={tok}")
    print(f"  text={repr(txt)}")
    print()
