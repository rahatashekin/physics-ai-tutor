"""Quick review of chunk quality"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def show_chunk(c, idx):
    print(f"[{idx}] content_type = {c['content_type']} / {c['content_subtype']}")
    print(f"     chunk_type   = {c['chunk_type']}")
    print(f"     chapter      = {c['chapter_num']} | section = {c['section_num']}")
    print(f"     book_page    = {c['book_page_start']}-{c['book_page_end']} | spans = {c['spans_pages']}")
    print(f"     tokens       = {c['token_count']}")
    if c['onusondhan_id']:
        print(f"     onusondhan   = {c['onusondhan_id']}")
    if c['figure_id']:
        print(f"     figure_id    = {c['figure_id']} | img = {c['figure_image_path']}")
    if c['question_num'] > 0:
        print(f"     question_num = {c['question_num']} | is_nomuna = {c['is_nomuna']}")
    if c['example_num'] > 0:
        print(f"     example_num  = {c['example_num']}")
    print(f"     text preview : {repr(c['text'][:200])}")
    print()

print("=" * 70)
print("PARENT CHUNKS — sample review")
print("=" * 70)

with open("data/processed/parent_chunks.jsonl", encoding="utf-8") as f:
    parents = [json.loads(l) for l in f]

# Show variety: first 2, then one of each special type
shown_types = set()
shown = []
for c in parents:
    ct = c["content_type"]
    if len(shown) < 3 or ct not in shown_types:
        shown.append(c)
        shown_types.add(ct)
    if len(shown) >= 12:
        break

for i, c in enumerate(shown):
    show_chunk(c, i)

print("=" * 70)
print("CHILD CHUNKS — sample review")
print("=" * 70)

with open("data/processed/child_chunks.jsonl", encoding="utf-8") as f:
    children = [json.loads(l) for l in f]

for i, c in enumerate(children[:5]):
    show_chunk(c, i)

# Stats
print("=" * 70)
print("STATS")
print("=" * 70)
from collections import Counter
p_types = Counter(c["content_type"] for c in parents)
c_types = Counter(c["content_type"] for c in children)

print("Parent type distribution:")
for ct, n in p_types.most_common():
    print(f"  {ct:30s}: {n}")

print("\nChild type distribution:")
for ct, n in c_types.most_common():
    print(f"  {ct:30s}: {n}")

# Token distribution
p_tokens = [c["token_count"] for c in parents]
c_tokens = [c["token_count"] for c in children]
print(f"\nParent token stats: min={min(p_tokens)} avg={sum(p_tokens)//len(p_tokens)} max={max(p_tokens)}")
print(f"Child  token stats: min={min(c_tokens)} avg={sum(c_tokens)//len(c_tokens)} max={max(c_tokens)}")

# Empty chunks?
empty_parents = [c for c in parents if not c["text"].strip()]
empty_children = [c for c in children if not c["text"].strip()]
print(f"\nEmpty parent chunks: {len(empty_parents)}")
print(f"Empty child chunks:  {len(empty_children)}")

# onusondhan with id
on_chunks = [c for c in parents if c["onusondhan_id"]]
print(f"\nOnusondhan chunks with ID: {len(on_chunks)}")
for c in on_chunks[:5]:
    print(f"  id={c['onusondhan_id']} page={c['book_page_start']} text={repr(c['text'][:80])}")

# MCQ nomuna
nomuna = [c for c in parents if c["is_nomuna"]]
print(f"\nMCQ Nomuna chunks: {len(nomuna)}")

# figures with image path
fig_with_img = [c for c in parents if c["figure_image_path"]]
print(f"Figures with image path: {len(fig_with_img)}")
