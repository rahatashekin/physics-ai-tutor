"""Root cause analysis for '8.8.4 ki bolse?' failure."""
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()

import lancedb
from src.retrieval import extract_hints

db        = lancedb.connect("data/lancedb")
parent_df = db.open_table("parent_chunks").to_pandas()

print("=" * 65)
print("1. Intent detection for the two queries")
print("=" * 65)
for q in ["8.8.4 ki bolse?", "na, subsection 8.8.4", "chapter 8 section 8.8"]:
    h = extract_hints(q)
    print(f"\nQ: '{q}'")
    print(f"   intent={h.intent.value}  ch={h.chapter}  sec={h.section}  is_subsec={h.is_subsection}  q#={h.question_num}")

print()
print("=" * 65)
print("2. Chapter 8 section data in DB")
print("=" * 65)
ch8 = parent_df[parent_df["chapter_num"] == 8][
    ["section_num","section_title","book_page_start","book_page_end","content_type"]
].drop_duplicates().sort_values("book_page_start")

for _, r in ch8.iterrows():
    print(f"  sec={str(r.section_num):10s} pg={r.book_page_start:>3}-{r.book_page_end:<3} [{r.content_type}] {str(r.section_title)[:40]}")

print()
print("=" * 65)
print("3. Pages 234-242 in DB (where 8.8.x should be)")
print("=" * 65)
near = parent_df[
    (parent_df["chapter_num"] == 8) &
    (parent_df["book_page_start"] >= 234) &
    (parent_df["book_page_start"] <= 242)
][["section_num","section_title","book_page_start","book_page_end","content_type","text"]].sort_values("book_page_start")

for _, r in near.iterrows():
    print(f"\n  sec={r.section_num} pg={r.book_page_start}-{r.book_page_end} [{r.content_type}]")
    print(f"  {str(r.section_title)[:50]}")
    print(f"  Text: {str(r.text)[:80]}")
