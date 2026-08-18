import io, sys, lancedb, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

db = lancedb.connect("data/lancedb")
pt = db.open_table("parent_chunks")
df = pt.to_pandas()

# subtype=question chunks
print("=== content_subtype='question' chunks ===")
qs = df[df["content_subtype"] == "question"].sort_values(["chapter_num","book_page_start"])
print(f"Total: {len(qs)}")
for _, row in qs.iterrows():
    print(f"  ch={row['chapter_num']} pg={row['book_page_start']}-{row['book_page_end']} "
          f"type={row['content_type']} sec={row['section_num']} "
          f"text={str(row['text'])[:80]}")

# chapter 2 specific: search for any "প্রশ্ন" pattern
print("\n\n=== Chapter 2: chunks with 'প্রশ্ন' in text ===")
ch2 = df[df["chapter_num"] == 2]
q_ch2 = ch2[ch2["text"].str.contains("প্রশ্ন", na=False)]
for _, row in q_ch2.iterrows():
    print(f"  pg={row['book_page_start']}-{row['book_page_end']} type={row['content_type']} "
          f"subtype={row['content_subtype']} text={str(row['text'])[:100]}")

# Also check extracted_book.md for nomuna proshno in chapter 2
print("\n\n=== Searching extracted_book.md for 'নমুনা প্রশ্ন' near chapter 2 ===")
with open("data/processed/extracted_book.md", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Find lines with নমুনা প্রশ্ন
for i, line in enumerate(lines):
    if "নমুনা প্রশ্ন" in line or "সংক্ষিপ্ত প্রশ্ন" in line:
        # print context
        context_start = max(0, i-1)
        context_end   = min(len(lines), i+3)
        for j in range(context_start, context_end):
            print(f"  L{j}: {lines[j].strip()[:100]}")
        print()
