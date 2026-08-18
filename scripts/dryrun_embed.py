"""Dry-run: count how many chunks will be created without embedding."""
import io, sys, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
from src.book_index import get_book_index

BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_Q_RE = re.compile(r"(?:^|\n)\s*([০-৯\d]+)[\.।]\s+")

def count_questions(text):
    matches = list(_Q_RE.finditer(text))
    return max(1, len(matches))

bidx = get_book_index()
total_parent = total_child = 0

print(f"{'Ch':>3} | {'MCQ_q':>6} | {'Sh_q':>5} | {'Cr_q':>5} | {'On':>3}")
print("-" * 40)

for ch in range(1, 14):
    ex = bidx.get_chapter(ch)
    if not ex:
        print(f"{ch:>3} | NO DATA")
        continue
    mcq_q = count_questions(ex.mcq_text)      if ex.mcq_text else 0
    sh_q  = count_questions(ex.shankhipto_text) if ex.shankhipto_text else 0
    cr_q  = count_questions(ex.creative_text)  if ex.creative_text else 0
    on    = len(ex.onusondhan_texts)

    # parents: 1 per section type + 1 per onusondhan
    p = (1 if ex.mcq_text else 0) + (1 if ex.shankhipto_text else 0) + \
        (1 if ex.creative_text else 0) + on
    # children: one per question
    c = mcq_q + sh_q + cr_q
    total_parent += p
    total_child  += c

    print(f"{ch:>3} | {mcq_q:>6} | {sh_q:>5} | {cr_q:>5} | {on:>3}  → {p}P + {c}C")

print("-" * 40)
print(f"Total: {total_parent} parent chunks, {total_child} child chunks")
print(f"Total API calls (embeds): ~{total_parent + total_child}")
print(f"Estimated time at 0.3s/embed: ~{((total_parent+total_child)*0.3/60):.1f} minutes")
