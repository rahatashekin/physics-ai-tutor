"""Quick coverage test for BookIndex."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
from src.book_index import get_book_index

idx = get_book_index()
print(idx.coverage_report())
print()

print("=== Ch3 MCQ (first 300 chars) ===")
print(idx.get_mcq(3)[:300])
print()

print("=== Ch5 Shankhipto (first 300 chars) ===")
print(idx.get_shankhipto(5)[:300])
print()

print("=== Ch8 Creative (first 300 chars) ===")
print(idx.get_creative(8)[:300])
print()

print("=== Ch2 Onusondhan ===")
for o in idx.get_onusondhan(2):
    print(f"  id={o['id']} book_pg={o['book_page']} | {o['text'][:80]}")

print()
print("=== Ch10 Onusondhan ===")
for o in idx.get_onusondhan(10):
    print(f"  id={o['id']} book_pg={o['book_page']} | {o['text'][:80]}")
