"""
Check per-chapter coverage of exercise content types.
Shows which chapters are MISSING নমুনা/সংক্ষিপ্ত/সৃজনশীল chunks.
"""
import io, sys, lancedb, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

db = lancedb.connect("data/lancedb")
df = db.open_table("parent_chunks").to_pandas()

EXERCISE_TYPES = {"mcq", "mcq_nomuna", "creative_question", "shankhipto"}
STRUCTURAL_TYPES = {"niche_koro", "onusondhan", "learning_objectives", "biography",
                    "theory", "example", "formula", "figure"}

print("=== Per-chapter chunk coverage ===\n")
print(f"{'Ch':>3} | {'theory':>7} | {'example':>7} | {'niche_koro':>10} | {'onusondhan':>10} | {'MCQ/nomuna':>10} | {'creative':>9} | {'shankhipto':>10}")
print("-" * 85)

all_types = df["content_type"].unique()
print("All content_types in DB:", sorted(all_types))
print()

for ch in range(1, 14):
    ch_df = df[df["chapter_num"] == ch]
    if len(ch_df) == 0:
        print(f"{ch:>3} | NO DATA")
        continue

    types_in_ch = set(ch_df["content_type"].unique())

    theory     = len(ch_df[ch_df["content_type"] == "theory"])
    example    = len(ch_df[ch_df["content_type"] == "example"])
    niche      = len(ch_df[ch_df["content_type"] == "niche_koro"])
    onusondan  = len(ch_df[ch_df["content_type"] == "onusondhan"])
    mcq        = len(ch_df[ch_df["content_type"].isin(["mcq", "mcq_nomuna"])])
    creative   = len(ch_df[ch_df["content_type"] == "creative_question"])
    shankhipto = len(ch_df[ch_df["content_type"] == "shankhipto"])

    # Mark missing exercise content
    missing = []
    if mcq == 0:        missing.append("MCQ")
    if creative == 0:   missing.append("creative")
    if shankhipto == 0: missing.append("shankhipto")

    flag = " ← MISSING: " + ", ".join(missing) if missing else " ✓"

    print(f"{ch:>3} | {theory:>7} | {example:>7} | {niche:>10} | {onusondan:>10} | {mcq:>10} | {creative:>9} | {shankhipto:>10}{flag}")

print("\n")
print("=== Metadata-based retrieval (সব chapter) ===")
print("niche_koro by chapter:")
print(df[df["content_type"]=="niche_koro"].groupby("chapter_num").size().to_string())
print()
print("onusondhan by chapter:")
print(df[df["content_type"]=="onusondhan"].groupby("chapter_num").size().to_string())
