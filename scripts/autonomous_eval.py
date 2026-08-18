"""
scripts/autonomous_eval.py
===========================
Autonomous student testing loop for the Physics Tutor.

Design:
- Reads actual book content per chapter
- Generates questions WITH expected answer keywords (so we verify actual answer quality)
- Tests retrieval: checks if keywords appear in retrieved context
- Logs all problems with root cause hints
- Run repeatedly after fixes to track progress
"""
import io, sys, json, time, re, pathlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import lancedb
from dotenv import load_dotenv
import os

load_dotenv()

sys.path.insert(0, '.')
from src.retrieval import PhysicsTutorRetriever, build_context, extract_hints, QueryIntent
from src.figure_context import inject_figures_into_context, get_figures_for_pages

# ── Init retriever ─────────────────────────────────────────────────────────
print("Loading DB...", flush=True)
_db        = lancedb.connect("data/lancedb")
_parent_df = _db.open_table("parent_chunks").to_pandas()
_child_df  = _db.open_table("child_chunks").to_pandas()
_child_vec = np.array(_child_df["vector"].tolist(), dtype=np.float32)
retriever  = PhysicsTutorRetriever(_child_df, _child_vec, _parent_df)
print(f"DB ready: {len(_parent_df)} parent, {len(_child_df)} child chunks\n")

from vertexai.language_models import TextEmbeddingModel
import vertexai
vertexai.init(project=os.getenv("GCP_PROJECT"), location="us-central1")
_em = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

def embed(text: str) -> list:
    return _em.get_embeddings([text])[0].values


# ── Test question bank ─────────────────────────────────────────────────────
# Each question has:
#   q: the student question (natural, colloquial)
#   ch: expected chapter
#   type: question category
#   must_contain: keywords that MUST appear in retrieved context (answer validation)
#   must_not_contain: content that should NOT appear (cross-chapter pollution check)

QUESTIONS = [
    # ── Ch=1: ভৌত রাশি ও পরিমাপ ──────────────────────────────────────────
    {"q": "স্ক্রু গেজের লঘিষ্ঠ গণন কত?", "ch": 1,
     "must_contain": ["স্ক্রু গেজ", "লঘিষ্ঠ"],
     "must_not_contain": ["তড়িৎ", "সার্কিট"]},

    {"q": "ভার্নিয়ার ক্যালিপার্স দিয়ে কীভাবে পরিমাপ করে?", "ch": 1,
     "must_contain": ["ভার্নিয়ার"],
     "must_not_contain": ["তরঙ্গ", "আলো"]},

    {"q": "মৌলিক রাশি কাকে বলে? কয়টি আছে?", "ch": 1,
     "must_contain": ["মৌলিক রাশি"],
     "must_not_contain": []},

    {"q": "SI একক কী?", "ch": 1,
     "must_contain": ["একক", "SI"],
     "must_not_contain": []},

    # ── Ch=2: গতি ──────────────────────────────────────────────────────────
    {"q": "সমত্বরণে চলমান বস্তুর তিনটি সমীকরণ কী?", "ch": 2,
     "must_contain": ["সমত্বরণ", "সমীকরণ"],
     "must_not_contain": ["আয়না", "লেন্স"]},

    {"q": "বেগ ও দ্রুতির পার্থক্য কী?", "ch": 2,
     "must_contain": ["বেগ", "দ্রুতি"],
     "must_not_contain": []},

    {"q": "মুক্তপতন কাকে বলে? g এর মান কত?", "ch": 2,
     "must_contain": ["মুক্তপতন"],
     "must_not_contain": []},

    # ── Ch=3: বল ──────────────────────────────────────────────────────────
    {"q": "নিউটনের গতির সূত্র তিনটি লিখো", "ch": 3,
     "must_contain": ["নিউটন", "সূত্র"],
     "must_not_contain": ["আলো", "তড়িৎ"]},

    {"q": "ভরবেগের সংরক্ষণ নীতি কী?", "ch": 3,
     "must_contain": ["ভরবেগ"],
     "must_not_contain": []},

    {"q": "ঘর্ষণ বল কাকে বলে? স্থিতি ও গতি ঘর্ষণের পার্থক্য?", "ch": 3,
     "must_contain": ["ঘর্ষণ"],
     "must_not_contain": []},

    # ── Ch=4: কাজ, ক্ষমতা ও শক্তি ────────────────────────────────────────
    {"q": "কাজ কাকে বলে? W = Fs cos θ সূত্রটা বুঝিয়ে দাও", "ch": 4,
     "must_contain": ["কাজ"],
    "must_not_contain": []},

    {"q": "শক্তির নিত্যতার সূত্র কী?", "ch": 4,
     "must_contain": ["শক্তি"],
     "must_not_contain": ["তড়িৎ"]},

    {"q": "গতিশক্তি ও স্থিতিশক্তির সূত্র লিখো", "ch": 4,
     "must_contain": ["গতিশক্তি", "স্থিতিশক্তি"],
     "must_not_contain": []},

    # ── Ch=5: পদার্থের অবস্থা ও চাপ ──────────────────────────────────────
    {"q": "আর্কিমিডিসের সূত্র কী? প্লবতা কীভাবে কাজ করে?", "ch": 5,
     "must_contain": ["আর্কিমিডিস", "প্লব"],
     "must_not_contain": []},

    {"q": "প্যাসকেলের সূত্র কী?", "ch": 5,
     "must_contain": ["প্যাসকেল"],
     "must_not_contain": []},

    {"q": "বায়ুমণ্ডলীয় চাপ কত?", "ch": 5,
     "must_contain": ["বায়ুমণ্ডলীয় চাপ"],
     "must_not_contain": ["তড়িৎ", "আলো"]},

    # ── Ch=6: বস্তুর উপর তাপের প্রভাব ────────────────────────────────────
    {"q": "তাপ ও তাপমাত্রার পার্থক্য কী?", "ch": 6,
     "must_contain": ["তাপমাত্র"],
     "must_not_contain": ["ওহম", "তড়িৎ"]},

    {"q": "রৈখিক প্রসারণ সহগ কাকে বলে? সূত্র লিখো।", "ch": 6,
     "must_contain": ["রৈখিক প্রসারণ"],
     "must_not_contain": []},

    {"q": "গলনাংক ও স্ফুটনাংক কী?", "ch": 6,
     "must_contain": ["গলনাংক"],
     "must_not_contain": []},

    # ── Ch=7: তরঙ্গ ও শব্দ ────────────────────────────────────────────────
    {"q": "অনুদৈর্ঘ্য ও অনুপ্রস্থ তরঙ্গ কাকে বলে? পার্থক্য কী?", "ch": 7,
     "must_contain": ["অনুদৈর্ঘ্য", "অনুপ্রস্থ"],
     "must_not_contain": ["আয়না", "লেন্স"]},

    {"q": "শব্দের বেগ কত? কোন মাধ্যমে বেশি?", "ch": 7,
     "must_contain": ["শব্দ"],
     "must_not_contain": ["আলো", "তড়িৎ"]},

    {"q": "কম্পাঙ্ক ও তরঙ্গদৈর্ঘ্যের সম্পর্ক কী?", "ch": 7,
     "must_contain": ["কম্পাঙ্ক", "তরঙ্গদৈর্ঘ্য"],
     "must_not_contain": []},

    # ── Ch=8: আলোর প্রতিফলন ──────────────────────────────────────────────
    {"q": "উত্তল ও অবতল আয়নার পার্থক্য কী?", "ch": 8,
     "must_contain": ["আয়না", "উত্তল", "অবতল"],
     "must_not_contain": ["কুলম্ব", "চার্জ"]},

    {"q": "আয়নার সূত্র কী? 1/f = 1/v + 1/u", "ch": 8,
     "must_contain": ["আয়না"],
     "must_not_contain": ["লেন্স মেকার"]},

    {"q": "বক্রতার কেন্দ্র ও ফোকাস দূরত্বের সম্পর্ক?", "ch": 8,
     "must_contain": ["বক্রতা", "ফোকাস"],
     "must_not_contain": ["চার্জ", "তড়িৎ"]},

    {"q": "৮.৮.৪ ধারায় কী আছে?", "ch": 8,
     "must_contain": ["পাহাড়ি", "বাঁক", "আয়না"],
     "must_not_contain": []},

    # ── Ch=9: আলোর প্রতিসরণ ──────────────────────────────────────────────
    {"q": "স্নেলের সূত্র কী?", "ch": 9,
     "must_contain": ["স্নেল", "প্রতিসরণ"],
     "must_not_contain": ["কুলম্ব", "চার্জ"]},

    {"q": "পূর্ণ অভ্যন্তরীণ প্রতিফলন কখন ঘটে?", "ch": 9,
     "must_contain": ["পূর্ণ অভ্যন্তরীণ", "প্রতিফলন"],
     "must_not_contain": []},

    {"q": "উত্তল লেন্সে বিম্ব কোথায় তৈরি হয়?", "ch": 9,
     "must_contain": ["লেন্স", "বিম্ব"],
     "must_not_contain": ["আয়না", "দর্পণ"]},

    # ── Ch=10: স্থির তড়িৎ ────────────────────────────────────────────────
    {"q": "কুলম্বের সূত্র কী?", "ch": 10,
     "must_contain": ["কুলম্ব"],
     "must_not_contain": ["আলো", "তরঙ্গ"]},

    {"q": "বৈদ্যুতিক ক্ষেত্র প্রাবল্য কাকে বলে?", "ch": 10,
     "must_contain": ["বৈদ্যুতিক ক্ষেত্র"],
     "must_not_contain": []},

    # ── Ch=11: চল তড়িৎ ───────────────────────────────────────────────────
    {"q": "ওহমের সূত্র কী? সূত্রটা লিখো।", "ch": 11,
     "must_contain": ["ওহম"],
     "must_not_contain": ["আলো", "তরঙ্গ"]},

    {"q": "কির্শফের সূত্র দুটো কী?", "ch": 11,
     "must_contain": ["কির্শফ"],
     "must_not_contain": []},

    {"q": "রোধের শ্রেণি ও সমান্তরাল সংযোগ কীভাবে হয়?", "ch": 11,
     "must_contain": ["রোধ", "সংযোগ"],
     "must_not_contain": ["আলো"]},

    # ── Ch=12: বিদ্যুতের চুম্বকীয় ক্রিয়া ───────────────────────────────
    {"q": "ফ্যারাডের তড়িৎচুম্বকীয় আবেশ সূত্র কী?", "ch": 12,
     "must_contain": ["ফ্যারাড", "আবেশ"],
     "must_not_contain": []},

    {"q": "বৈদ্যুতিক মোটর কীভাবে কাজ করে?", "ch": 12,
     "must_contain": ["মোটর"],
     "must_not_contain": []},

    # ── Ch=13: আধুনিক পদার্থবিজ্ঞান ──────────────────────────────────────
    {"q": "ফটো ইলেকট্রিক ক্রিয়া কী? আইনস্টাইনের সমীকরণ লিখো।", "ch": 13,
     "must_contain": ["ফটো ইলেকট্রিক", "আইনস্টাইন"],
     "must_not_contain": []},

    {"q": "তেজস্ক্রিয়তা কী? আলফা বিটা গামা রশ্মির পার্থক্য?", "ch": 13,
     "must_contain": ["তেজস্ক্রিয়"],
     "must_not_contain": []},

    # ── Figure/Diagram queries (user wants image) ────────────────────────
    {"q": "আলোর পূর্ণ অভ্যন্তরীণ প্রতিফলনের চিত্র দেখাও।", "ch": 9,
     "must_contain": ["প্রতিফলন"],
     "must_not_contain": ["আয়না", "দর্পণ"],
     "fig": True},

    {"q": "উত্তল লেন্সের কিরণচিত্র কেমন দেখতে?", "ch": 9,
     "must_contain": ["লেন্স"],
     "must_not_contain": ["আয়না"],
     "fig": True},

    {"q": "ফ্যারাডের আবেশ সংক্রান্ত চিত্র দেখাও।", "ch": 12,
     "must_contain": ["আবেশ"],
     "must_not_contain": ["আলো"],
     "fig": True},
]


# ── Evaluation ────────────────────────────────────────────────────────────
def evaluate(q_data: dict) -> dict:
    q          = q_data["q"]
    exp_ch     = q_data["ch"]
    must_have  = q_data.get("must_contain", [])
    must_not   = q_data.get("must_not_contain", [])

    try:
        q_vec = embed(q)
        results, hints = retriever.retrieve(q, q_vec, top_k=6)
        ctx = build_context(results, hints)

        intent     = hints.intent.name
        chapters   = sorted(set(r.get("chapter", r.get("chapter_num", 0))
                                for r in results if r.get("chapter") or r.get("chapter_num")))
        sec_labels = [r.get("section", "") for r in results[:3]]

        # Answer keyword check
        missing    = [kw for kw in must_have    if kw not in ctx]
        pollutants = [kw for kw in must_not     if kw in ctx]
        ch_correct = exp_ch in chapters if exp_ch > 0 else True
        answered   = len(missing) == 0
        clean      = len(pollutants) == 0

        # Figure injection check (for fig: True queries)
        fig_ok  = True   # default pass for non-figure queries
        fig_note = ""
        if q_data.get("fig"):
            ctx_with_figs = inject_figures_into_context(ctx, results, max_total_figs=4)
            has_figs = "সংশ্লিষ্ট চিত্রের বর্ণনা" in ctx_with_figs
            if not has_figs:
                # Fall back: look for figures in found chapters
                for r in results:
                    ps = r.get("page_start", 0)
                    pe = r.get("page_end", ps)
                    figs = get_figures_for_pages(ps, pe, max_figs=2)
                    if figs:
                        has_figs = True
                        break
            fig_ok   = has_figs
            fig_note = f" [figs={'✓' if has_figs else '✗'}]"

        if ch_correct and answered and clean and fig_ok:
            status = "PASS"
        elif ch_correct and answered:
            status = "WARN"   # right chapter + answer but polluted or no figs
        elif ch_correct:
            status = "PARTIAL"  # right chapter but answer keywords missing
        else:
            status = "FAIL"

        return {
            "q": q, "ch": exp_ch, "status": status,
            "intent": intent, "chapters": chapters,
            "sections": sec_labels,
            "missing_kw": missing, "pollutants": pollutants,
            "ctx_len": len(ctx),
            "ctx_preview": ctx[:300],
            "fig_note": fig_note,
        }
    except Exception as e:
        return {"q": q, "ch": exp_ch, "status": "ERROR", "error": str(e)}


# ── Run ───────────────────────────────────────────────────────────────────
print("=" * 65)
print("AUTONOMOUS STUDENT EVALUATION — Physics Tutor")
print("=" * 65)
print()

results   = []
by_status = {"PASS": [], "WARN": [], "PARTIAL": [], "FAIL": [], "ERROR": []}

for i, qd in enumerate(QUESTIONS):
    r = evaluate(qd)
    results.append(r)
    by_status[r["status"]].append(r)

    icon = {"PASS": "✅", "WARN": "⚠️", "PARTIAL": "🔶", "FAIL": "❌", "ERROR": "💥"}[r["status"]]
    print(f"[{i+1:02d}] Ch={qd['ch']} {icon} [{r['intent']:18s}] "
          f"ch_found={r.get('chapters',[])} | {qd['q'][:45]}")
    if r["status"] not in ("PASS", "WARN"):
        if r.get("missing_kw"):
            print(f"     → Missing keywords: {r['missing_kw']}")
        if r.get("pollutants"):
            print(f"     → Polluted by: {r['pollutants']}")
        if r.get("error"):
            print(f"     → ERROR: {r['error'][:80]}")
    time.sleep(0.05)

print()
print("=" * 65)
print(f"SUMMARY: {len(QUESTIONS)} questions tested")
for s, items in by_status.items():
    pct = 100 * len(items) // len(QUESTIONS)
    bar = "█" * (pct // 5)
    print(f"  {s:8s}: {len(items):2d} ({pct:2d}%) {bar}")
print()

# ── Detailed failure analysis ─────────────────────────────────────────────
failures = by_status["FAIL"] + by_status["PARTIAL"] + by_status["ERROR"]
if failures:
    print("FAILURES / PARTIALS — Root Cause Analysis:")
    print("-" * 65)
    for r in failures:
        print(f"\n  [{r['status']}] Ch={r['ch']}: {r['q']}")
        print(f"  Intent={r.get('intent','?')} | Chapters found={r.get('chapters',[])}")
        print(f"  Missing: {r.get('missing_kw',[])} | Polluted: {r.get('pollutants',[])}")
        print(f"  Preview: {r.get('ctx_preview','')[:120]}")
else:
    print("All questions PASSED! ✅")

# ── Save results ──────────────────────────────────────────────────────────
out = pathlib.Path("scripts/autonomous_eval_results.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nFull results → {out}")
