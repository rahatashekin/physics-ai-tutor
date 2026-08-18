import io, sys, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import lancedb

sys.path.insert(0, '.')
from src.retrieval import PhysicsTutorRetriever, build_context, extract_hints

# Initialize retriever exactly as app.py does
print("Loading database...", flush=True)
_db         = lancedb.connect("data/lancedb")
_parent_df  = _db.open_table("parent_chunks").to_pandas()
_child_df   = _db.open_table("child_chunks").to_pandas()
_child_vec  = np.array(_child_df["vector"].tolist(), dtype=np.float32)
retriever   = PhysicsTutorRetriever(_child_df, _child_vec, _parent_df)
print(f"DB loaded: {len(_parent_df)} parent chunks, {len(_child_df)} child chunks\n")

# We also need embeddings to do vector search — use the same embed function
from vertexai.language_models import TextEmbeddingModel
import vertexai
from dotenv import load_dotenv
import os
load_dotenv()
vertexai.init(project=os.getenv("GCP_PROJECT"), location="us-central1")
_embed_model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

def embed(text: str) -> list:
    return _embed_model.get_embeddings([text])[0].values


# ────────────────────────────────────────────────────────────────────────────
# Student Questions — diverse types across chapters
# Types: direct concept, specific section, formula-based, MCQ, numerical,
#         sub-section, cross-chapter, vague/natural language
# ────────────────────────────────────────────────────────────────────────────
QUESTIONS = [
    # CH=1 ভৌত রাশি ও পরিমাপ
    {"q": "স্ক্রু গেজের লঘিষ্ঠ গণন কত?",                           "ch": 1, "type": "specific_formula"},
    {"q": "১.৩ ধারায় কী আছে?",                                      "ch": 1, "type": "section_lookup"},

    # CH=2 গতি
    {"q": "সমত্বরণে গতির সমীকরণগুলো কী কী?",                       "ch": 2, "type": "formula"},
    {"q": "v² = u² + 2as এটা কোথা থেকে এলো?",                       "ch": 2, "type": "derivation"},
    {"q": "৯.৮ m/s² মানে কী?",                                      "ch": 2, "type": "concept"},

    # CH=3 বল
    {"q": "নিউটনের তৃতীয় সূত্রটা বলো",                              "ch": 3, "type": "concept"},
    {"q": "ঘর্ষণ বল কখন স্থিতিস্থাপক সীমা ছাড়ায়?",                "ch": 3, "type": "conceptual"},

    # CH=4 কাজ, ক্ষমতা ও শক্তি
    {"q": "শক্তির নিত্যতার সূত্র কী?",                               "ch": 4, "type": "formula"},
    {"q": "অনুসন্ধান ৪.০১ এ কী আছে?",                               "ch": 4, "type": "onusondhan"},

    # CH=5 পদার্থের অবস্থা ও চাপ
    {"q": "আর্কিমিডিসের সূত্র কী?",                                  "ch": 5, "type": "formula"},
    {"q": "প্লবতা কাকে বলে?",                                        "ch": 5, "type": "definition"},
    {"q": "১৩.৬ g/cc কী জিনিস?",                                    "ch": 5, "type": "tricky_formula_mislabel"},

    # CH=6 বস্তুর উপর তাপের প্রভাব
    {"q": "তাপমাত্রা ও তাপের পার্থক্য কী?",                          "ch": 6, "type": "concept"},
    {"q": "রৈখিক প্রসারণ সহগ কাকে বলে?",                            "ch": 6, "type": "definition"},

    # CH=7 তরঙ্গ ও শব্দ
    {"q": "শব্দের বেগ কত?",                                          "ch": 7, "type": "value"},
    {"q": "অনুদৈর্ঘ্য ও অনুপ্রস্থ তরঙ্গের পার্থক্য?",               "ch": 7, "type": "comparison"},

    # CH=8 আলোর প্রতিফলন — most tested
    {"q": "৮.৮.৪ কী বলে?",                                          "ch": 8, "type": "subsection"},
    {"q": "উত্তল ও অবতল আয়নার পার্থক্য কী?",                        "ch": 8, "type": "comparison"},
    {"q": "আয়নার সূত্র লিখো",                                        "ch": 8, "type": "formula"},
    {"q": "৮.৫ ধারায় কী আছে?",                                      "ch": 8, "type": "section_lookup"},
    {"q": "বক্রতার কেন্দ্র মানে কী?",                                "ch": 8, "type": "definition"},

    # CH=9 আলোর প্রতিসরণ
    {"q": "স্নেলের সূত্র কী?",                                       "ch": 9, "type": "formula"},
    {"q": "পূর্ণ অভ্যন্তরীণ প্রতিফলন কখন হয়?",                      "ch": 9, "type": "condition"},
    {"q": "লেন্সমেকারের সমীকরণ কী?",                                 "ch": 9, "type": "formula"},

    # CH=10 স্থির তড়িৎ
    {"q": "কুলম্বের সূত্র কী?",                                       "ch": 10, "type": "formula"},
    {"q": "বৈদ্যুতিক ক্ষেত্র ও বৈদ্যুতিক বিভব কী?",                 "ch": 10, "type": "concept"},

    # CH=11 চল তড়িৎ
    {"q": "ওহমের সূত্র কী?",                                         "ch": 11, "type": "formula"},
    {"q": "কির্শফের সূত্র দুটো বলো",                                  "ch": 11, "type": "two_part"},

    # CH=12 বিদ্যুতের চুম্বকীয় ক্রিয়া
    {"q": "ফ্যারাডের সূত্র কী?",                                      "ch": 12, "type": "formula"},

    # CH=13 আধুনিক পদার্থবিজ্ঞান
    {"q": "ফটো ইলেকট্রিক ক্রিয়া কী?",                               "ch": 13, "type": "concept"},
    {"q": "আইনস্টাইনের আলোকতড়িৎ সমীকরণ কী?",                       "ch": 13, "type": "formula"},

    # VAGUE / NATURAL LANGUAGE — tests intent detection
    {"q": "আলো কীভাবে কাজ করে?",                                    "ch": 8, "type": "vague"},
    {"q": "পদার্থ মাপা কীভাবে হয়?",                                  "ch": 1, "type": "vague"},

    # CROSS-CHAPTER
    {"q": "তরঙ্গ আর আলো কি একই জিনিস?",                             "ch": 0, "type": "cross_chapter"},
]


def evaluate_context(question_data: dict) -> dict:
    """Retrieve context for a question and evaluate quality."""
    q = question_data["q"]
    expected_ch = question_data["ch"]

    try:
        q_vec = embed(q)
        results, hints = retriever.retrieve(q, q_vec, top_k=5)
        ctx = build_context(results, hints)

        chunks_returned = len(results)
        chapters_found = sorted(set(r.get("chapter_num", 0) for r in results if r.get("chapter_num")))
        correct_chapter = expected_ch == 0 or expected_ch in chapters_found
        has_relevant_content = len(ctx) > 100

        return {
            "question": q,
            "type": question_data["type"],
            "expected_ch": expected_ch,
            "intent_detected": hints.intent.name if hints else "?",
            "chapters_found": chapters_found,
            "correct_chapter": correct_chapter,
            "chunks": chunks_returned,
            "context_len": len(ctx),
            "context_preview": ctx[:300],
            "status": "OK" if correct_chapter and has_relevant_content else "PROBLEM",
        }
    except Exception as e:
        return {
            "question": q,
            "type": question_data["type"],
            "expected_ch": expected_ch,
            "error": str(e),
            "status": "ERROR",
        }


# ── Run tests ────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHYSICS TUTOR STUDENT EVALUATION TEST")
print("=" * 70)
print()

results = []
problems = []

for i, qdata in enumerate(QUESTIONS):
    print(f"[{i+1:02d}/{len(QUESTIONS)}] {qdata['type']:25s} | Ch={qdata['ch']} | {qdata['q'][:50]}")
    r = evaluate_context(qdata)
    results.append(r)

    status_icon = "✓" if r["status"] == "OK" else ("✗" if r["status"] == "PROBLEM" else "!")
    found = r.get("chapters_found", [])
    print(f"        {status_icon} chapters_found={found} chunks={r.get('chunks',0)} ctx_len={r.get('context_len',0)}")

    if r["status"] != "OK":
        problems.append(r)
        err = r.get("error", "")
        if err:
            print(f"        ERROR: {err[:80]}")

    time.sleep(0.1)

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"RESULTS: {len(QUESTIONS)} questions | {len(problems)} problems found")
print("=" * 70)
print()
print("PROBLEMS:")
for p in problems:
    print(f"\n  [{p['type']}] Ch={p['expected_ch']}: {p['question']}")
    print(f"  → chapters_found={p.get('chapters_found', [])} status={p['status']}")
    print(f"  → context_preview: {p.get('context_preview', p.get('error', ''))[:120]}")

# Save full results to JSON
with open('scripts/tutor_eval_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nFull results saved to scripts/tutor_eval_results.json")
