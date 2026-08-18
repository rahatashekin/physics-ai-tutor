import io, sys, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
import numpy as np
import lancedb
from dotenv import load_dotenv
load_dotenv()

from src.retrieval import PhysicsTutorRetriever, build_context, extract_hints
from src.figure_context import inject_figures_into_context, get_figures_for_pages

# ── Init ──────────────────────────────────────────────────────────────────
print("Loading retriever...", flush=True)
db = lancedb.connect('data/lancedb')
cdf = db.open_table('child_chunks').to_pandas()
pdf = db.open_table('parent_chunks').to_pandas()
cvec = np.array(cdf['vector'].tolist(), dtype=np.float32)
retriever = PhysicsTutorRetriever(cdf, cvec, pdf)

# Embed function
from vertexai.language_models import TextEmbeddingModel
import vertexai
vertexai.init(project=os.getenv("GCP_PROJECT"), location="us-central1")
_em = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

def embed(text: str):
    return np.array(_em.get_embeddings([text])[0].values, dtype=np.float32)

# ── 20 Critical Questions ─────────────────────────────────────────────────
QUESTIONS = [
    {"id": 1,  "ch": 1,  "q": "স্ক্রু গেইজের ন্যূনাংক কত? পিচ 1 mm এবং বৃত্তাকার স্কেল 100 ভাগে ভাগ হলে?",
     "must": ["স্ক্রু", "0.01"], "must_not": [], "fig": False},
    {"id": 2,  "ch": 1,  "q": "স্লাইড ক্যালিপার্সে প্রধান স্কেল পাঠ 4 cm, ভার্নিয়ার সমপাতন 7 এবং ভার্নিয়ার ধ্রুবক 0.1 mm হলে দণ্ডের দৈর্ঘ্য কত?",
     "must": ["4.07"], "must_not": [], "fig": False},
    {"id": 3,  "ch": 2,  "q": "মুক্তভাবে পড়ন্ত বস্তু কাকে বলে? g-এর মান কত এবং এর মাত্রা কী?",
     "must": ["9.8", "মুক্তপতন"], "must_not": [], "fig": False},
    {"id": 4,  "ch": 3,  "q": "নিউটনের তৃতীয় সূত্রটি লেখো। 60 kg ভরের ব্যক্তি 10 m/s² ত্বরণে লিফটে উঠলে আপাত ওজন কত?",
     "must": ["ক্রিয়া", "বিক্রিয়া"], "must_not": [], "fig": False},
    {"id": 5,  "ch": 4,  "q": "কাজ ও ক্ষমতার সম্পর্ক কী? কোনো মেশিন 500 J কাজ করতে 10 s নিলে ক্ষমতা কত?",
     "must": ["ক্ষমতা", "কাজ"], "must_not": [], "fig": False},
    {"id": 6,  "ch": 4,  "q": "ভর ও শক্তির সম্পর্ক সূত্রটি কে দিয়েছেন? E=mc² সূত্রে 1 kg ভরের সমতুল্য শক্তি কত?",
     "must": ["আইনস্টাইন", "mc"], "must_not": [], "fig": False},
    {"id": 7,  "ch": 5,  "q": "বায়ুমণ্ডলীয় চাপ কত প্যাস্কেল? পানির 10 m গভীরে মোট চাপ কত হবে?",
     "must": ["চাপ", "বায়ুমণ্ডলীয়"], "must_not": [], "fig": False},
    {"id": 8,  "ch": 5,  "q": "আর্কিমিডিসের সূত্রটি লেখো এবং কেন লোহার জাহাজ ভাসে ব্যাখ্যা করো। সংশ্লিষ্ট চিত্রটি দেখাও।",
     "must": ["আর্কিমিডিস", "প্লবতা"], "must_not": [], "fig": True},
    {"id": 9,  "ch": 6,  "q": "রৈখিক প্রসারণ সহগ কাকে বলে? লোহার দণ্ড 20°C তে 1 m লম্বা, 100°C তে দৈর্ঘ্য কত? (α=1.2×10⁻⁵)",
     "must": ["প্রসারণ", "সহগ"], "must_not": [], "fig": False},
    {"id": 10, "ch": 7,  "q": "তরঙ্গদৈর্ঘ্য, কম্পাঙ্ক এবং বেগের মধ্যে সম্পর্কটি লেখো। শব্দের বেগ 340 m/s, কম্পাঙ্ক 440 Hz হলে তরঙ্গদৈর্ঘ্য কত?",
     "must": ["তরঙ্গদৈর্ঘ্য", "কম্পাঙ্ক"], "must_not": [], "fig": False},
    {"id": 11, "ch": 7,  "q": "অনুদৈর্ঘ্য এবং অনুপ্রস্থ তরঙ্গের মধ্যে পার্থক্য কী? শব্দ কোন ধরনের তরঙ্গ? চিত্রসহ দেখাও।",
     "must": ["অনুদৈর্ঘ্য", "শব্দ"], "must_not": [], "fig": True},
    {"id": 12, "ch": 8,  "q": "প্রতিফলনের দুটি সূত্র লেখো। উত্তল আয়নায় কোন ধরনের প্রতিবিম্ব তৈরি হয়? বইয়ের চিত্র দেখাও।",
     "must": ["প্রতিফলন", "উত্তল"], "must_not": [], "fig": True},
    {"id": 13, "ch": 9,  "q": "স্নেলের সূত্র লেখো। কাচের প্রতিসরণাঙ্ক 1.52 হলে কাচে আলোর বেগ কত? (c=3×10⁸ m/s)",
     "must": ["স্নেল", "প্রতিসরণ"], "must_not": [], "fig": False},
    {"id": 14, "ch": 9,  "q": "পূর্ণ অভ্যন্তরীণ প্রতিফলন কখন ঘটে? চিত্রটি দেখাও এবং দুটি ব্যবহারিক প্রয়োগ লেখো।",
     "must": ["পূর্ণ অভ্যন্তরীণ", "ক্রান্তি"], "must_not": [], "fig": True},
    {"id": 15, "ch": 10, "q": "কুলম্বের সূত্র লেখো। +1 C এবং -1 C চার্জ 10 cm দূরে থাকলে বল কত? (k=9×10⁹)",
     "must": ["কুলম্ব", "বল"], "must_not": [], "fig": False},
    {"id": 16, "ch": 11, "q": "ওহমের সূত্র লেখো। 220 V লাইনে 100 W বাল্বের ফিলামেন্টের রোধ কত?",
     "must": ["ওহম", "রোধ"], "must_not": [], "fig": False},
    {"id": 17, "ch": 11, "q": "কির্শফের প্রথম সূত্রটি লেখো। তিনটি রোধ সমান্তরালে যুক্ত: R₁=2Ω, R₂=3Ω, R₃=6Ω — সমতুল্য রোধ কত?",
     "must": ["কির্শফ", "সমান্তরাল"], "must_not": [], "fig": False},
    {"id": 18, "ch": 12, "q": "তড়িৎ চুম্বকীয় আবেশ কী? বইয়ের ১২.৩ ধারায় এই ঘটনার চিত্রটি বর্ণনা করো।",
     "must": ["আবেশ", "চৌম্বক"], "must_not": [], "fig": True},
    {"id": 19, "ch": 12, "q": "ট্রান্সফর্মারের প্রাইমারিতে 100 প্যাঁচ ও সেকেন্ডারিতে 1000 প্যাঁচ। প্রাইমারিতে 10 V AC দিলে সেকেন্ডারিতে কত ভোল্ট? DC দিলে কী হবে?",
     "must": ["ট্রান্সফর্মার", "100"], "must_not": [], "fig": False},
    {"id": 20, "ch": 13, "q": "তেজস্ক্রিয় মৌলের অর্ধায়ু কাকে বলে? 1 kg নমুনার অর্ধায়ু 100 বছর হলে 200 বছর পর কত kg অবশিষ্ট থাকবে?",
     "must": ["অর্ধায়ু", "তেজস্ক্রিয়"], "must_not": [], "fig": False},

    # ── 10 Location-Specific Questions (Q21–Q30) ────────────────────────────
    # Pattern: section number, page number, example number, নিজে করো,
    # অনুসন্ধান, নমুনা প্রশ্ন, সংক্ষিপ্ত প্রশ্ন, সৃজনশীল — real refs from book
    {"id": 21, "ch": 1,  "loc_type": "section",
     "q": "অধ্যায় ১ এর ধারা ১.৩ তে কী আলোচনা করা হয়েছে? এই ধারার মূল বিষয়বস্তু কী?",
     "must": ["পরিমাপ", "রাশি"], "must_not": [], "fig": False},

    {"id": 22, "ch": 2,  "loc_type": "onusondhan",
     "q": "অধ্যায় ২ এর অনুসন্ধান ২.০১ কোন পৃষ্ঠায় আছে এবং এর উদ্দেশ্য কী?",
     "must": ["অনুসন্ধান", "গড়"], "must_not": [], "fig": False},

    {"id": 23, "ch": 3,  "loc_type": "example",
     "q": "পৃষ্ঠা ৬৫ তে অধ্যায় ৩ এর কোন উদাহরণ আছে? সেটি সমাধান করে দাও।",
     "must": ["বল", "নিউটন"], "must_not": [], "fig": False},

    {"id": 24, "ch": 4,  "loc_type": "niche_koro",
     "q": "অধ্যায় ৪ এর পৃষ্ঠা ১০৭ তে নিজে করো অংশে কী আছে? প্রশ্নটি বুঝিয়ে দাও।",
     "must": ["কাজ", "ক্ষমতা"], "must_not": [], "fig": False},

    {"id": 25, "ch": 5,  "loc_type": "shankhipto",
     "q": "অধ্যায় ৫ এর সংক্ষিপ্ত প্রশ্ন ৩ নং এর উত্তর কী?",
     "must": ["চাপ", "পারদ"], "must_not": [], "fig": False},

    {"id": 26, "ch": 6,  "loc_type": "section",
     "q": "অধ্যায় ৬ এর ধারা ৬.৩ তে কী আলোচনা হয়েছে? মূল সূত্রটি কী?",
     "must": ["তাপ", "প্রসারণ"], "must_not": [], "fig": False},

    {"id": 27, "ch": 7,  "loc_type": "creative",
     "q": "অধ্যায় ৭ এর সৃজনশীল প্রশ্ন ১ নং এর ক অংশ কী এবং তার উত্তর কী হবে?",
     "must": ["তরঙ্গ", "শব্দ"], "must_not": [], "fig": False},

    {"id": 28, "ch": 9,  "loc_type": "nomuna_proshno",
     "q": "অধ্যায় ৯ এর নমুনা প্রশ্ন ২ নং কী? সঠিক উত্তরটি কোনটি?",
     "must": ["আলো", "প্রতিসরণ"], "must_not": [], "fig": False},

    {"id": 29, "ch": 10, "loc_type": "onusondhan",
     "q": "অধ্যায় ১০ এর অনুসন্ধান ১০.০১ কোন পৃষ্ঠায় আছে এবং এর বিষয় কী?",
     "must": ["আধান", "ঘর্ষণ"], "must_not": [], "fig": False},

    {"id": 30, "ch": 11, "loc_type": "section",
     "q": "অধ্যায় ১১ এর ধারা ১১.২ তে কোন বিষয় আলোচনা করা হয়েছে? এই ধারার শিরোনাম কী?",
     "must": ["বিভব", "তড়িৎ"], "must_not": [], "fig": False},
]

# ── Evaluation ─────────────────────────────────────────────────────────────
results = []
print("\n" + "="*65)
print("PHASE 2 — Critical Question Evaluation")
print("="*65)

for qd in QUESTIONS:
    qid = qd["id"]
    q   = qd["q"]
    exp_ch = qd["ch"]
    must   = qd["must"]
    mnot   = qd["must_not"]

    try:
        qvec = embed(q)
        res, hints = retriever.retrieve(q, qvec, top_k=8)
        ctx  = build_context(res, hints)

        # Figure injection
        if qd.get("fig"):
            ctx = inject_figures_into_context(ctx, res, max_total_figs=4)

        chapters = sorted(set(
            r.get("chapter", r.get("chapter_num", 0)) for r in res
            if r.get("chapter") or r.get("chapter_num")
        ))
        pages = sorted(set(
            r.get("page_start", r.get("book_page_start", 0)) for r in res
        ))
        sections = [r.get("section", "") for r in res[:3]]
        has_fig_ctx = "সংশ্লিষ্ট চিত্রের বর্ণনা" in ctx if qd.get("fig") else None

        missing  = [kw for kw in must if kw not in ctx]
        polluted = [kw for kw in mnot if kw in ctx]
        ch_ok    = exp_ch in chapters

        if ch_ok and not missing and not polluted:
            if qd.get("fig") and not has_fig_ctx:
                status = "WARN-FIG"
            else:
                status = "PASS"
        elif ch_ok and not missing:
            status = "WARN"
        elif ch_ok:
            status = "PARTIAL"
        else:
            status = "FAIL"

        results.append({
            "id": qid, "ch": exp_ch, "status": status,
            "chapters": chapters, "pages": pages,
            "missing": missing, "polluted": polluted,
            "has_fig": has_fig_ctx,
            "ctx_preview": ctx[:400],
            "sections": sections,
        })

        icon = "✅" if status=="PASS" else ("⚠️" if status.startswith("WARN") else ("🔶" if status=="PARTIAL" else "❌"))
        fig_note = f" [fig={'✓' if has_fig_ctx else '✗'}]" if qd.get("fig") else ""
        print(f"[Q{qid:02d}] Ch={exp_ch} {icon} {status}{fig_note} | ch_found={chapters} | miss={missing}")

    except Exception as e:
        results.append({"id": qid, "ch": exp_ch, "status": "ERROR", "error": str(e)})
        print(f"[Q{qid:02d}] Ch={exp_ch} 💥 ERROR: {e}")

# ── Summary ────────────────────────────────────────────────────────────────
from collections import Counter
counts = Counter(r["status"] for r in results)
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
for s in ["PASS","WARN","WARN-FIG","PARTIAL","FAIL","ERROR"]:
    n = counts.get(s, 0)
    bar = "█"*(n*2)
    print(f"  {s:<10}: {n:2d}  {bar}")

# Save full results
with open("scripts/phase2_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nFull results saved → scripts/phase2_results.json")

# ── Failures detail ─────────────────────────────────────────────────────────
failures = [r for r in results if r["status"] not in ("PASS","WARN")]
if failures:
    print("\n" + "-"*65)
    print("FAILURES / PARTIALS / WARN-FIG — Detail:")
    print("-"*65)
    for r in failures:
        print(f"\n[Q{r['id']}] Ch={r['ch']} → {r['status']}")
        print(f"  ch_found : {r.get('chapters', [])}")
        print(f"  pages    : {r.get('pages', [])}")
        print(f"  missing  : {r.get('missing', [])}")
        print(f"  sections : {r.get('sections', [])}")
        if r.get("has_fig") is not None:
            print(f"  has_fig  : {r['has_fig']}")
        print(f"  ctx[0:300]: {r.get('ctx_preview','')[:300]}")
