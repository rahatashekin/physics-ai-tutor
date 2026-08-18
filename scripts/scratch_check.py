import io, sys, os, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
import lancedb
from dotenv import load_dotenv
load_dotenv()

from src.retrieval import PhysicsTutorRetriever, build_context

db = lancedb.connect('data/lancedb')
cdf = db.open_table('child_chunks').to_pandas()
pdf = db.open_table('parent_chunks').to_pandas()
cvec = np.array(cdf['vector'].tolist(), dtype=np.float32)
r = PhysicsTutorRetriever(cdf, cvec, pdf)

from vertexai.language_models import TextEmbeddingModel
import vertexai
vertexai.init(project=os.getenv("GCP_PROJECT"), location="us-central1")
_em = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")
def embed(t): return np.array(_em.get_embeddings([t])[0].values, dtype=np.float32)

tests = [
    (1, 'স্ক্রু গেইজের ন্যূনাংক কত?'),
    (3, 'নিউটনের তৃতীয় সূত্রটি লেখো।'),
    (11, 'ওহমের সূত্র ও সমান্তরাল বর্তনীতে রোধ নির্ণয়।'),
    (12, 'ট্রান্সফর্মারের কার্যনীতি কী?'),
]
all_ok = True
for ch, q in tests:
    qv = embed(q)
    res, hints = r.retrieve(q, qv, top_k=8)
    chs = sorted(set(x.get('chapter',0) for x in res))
    ctx = build_context(res, hints)
    rescue = any(x.get('intent')=='keyword_rescue' for x in res)
    print(f"Ch{ch}: {q[:35]}...")
    print(f"  chapters={chs}, rescue={rescue}, ctx_len={len(ctx)}")
    if ch not in chs:
        print(f"  ❌ Ch{ch} NOT in results!")
        all_ok = False
    else:
        print(f"  ✅ Ch{ch} found")

print()
print('✅ All vector searches working!' if all_ok else '❌ Some searches failed')
