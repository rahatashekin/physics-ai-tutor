"""Retry 4 failed figure descriptions from ch=13."""
import io, sys, os, json, base64
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
import time

FAILED_FIGS = [
    "fig_page362_0.png",
    "fig_page362_1.png",
    "fig_page364_0.png",
    "fig_page364_1.png",
]
FIG_DIR  = Path("data/processed/figures")
OUT_PATH = Path("data/processed/figure_descriptions.json")
PROMPT   = "এই পদার্থবিজ্ঞানের চিত্রটি বাংলায় বর্ণনা করো। চিত্রে কী দেখানো হয়েছে, পদার্থবিজ্ঞানগত তাৎপর্য কী সেটা সহজ ভাষায় লেখো।"

def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    with open(OUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    
    for fn in FAILED_FIGS:
        fp = FIG_DIR / fn
        if not fp.exists():
            print(f"  NOT FOUND: {fn}")
            continue
        
        img_b64 = base64.b64encode(fp.read_bytes()).decode()
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                }],
                max_tokens=300,
            )
            desc = resp.choices[0].message.content.strip()
            data[fn] = {
                "key":          fn,
                "image_path":   str(fp),
                "book_page":    int(fn.split("page")[1].split("_")[0]) - 5,  # pdf→book offset
                "chapter_num":  13,
                "index_on_page": int(fn.split("_")[-1].replace(".png", "")),
                "figure_id":    "",
                "caption":      "",
                "description":  desc,
            }
            print(f"  ✓ {fn}: {desc[:60]}")
            time.sleep(1)
        except Exception as e:
            print(f"  ERROR {fn}: {e}")
    
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nTotal figures: {len(data)}")
    print("Done! ✓")

main()
