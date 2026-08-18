"""
scripts/6_describe_figures.py
==============================
PURPOSE:
    বইয়ের সব figure এর Bengali description তৈরি করা।
    GPT-4o-mini Vision ব্যবহার করা হবে।
    Results → data/processed/figure_descriptions.json এ save হবে।
    
    Re-runnable: আগে যেগুলো হয়েছে সেগুলো skip করবে।
"""
import io, sys, json, base64, time, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FIGURE_LOC_PATH = Path("data/processed/figure_locations.json")
OUTPUT_PATH     = Path("data/processed/figure_descriptions.json")
FIGURES_DIR     = Path("data/processed/figures")

# Chapter page ranges (same as fix_chapter_by_page.py)
CHAPTER_RANGES = [
    (1,   1,   31),  (2,  32,  61),  (3,  62,  97),
    (4,  98,  126),  (5, 127, 158),  (6, 159, 185),
    (7, 186,  209),  (8, 210, 240),  (9, 241, 269),
    (10, 270, 297),  (11, 298, 328), (12, 329, 345),
    (13, 346, 366),
]

def page_to_chapter(page: int) -> int:
    for ch, s, e in CHAPTER_RANGES:
        if s <= page <= e:
            return ch
    return -1

VISION_PROMPT = """তুমি একজন নবম-দশম শ্রেণির পদার্থবিজ্ঞান শিক্ষক।
এই চিত্রটি বাংলাদেশের HSC পদার্থবিজ্ঞান পাঠ্যবই থেকে নেওয়া।

নিচের format এ সংক্ষিপ্ত কিন্তু সম্পূর্ণ বাংলা বর্ণনা দাও (max 150 শব্দ):

1. চিত্রে কী দেখানো হয়েছে (এক বাক্যে)
2. পদার্থবিজ্ঞানগত তাৎপর্য (যদি থাকে)
3. যদি graph/chart হয় — axes, curve বা pattern বর্ণনা করো
4. যদি circuit/diagram হয় — components এর বর্ণনা দাও
5. যদি শুধু decorative/portrait হয় — "এটি একটি [বিষয়] এর ছবি" লিখলেই হবে

শুধু বাংলায় লিখবে।"""

def describe_figure(image_path: Path) -> str:
    """GPT-4o-mini Vision দিয়ে figure describe করো।"""
    with open(image_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",  "text": VISION_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "low"}},
                ],
            }
        ],
        max_tokens=250,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def main():
    # Load figure locations
    with open(FIGURE_LOC_PATH, encoding="utf-8") as f:
        figures = json.load(f)

    # Load existing descriptions (resume support)
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    # Filter: only content pages (book_page > 0)
    content_figures = [fig for fig in figures if fig.get("book_page", -1) > 0]
    print(f"Total figures in JSON: {len(figures)}")
    print(f"Content figures (book_page > 0): {len(content_figures)}")
    print(f"Already described: {len(existing)}")
    print()

    errors = 0
    for i, fig in enumerate(content_figures):
        img_path = Path(fig["image_path"])
        key      = img_path.name   # e.g. "fig_page039_0.png"

        if key in existing:
            continue   # already done

        if not img_path.exists():
            print(f"  [SKIP] File not found: {img_path}")
            continue

        book_pg = fig["book_page"]
        ch      = page_to_chapter(book_pg)
        pg_idx  = fig["index_on_page"]

        print(f"[{i+1}/{len(content_figures)}] ch={ch} pg={book_pg} idx={pg_idx} → {key}", end=" ", flush=True)

        try:
            desc = describe_figure(img_path)
            existing[key] = {
                "key":          key,
                "image_path":   str(img_path),
                "book_page":    book_pg,
                "chapter_num":  ch,
                "index_on_page": pg_idx,
                "figure_id":    fig.get("figure_id", ""),
                "caption":      fig.get("caption", ""),
                "description":  desc,
            }
            print("✓")
        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")
            time.sleep(5)
            continue

        # Save after every figure (resume-safe)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        time.sleep(0.3)   # rate limit friendly

    print(f"\nDone. Described: {len(existing)}, Errors: {errors}")
    print(f"Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
