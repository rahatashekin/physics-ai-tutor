"""
scripts/redescribe_wrong_figures.py
=====================================
Fix 2 (Correct version): Wrong figure descriptions কে AI Vision দিয়ে re-describe করো।

Design (no hardcoding):
- "Wrong" detection: electrical circuit keywords in non-electricity chapters
  (ch 10 = স্থির তড়িৎ, ch 11 = চল তড়িৎ এ circuit সঠিক; বাকিতে circuit wrong)
- Re-description: GPT-4o-mini Vision + dynamic chapter context prompt
- Update: figure_descriptions.json + LanceDB parent_chunks

This script is idempotent — run again to re-check/fix remaining wrong entries.
"""
import io, sys, json, base64, time, os, pathlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import lancedb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client   = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
FIG_JSON = pathlib.Path('data/processed/figure_descriptions.json')

# ── Chapter names (full list, not hardcoded to any specific chapter) ────────
# Used to give AI the chapter context so it doesn't hallucinate wrong topics
CHAPTER_NAMES = {
    1:  "ভৌত রাশি ও পরিমাপ",
    2:  "গতি",
    3:  "বল",
    4:  "কাজ, ক্ষমতা ও শক্তি",
    5:  "পদার্থের অবস্থা ও চাপ",
    6:  "বস্তুর উপর তাপের প্রভাব",
    7:  "তরঙ্গ ও শব্দ",
    8:  "আলোর প্রতিফলন",
    9:  "আলোর প্রতিসরণ",
    10: "স্থির তড়িৎ",
    11: "চল তড়িৎ",
    12: "বিদ্যুতের চুম্বকীয় ক্রিয়া",
    13: "আধুনিক পদার্থবিজ্ঞান ও ইলেকট্রনিক্স",
}

# Chapters where electrical/circuit descriptions ARE expected to be correct
# ch=10: স্থির তড়িৎ, ch=11: চল তড়িৎ, ch=12: বিদ্যুতের চুম্বকীয় ক্রিয়া
# ch=13: আধুনিক পদার্থবিজ্ঞান ও ইলেকট্রনিক্স (electronics circuits expected)
ELECTRICITY_CHAPTERS = {10, 11, 12, 13}

# Markers that indicate an electrical circuit description
ELECTRICAL_MARKERS = ['সার্কিট', 'ব্যাটারি', 'প্রতিরোধক', 'বৈদ্যুতিক সার্কিট']


def is_wrong_description(chapter_num: int, description: str) -> bool:
    """
    Returns True if the description is likely wrong for the given chapter.
    No chapter-specific hardcoding — uses general heuristic:
    - Explicit NEEDS_REDESCRIPTION marker → always wrong
    - Circuit keywords in non-electricity chapters → wrong
    """
    if description == 'NEEDS_REDESCRIPTION':
        return True  # explicitly marked for redescription
    if chapter_num in ELECTRICITY_CHAPTERS:
        return False  # circuit description is valid for electricity chapters
    return any(marker in description for marker in ELECTRICAL_MARKERS)


def make_prompt(chapter_num: int) -> str:
    """Dynamic prompt with chapter context — no page/section hardcoding."""
    ch_name = CHAPTER_NAMES.get(chapter_num, "পদার্থবিজ্ঞান")
    return f"""তুমি একজন নবম-দশম শ্রেণির পদার্থবিজ্ঞান শিক্ষক।
এই চিত্রটি বাংলাদেশের নবম-দশম শ্রেণির পদার্থবিজ্ঞান পাঠ্যবইয়ের
"{ch_name}" অধ্যায় থেকে নেওয়া।

অধ্যায়ের বিষয়: {ch_name}

নিচের format এ সংক্ষিপ্ত কিন্তু সম্পূর্ণ বাংলা বর্ণনা দাও (max 120 শব্দ):
1. চিত্রে কী দেখানো হয়েছে (এক বাক্যে)
2. এই অধ্যায়ের ({ch_name}) সাথে এর সম্পর্ক
3. যদি ray diagram/lens/mirror হয় — কী ধরনের প্রতিফলন/প্রতিসরণ দেখানো হয়েছে
4. যদি graph হয় — axes ও curve বর্ণনা করো
5. যদি portrait/photo হয় — এটি কোন physics concept illustrate করছে

শুধু বাংলায় লিখবে। অধ্যায়ের বিষয়ের সাথে সামঞ্জস্যপূর্ণ বর্ণনা দাও।"""


def redescribe(image_path: str, chapter_num: int) -> str:
    """Re-describe a figure using GPT-4o-mini Vision with chapter context."""
    with open(image_path, 'rb') as f:
        img_b64 = base64.standard_b64encode(f.read()).decode('utf-8')

    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': make_prompt(chapter_num)},
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/png;base64,{img_b64}', 'detail': 'low'}},
            ],
        }],
        max_tokens=200,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def main():
    # Load current descriptions
    with open(FIG_JSON, encoding='utf-8') as f:
        figs = json.load(f)

    # Find wrong entries
    wrong = {
        key: val for key, val in figs.items()
        if is_wrong_description(val.get('chapter_num', 0), val.get('description', ''))
    }

    print(f'Total figures: {len(figs)}')
    print(f'Wrong descriptions found: {len(wrong)}')
    print()

    if not wrong:
        print('No wrong descriptions found. Done.')
        return

    # Show what will be re-described
    for key, val in wrong.items():
        print(f'  ch={val["chapter_num"]} pg={val["book_page"]} → {key}')
        print(f'    current: {val["description"][:80]}')
        print()

    # Re-describe each wrong figure
    fixed = 0
    for key, val in wrong.items():
        img_path = val.get('image_path', '')
        ch       = val.get('chapter_num', 0)
        pg       = val.get('book_page', 0)

        if not pathlib.Path(img_path).exists():
            print(f'[SKIP] Image not found: {img_path}')
            continue

        print(f'Redescribing ch={ch} pg={pg} {key}...', end=' ', flush=True)
        try:
            new_desc = redescribe(img_path, ch)
            figs[key]['description'] = new_desc
            print('✓')
            print(f'  NEW: {new_desc[:80]}')
            fixed += 1
        except Exception as e:
            print(f'ERROR: {e}')
            time.sleep(3)
            continue

        # Save after each (resume-safe)
        with open(FIG_JSON, 'w', encoding='utf-8') as f:
            json.dump(figs, f, ensure_ascii=False, indent=2)

        time.sleep(0.5)   # rate limit

    print(f'\nFixed {fixed}/{len(wrong)} wrong descriptions in figure_descriptions.json')

    # ── Update LanceDB parent_chunks ────────────────────────────────────────
    if fixed == 0:
        return

    print('\nUpdating LanceDB parent_chunks...')
    db  = lancedb.connect('data/lancedb')
    tbl = db.open_table('parent_chunks')
    df  = tbl.to_pandas()

    update_count = 0
    for key, val in wrong.items():
        new_desc = figs[key]['description']
        pg = val.get('book_page')
        ch = val.get('chapter_num')
        if pg is None or ch is None:
            continue

        mask = (
            (df['content_type'] == 'figure') &
            (df['chapter_num'] == ch) &
            (df['book_page_start'] == pg)
        )
        matched = df[mask]
        if matched.empty:
            continue

        for idx in matched.index:
            old_text = df.at[idx, 'text']
            header_end = old_text.find('\n')
            new_text = (old_text[:header_end + 1] + new_desc
                       if header_end >= 0 else new_desc)
            df.at[idx, 'text'] = new_text
            update_count += 1

    if update_count > 0:
        tbl.delete('1=1')
        tbl.add(df)
        print(f'Updated {update_count} DB chunks.')

    # Verify: check if any wrong descriptions remain
    remaining_wrong = sum(
        1 for v in figs.values()
        if is_wrong_description(v.get('chapter_num', 0), v.get('description', ''))
    )
    print(f'\nRemaining wrong descriptions: {remaining_wrong}')
    print('[DONE]')


if __name__ == '__main__':
    main()
