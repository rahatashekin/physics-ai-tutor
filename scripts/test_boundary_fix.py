"""Test that BookIndex boundary fix removed ch=8 contamination from ch=4 onusondhan."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from src.book_index import BookIndex
bidx = BookIndex()
bidx.build()

CH8_MARKERS = ['পাহাড়ি রাস্তার', '৮.৮', 'আলোর প্রতিফলন', 'আয়নার ব্যবহার']

# Check ch=4 onusondhan
ch4 = bidx.get_onusondhan(4)
print(f'Ch4 onusondhan count: {len(ch4)}')
for o in ch4:
    text = o['text']
    has_ch8 = any(m in text for m in CH8_MARKERS)
    print(f'  id={o["id"]} len={len(text)} chars  ch8_contaminated={has_ch8}')
    if has_ch8:
        for m in CH8_MARKERS:
            if m in text:
                idx = text.find(m)
                print(f'    CONTAMINATION at char {idx}: ...{text[max(0,idx-30):idx+50]}...')

# Check ch=7 shankhipto
ch7 = bidx._chapters.get(7)
if ch7:
    sh = ch7.shankhipto_text
    has_ch8 = any(m in sh for m in CH8_MARKERS)
    print(f'\nCh7 shankhipto len={len(sh)} chars  ch8_contaminated={has_ch8}')
    if not has_ch8:
        print(f'  last 120 chars: {sh[-120:]}')

print('\n[PASS] BookIndex boundary fix verified.' if not any(
    any(m in o['text'] for m in CH8_MARKERS) for o in ch4
) else '[FAIL] Contamination still present.')
