"""
src/retrieval.py
=================
Intent-based smart retrieval for the physics tutor.

Intent types:
  CHAPTER_OVERVIEW  → list all sections of a chapter (metadata)
  SECTION_LOOKUP    → get content for a section like 2.6 (metadata)
  PAGE_LOOKUP       → get all content on a specific page (metadata)
  NICHE_KORO        → নিজে করো section (metadata + content_type)
  ONUSONDHAN        → অনুসন্ধান section (metadata + content_type)
  NOMUNA_PROSHNO    → নমুনা প্রশ্ন / MCQ (metadata + content_type)
  SHANKHIPTO        → সংক্ষিপ্ত প্রশ্ন (metadata + content_type)
  CREATIVE_Q        → সৃজনশীল প্রশ্ন (metadata + content_type)
  GENERAL           → vector search fallback
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import lancedb
import numpy as np
import pandas as pd

from src.book_index import get_book_index


# ─────────────────────────────────────────────────────────
# Intent Enum
# ─────────────────────────────────────────────────────────
class QueryIntent(Enum):
    CHAPTER_OVERVIEW = "chapter_overview"
    SECTION_LOOKUP   = "section_lookup"
    PAGE_LOOKUP      = "page_lookup"
    NICHE_KORO       = "niche_koro"
    ONUSONDHAN       = "onusondhan"
    NOMUNA_PROSHNO   = "nomuna_proshno"
    SHANKHIPTO       = "shankhipto"
    CREATIVE_Q       = "creative_question"
    GENERAL          = "general"


# ─────────────────────────────────────────────────────────
# Hint dataclass
# ─────────────────────────────────────────────────────────
@dataclass
class QueryHints:
    chapter:      Optional[int]   = None
    page:         Optional[int]   = None
    section:      Optional[str]   = None    # e.g. "5.3" (parent of 5.3.1)
    is_subsection: bool           = False   # True when query had 3-part like 5.3.1
    question_num: Optional[int]   = None    # e.g. 3
    intent:       QueryIntent     = QueryIntent.GENERAL
    raw:          str             = ""


# ─────────────────────────────────────────────────────────
# Bengali helpers
# ─────────────────────────────────────────────────────────
_BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def _bn2int(s: str) -> int:
    return int(s.translate(_BN2EN))

def _parse_num(s: str) -> Optional[int]:
    m = re.search(r"[০-৯]+", s)
    if m:
        return _bn2int(m.group())
    m = re.search(r"\d+", s)
    if m:
        return int(m.group())
    return None

_CHAPTER_MAP = {
    "প্রথম": 1, "দ্বিতীয়": 2, "তৃতীয়": 3, "চতুর্থ": 4,
    "পঞ্চম": 5, "ষষ্ঠ": 6, "সপ্তম": 7, "অষ্টম": 8,
    "নবম": 9, "দশম": 10, "একাদশ": 11, "দ্বাদশ": 12, "ত্রয়োদশ": 13,
}


# ─────────────────────────────────────────────────────────
# Hint extraction
# ─────────────────────────────────────────────────────────
def extract_hints(query: str) -> QueryHints:
    h = QueryHints(raw=query)

    # Chapter number
    m = re.search(r"(?:chapter|অধ্যায়)\s*([০-৯\d]+)", query, re.IGNORECASE)
    if m:
        raw = m.group(1)
        h.chapter = _bn2int(raw) if re.search(r"[০-৯]", raw) else int(raw)
    else:
        # Ordinal words (প্রথম=1, দ্বিতীয়=2, ...) only signal chapter when used as
        # chapter ordinals — NOT when they modify laws/rules ("প্রথম সূত্র" = Kirchhoff's 1st law).
        # Guard: skip if ordinal is immediately followed by সূত্র/বিধি/নিয়ম/তত্ত্ব/ধাপ.
        _LAW_WORDS_RE = re.compile(r"\s*(?:সূত্র|বিধি|নিয়ম|তত্ত্ব|ধাপ|আইন|ধর্ম|বৈশিষ্ট্য|শর্ত)")
        for word, num in _CHAPTER_MAP.items():
            if word in query:
                idx = query.index(word)
                after = query[idx + len(word):]
                if _LAW_WORDS_RE.match(after):
                    continue  # "প্রথম সূত্র" → not a chapter reference
                h.chapter = num
                break

    # Page number
    m = re.search(r"(?:page|পৃষ্ঠা|পেজ)\s*([০-৯\d]+)", query, re.IGNORECASE)
    if not m:
        m = re.search(r"([০-৯\d]+)\s*পৃষ্ঠা", query)
    if not m:
        m = re.search(r"(\d+)(?:th|st|nd|rd)?\s*page", query, re.IGNORECASE)
    if m:
        raw = m.group(1)
        h.page = _bn2int(raw) if re.search(r"[০-৯]", raw) else int(raw)

    # Section — capture 3-part (5.3.1) OR 2-part (2.6)
    # Guard: X.Y is a physical constant (not a section) when followed by a unit.
    _UNIT_AFTER_RE = re.compile(
        r"\s*(?:m/s²|m/s2|m/s|g/cc|g/cm³|g/cm3|kg/m³|kg/m3"
        r"|kg\b|cm\b|mm\b|μm|nm|°C|°K|°F|J\b|kJ\b|W\b|kW\b"
        r"|A\b|mA\b|V\b|Ω\b|N\b|Pa\b|Hz\b|kHz\b|MHz\b|eV\b)",
        re.IGNORECASE,
    )
    # Scientific notation immediately after number: "1.52 × 10", "1.2×10⁻⁵", "3e8"
    _SCI_NOTATION_RE = re.compile(
        r"\s*(?:[×x]\s*10|[eE][+\-]|\u207b|\u207a|[⁰¹²³⁴⁵⁶⁷⁸⁹])",
    )

    def _is_physical_constant(match_end: int, minor: str = "") -> bool:
        after = query[match_end:match_end + 20]
        # Unit after number → physical constant
        if _UNIT_AFTER_RE.match(after):
            return True
        # Scientific notation follows → physical constant (e.g. 1.2×10⁻⁵)
        if _SCI_NOTATION_RE.match(after):
            return True
        # Section minor part sanity: sections are like "2.6", "11.3", never "1.52"
        # Minor part > 15 means it's a physical constant (no chapter has 52 sections)
        if minor:
            try:
                if int(minor) > 15:
                    return True
            except ValueError:
                pass
        return False

    m3 = re.search(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)", query)   # 3-part: 5.3.1
    m2 = re.search(r"(?<!\d)(\d+\.\d+)(?!\d)", query)         # 2-part: 5.3
    if m3 and not _is_physical_constant(m3.end()):
        full_sec = m3.group(1).translate(_BN2EN)               # Bengali → ASCII
        parent   = ".".join(full_sec.split(".")[:2])           # "5.3"
        h.section        = parent
        h.is_subsection  = True
        if h.chapter is None:
            try:
                h.chapter = int(full_sec.split(".")[0])
            except ValueError:
                pass
    elif m2:
        raw_sec = m2.group(1).translate(_BN2EN)                # Bengali → ASCII
        # Sanity: major digit must be a plausible chapter number (1–13)
        try:
            major = int(raw_sec.split(".")[0])
            minor = raw_sec.split(".")[1]
        except (ValueError, IndexError):
            major = 0
            minor = ""
        if 1 <= major <= 13 and not _is_physical_constant(m2.end(), minor):
            h.section = raw_sec
            h.is_subsection = False
            if h.chapter is None:
                h.chapter = major


    # Question number: "৩ নং", "3 number", "তৃতীয় প্রশ্ন"
    m = re.search(r"([০-৯\d]+)\s*(?:নং|নম্বর|number|no\.?)\b", query, re.IGNORECASE)
    if m:
        raw = m.group(1)
        h.question_num = _bn2int(raw) if re.search(r"[০-৯]", raw) else int(raw)

    # Intent
    if "নিজে করো" in query:
        h.intent = QueryIntent.NICHE_KORO
    elif "অনুসন্ধান" in query:
        h.intent = QueryIntent.ONUSONDHAN
    elif any(kw in query for kw in ["সংক্ষিপ্ত প্রশ্ন", "সংক্ষিপ্ত উত্তর"]):
        h.intent = QueryIntent.SHANKHIPTO
    elif any(kw in query for kw in ["নমুনা প্রশ্ন", "বহুনির্বাচনি", "MCQ", "mcq"]):
        h.intent = QueryIntent.NOMUNA_PROSHNO
    elif "সৃজনশীল" in query:
        h.intent = QueryIntent.CREATIVE_Q
    elif (h.chapter is not None
          and h.page is None
          and h.section is None
          and any(kw in query for kw in [
              "section", "ধারা", "অংশ", "overview", "তালিকা",
              "সব কিছু", "পড়িয়ে দাও", "পুরো", "কী কী আছে",
              "প্রতিটা", "সব section", "সব ধারা",
          ])):
        h.intent = QueryIntent.CHAPTER_OVERVIEW
    elif h.section is not None:
        h.intent = QueryIntent.SECTION_LOOKUP
    elif h.page is not None:
        h.intent = QueryIntent.PAGE_LOOKUP
    else:
        h.intent = QueryIntent.GENERAL

    return h


# ─────────────────────────────────────────────────────────
# Keyword helpers
# ─────────────────────────────────────────────────────────
_BN_STOP = {
    "কী", "কি", "কোনটি", "কাকে", "বলে", "কেন", "কীভাবে",
    "হয়", "করো", "নির্ণয়", "হলে", "পরে", "কত", "কোথায়",
    "এবং", "বা", "থেকে", "এর", "তার", "এই", "একটি", "একটা",
    "দিয়ে", "নিয়ে", "আছে", "করে", "হলো", "হবে", "সেটা",
    "আমি", "আমার", "তুমি", "তোমার", "পারবে", "পারব",
    "বোঝাও", "বুঝিয়ে", "দাও", "দেখো", "বলো", "ধরে",
    "কোন", "কোনো", "কিভাবে", "ব্যাখ্যা", "সমাধান",
}

def _strip_suffix(w: str) -> str:
    # "সের" first: handles English possessive transliterations (স্নেলসের→স্নেল, নিউটনসের→নিউটন)
    for suf in ["সের", "ের", "এর", "টির", "টার", "দের", "কে", "টি", "টা", "তে", "য়"]:
        if w.endswith(suf) and len(w) - len(suf) >= 2:
            return w[:-len(suf)]
    return w

# OCR artifacts: apostrophes in Bengali transliterations of English names
# e.g. "ও'মের সূত্র" (Ohm's Law) should match "ওহম" in query and vocab
_BN_OCR_NORMALIZE: dict[str, str] = {
    "ও'ম":      "ওহম",    # Ohm
    "ও'মের":    "ওহমের",  # Ohm's
    "ও'মের্স":  "ওহম",
}

# Query-side synonym mapping: common student terms → book-equivalent terms.
# Applied at extract_keywords() time so the query keywords match book vocab.
# Pattern: book uses "পড়ন্ত বস্তু" but student writes "মুক্তপতন".
# We append the book-equivalent token so IDF scoring can match.
_BN_QUERY_SYNONYMS: list[tuple[str, str]] = [
    # Student term              # Book equivalent token (appended to query)
    ("মুক্তপতন",               "পড়ন্ত"),        # Free fall → ch=2
    ("মুক্তভাবে পড়ন্ত",        "পড়ন্ত"),        # Free fall → ch=2
    ("কির্শফের",               "কির্শফের_সূত্র"),# Kirchhoff → synthetic → ch=11
    ("কির্শফ",                 "কির্শফের_সূত্র"),# Kirchhoff → synthetic → ch=11
    ("লেঞ্জের",                "লেঞ্জ_সূত্র"),   # Lenz's law → ch=12
    ("রৈখিক প্রসারণ সহগ",     "রৈখিকপ্রসারণসহগ"), # Linear expansion → ch=6 compound
    ("রৈখিক প্রসারণ",         "রৈখিকপ্রসারণসহগ"), # Linear expansion short → ch=6 compound
    ("দৈর্ঘ্য প্রসারণ সহগ",   "রৈখিকপ্রসারণসহগ"), # Alternative phrasing → ch=6
    ("দৈর্ঘ্য প্রসারণ",       "রৈখিকপ্রসারণসহগ"), # Alternative phrasing → ch=6
    ("ভর শক্তি সম্পর্ক",      "ভরশক্তিসম্পর্ক"),  # Mass-energy relation → ch=4 compound
    ("ভর ও শক্তির সম্পর্ক",   "ভরশক্তিসম্পর্ক"),  # Mass-energy relation → ch=4 compound
    ("E=mc",                   "ভরশক্তিসম্পর্ক"),  # E=mc² → ch=4 compound
    ("তৃতীয় সূত্র",            "বিক্রিয়া"),         # Newton 3rd law → ch=3 (ক্রিয়া-বিক্রিয়া)
    ("নিউটনের তৃতীয়",         "বিক্রিয়া"),         # Newton 3rd → ch=3
    ("সমান্তরাল বর্তনী",        "সমান্তরাল"),        # Parallel circuit → ch=11
    ("সমান্তরাল সংযোগ",        "সমান্তরাল"),        # Parallel connection → ch=11

    # ── Chapter 1: Measurement ──────────────────────────────────────────────
    ("স্ক্রু-গেইজ",             "স্ক্রু"),            # Screw gauge → ch=1
    ("স্ক্রুগেইজ",              "স্ক্রু"),            # Screw gauge no hyphen → ch=1
    ("ন্যূনাংক",                "ক্ষুদ্রতম"),         # Least count synonym
    ("ভার্নিয়ার",               "স্লাইড"),            # Vernier → Slide caliper ch=1
    ("সিজিএস",                  "CGS"),               # CGS unit system
    ("এসআই",                    "SI"),                # SI unit system

    # ── Chapter 2: Motion ───────────────────────────────────────────────────
    ("বেগ-সময়",                "বেগ"),               # v-t graph → ch=2
    ("সরণ-সময়",                "সরণ"),               # s-t graph → ch=2
    ("গড় বেগ",                  "গড়বেগ"),            # average velocity
    ("তাৎক্ষণিক বেগ",           "তাৎক্ষণিকবেগ"),     # instantaneous velocity
    ("সমত্বরণ",                 "ত্বরণ"),             # uniform acceleration → ch=2
    ("অভিকর্ষজ ত্বরণ",         "মাধ্যাকর্ষণ"),       # gravitational acceleration
    ("g এর মান",                "9.8"),               # value of g

    # ── Chapter 3: Force ────────────────────────────────────────────────────
    ("ক্রিয়া বিক্রিয়া",         "বিক্রিয়া"),         # action-reaction → ch=3
    ("আপাত ওজন",                "আপাত"),             # apparent weight → ch=3
    ("নিউটনের সূত্র",           "নিউটন"),             # Newton laws generic
    ("প্রথম সূত্র",              "জড়তা"),             # 1st law = inertia
    ("দ্বিতীয় সূত্র",           "ভরবেগ"),            # 2nd law = momentum
    ("জড়তার সূত্র",             "জড়তা"),             # law of inertia
    ("মহাকর্ষ সূত্র",           "মহাকর্ষ"),           # gravitational law

    # ── Chapter 4: Work, Energy ─────────────────────────────────────────────
    ("কাজের সূত্র",             "কাজ"),               # work formula → ch=4
    ("গতিশক্তি",                "গতি শক্তি"),         # kinetic energy
    ("বিভবশক্তি",               "বিভব শক্তি"),        # potential energy
    ("যান্ত্রিক শক্তি",          "যান্ত্রিক"),         # mechanical energy
    ("শক্তির রূপান্তর",          "শক্তি রূপান্তর"),    # energy transformation

    # ── Chapter 5: Pressure ─────────────────────────────────────────────────
    ("বায়ুচাপ",                 "বায়ুমণ্ডলীয়"),      # atmospheric pressure → ch=5
    ("প্যাসকেলের সূত্র",        "প্যাসকেল"),          # Pascal law → ch=5
    ("আর্কিমিডিসের সূত্র",      "আর্কিমিডিস"),        # Archimedes → ch=5
    ("প্লবতা",                  "ভাসমান"),            # buoyancy → ch=5
    ("উচ্ছ্বাস বল",             "উচ্ছ্বাস"),          # upthrust

    # ── Chapter 6: Heat ─────────────────────────────────────────────────────
    ("তাপমাত্রা পরিবর্তন",     "তাপমাত্রার পরিবর্তন"), # temperature change → ch=6
    ("আপেক্ষিক তাপ",            "আপেক্ষিক তাপধারণ ক্ষমতা"), # specific heat
    ("তাপ পরিবাহিতা",           "তাপপরিবাহিতা"),      # thermal conductivity
    ("দৈর্ঘ্য সম্প্রসারণ",       "রৈখিকপ্রসারণসহগ"),   # length expansion → ch=6

    # ── Chapter 7: Waves ────────────────────────────────────────────────────
    ("তরঙ্গদৈর্ঘ্য",             "তরঙ্গের দৈর্ঘ্য"),   # wavelength → ch=7
    ("কম্পাঙ্ক",                "কম্পন"),             # frequency
    ("শব্দের বেগ",              "শব্দবেগ"),            # speed of sound
    ("প্রতিধ্বনি",              "প্রতিফলন"),           # echo = reflection of sound

    # ── Chapter 8: Light reflection ─────────────────────────────────────────
    ("আলোর প্রতিফলন",           "দর্পণ"),             # light reflection → mirror ch=8
    ("অবতল দর্পণ",              "দর্পণ"),             # concave mirror
    ("উত্তল দর্পণ",             "দর্পণ"),             # convex mirror
    ("দর্পণের সূত্র",           "ফোকাস"),             # mirror formula → focal length

    # ── Chapter 9: Refraction, TIR ──────────────────────────────────────────
    ("আলোর প্রতিসরণ",           "স্নেল"),             # refraction → Snell ch=9
    ("পূর্ণ অভ্যন্তরীণ",         "পূর্ণঅভ্যন্তরীণপ্রতিফলন"), # TIR
    ("ক্রান্তি কোণ",             "সংকট"),             # critical angle
    ("প্রতিসরাঙ্ক",             "প্রতিসরণ"),           # refractive index

    # ── Chapter 10: Static electricity ─────────────────────────────────────
    ("কুলম্বের সূত্র",           "কুলম্ব"),            # Coulomb law → ch=10
    ("বৈদ্যুতিক ক্ষেত্র",        "বৈদ্যুতিকক্ষেত্র"),  # electric field
    ("বৈদ্যুতিক বিভব",          "বিভব"),              # electric potential
    ("তড়িৎ বল",                "কুলম্ব"),            # electric force = Coulomb

    # ── Chapter 11: Current electricity ────────────────────────────────────
    ("ওমের সূত্র",              "ওহমের"),             # Ohm law (variant spelling)
    ("রোধ",                     "ওহম"),               # resistance → Ohm unit ch=11
    ("শ্রেণি বর্তনী",            "শ্রেণি"),            # series circuit
    ("শ্রেণি সংযোগ",             "শ্রেণি"),            # series connection
    ("বৈদ্যুতিক ক্ষমতা",        "ক্ষমতা"),            # electric power

    # ── Chapter 12: Electromagnetism ────────────────────────────────────────
    ("তড়িৎচুম্বকীয় আবেশ",      "তড়িৎচুম্বকীয়আবেশ"), # EM induction → ch=12
    ("বিদ্যুৎ উৎপাদন",          "জেনারেটর"),          # electricity generation
    ("AC DC",                   "পরিবর্তী"),           # AC/DC currents

    # ── Chapter 13: Modern Physics ──────────────────────────────────────────
    ("ফটোইলেকট্রিক",            "ফটোইলেকট্রিক"),      # photoelectric → ch=13
    ("কোয়ান্টাম",               "ফোটন"),              # quantum = photon
    ("তেজস্ক্রিয়তা",             "তেজস্ক্রিয়"),        # radioactivity → ch=13
    ("নিউক্লিয় শক্তি",          "নিউক্লিয়"),          # nuclear energy
]

# Vocabulary injections: synthetic terms inserted into specific chapter vocabs.
# Used when the book's OCR text lacks a canonical form but students query it.
# High frequency (10) ensures the term strongly pulls IDF toward the right chapter.
# These complement (not replace) existing vocab from real book text.
_BN_VOCAB_INJECTIONS: dict[int, dict[str, int]] = {
    2:  {"পড়ন্তবস্তু": 10, "মুক্তপতন": 10},                # Free fall → ch=2
    6:  {"রৈখিকপ্রসারণসহগ": 10, "তাপীয়প্রসারণ": 10},     # Thermal expansion → ch=6
    11: {"কির্শফের_সূত্র": 10, "ওহম": 10, "ওহমের": 10},   # Kirchhoff + Ohm → ch=11
    12: {"লেঞ্জ_সূত্র": 10},                                # Lenz → ch=12
    4:  {"ভরশক্তিসম্পর্ক": 10},                             # Mass-energy → ch=4
}

# Multi-word physics phrases that must be treated as single vocabulary units.
# When split by whitespace, individual words become ambiguous (e.g., "প্রতিফলন"
# is in ch=8 mirror AND ch=9 TIR). Joining them creates a unique compound term
# that only appears in the correct chapter's vocabulary.
_BN_COMPOUND_PHRASES: list[tuple[str, str]] = [
    ("পূর্ণ অভ্যন্তরীণ প্রতিফলন", "পূর্ণঅভ্যন্তরীণপ্রতিফলন"),   # Total Internal Reflection → ch=9
    ("পূর্ণ অভ্যন্তরীণ",           "পূর্ণঅভ্যন্তরীণ"),             # TIR partial phrase
    ("ফটো ইলেকট্রিক",             "ফটোইলেকট্রিক"),                # Photoelectric → ch=13
    ("তড়িৎচুম্বকীয় আবেশ",        "তড়িৎচুম্বকীয়আবেশ"),           # EM induction → ch=12
    ("বৈদ্যুতিক ক্ষেত্র",          "বৈদ্যুতিকক্ষেত্র"),             # Electric field → ch=10
    ("বৈদ্যুতিক বিভব",            "বৈদ্যুতিকবিভব"),               # Electric potential → ch=10
    ("তাপ পরিবাহিতা",             "তাপপরিবাহিতা"),                # Thermal conductivity → ch=6
    ("মহাকর্ষ বল",                "মহাকর্ষবল"),                   # Gravitational force → ch=3/5
    ("চৌম্বক ক্ষেত্র",             "চৌম্বকক্ষেত্র"),                 # Magnetic field → ch=12
    ("তাপীয় প্রসারণ",            "তাপীয়প্রসারণ"),                # Thermal expansion → ch=6
    ("প্রসারণ সহগ",               "রৈখিকপ্রসারণসহগ"),             # expansion coeff → ch=6
    ("ভর ও শক্তির",               "ভরশক্তিসম্পর্ক"),               # Mass-energy → ch=4
    ("ভর শক্তির",                 "ভরশক্তিসম্পর্ক"),               # Mass-energy → ch=4
]

def _bn_normalize(text: str) -> str:
    """Normalize OCR artifacts and join compound physics phrases for IDF matching."""
    # 1. OCR apostrophe artifacts (must run before compound phrases)
    for src, dst in _BN_OCR_NORMALIZE.items():
        text = text.replace(src, dst)
    # 2. Multi-word compound phrases → single compound token
    for phrase, joined in _BN_COMPOUND_PHRASES:
        text = text.replace(phrase, joined)
    return text


def _apply_query_synonyms(keywords: list[str], raw_query: str) -> list[str]:
    """Append book-equivalent synonym tokens to query keyword list.

    When a student uses a term not found in the book's OCR text
    (e.g. 'মুক্তপতন' instead of 'পড়ন্ত'), this function appends the
    book-equivalent token so IDF scoring can route to the right chapter.
    Also handles compound synonyms from raw query (pre-keyword extraction).
    """
    extra: list[str] = []
    # Check raw query for multi-word synonym phrases
    for phrase, synonym in _BN_QUERY_SYNONYMS:
        if phrase in raw_query:
            if synonym not in keywords:
                extra.append(synonym)
    # Check individual keywords for synonym mapping
    for phrase, synonym in _BN_QUERY_SYNONYMS:
        if " " not in phrase and phrase in keywords and synonym not in keywords:
            extra.append(synonym)
    return list(dict.fromkeys(keywords + extra))


def extract_keywords(query: str) -> list[str]:
    raw_query = query  # keep original for synonym matching
    query = _bn_normalize(query)          # fix OCR artifacts first
    clean = re.sub(r"[।?!,;:.\"'()\[\]]", " ", query)
    out: list[str] = []
    for w in clean.split():
        if len(w) < 3 or w in _BN_STOP:
            continue
        out.append(w)
        s = _strip_suffix(w)
        if s != w and len(s) >= 2:
            out.append(s)
    out = list(dict.fromkeys(out))
    # Append synonym tokens for terms absent from book OCR text
    out = _apply_query_synonyms(out, raw_query)
    return out


# ─────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────
class PhysicsTutorRetriever:
    """
    Intent-based retrieval from the physics tutor database.

    Args:
        child_df   : pandas DataFrame from child_chunks table (with 'vector' col)
        child_vecs : numpy float32 array of vectors (shape: [N, 768])
        parent_df  : pandas DataFrame from parent_chunks table
    """

    def __init__(
        self,
        child_df:   pd.DataFrame,
        child_vecs: np.ndarray,
        parent_df:  pd.DataFrame,
    ):
        self.cdf  = child_df.reset_index(drop=True)
        self.cvec = child_vecs
        self.pdf  = parent_df.reset_index(drop=True)
        self._pid = dict(zip(parent_df["chunk_id"], range(len(parent_df))))
        # Build chapter vocabulary for auto chapter-inference (GENERAL intent)
        self._ch_vocab, self._kw_ch_count = self._build_chapter_vocab()
        # FTS table reference — for hybrid dense+sparse retrieval
        # Opened lazily on first use to avoid startup cost
        self._fts_table = None
        try:
            _fts_db = lancedb.connect("data/lancedb")
            self._fts_table = _fts_db.open_table("parent_chunks")
        except Exception:
            pass   # FTS optional — dense-only fallback if unavailable

    # ── Chapter vocab (built at init time) ──────────────────────────────────

    _VOCAB_CONTENT_TYPES = frozenset({
        "theory", "example", "subsection", "niche_koro",
        "figure",     # physicist names often in figure captions (e.g. স্নেলের in ch=9)
        # shankhipto intentionally excluded — exercise questions contain cross-chapter
        # unit mentions (e.g. 'বায়ুমণ্ডলীয়' in ch=4 work exercise, 'ওহম' in ch=10,11 exercises)
        # that pollute IDF inference. Named laws are covered by theory + figure types.
    })

    def _build_chapter_vocab(self) -> tuple[dict, dict]:
        """
        Build per-chapter keyword vocabulary from theory chunks.

        Returns:
          ch_vocab     : {chapter_num: {keyword: freq}}
          kw_ch_count  : {keyword: n_chapters_containing_it}

        IDF intuition:
          - "স্নেলের" only in ch=9 → kw_ch_count=1 → IDF weight=1/1=1.0
          - "সূত্র" in all 13 chapters → IDF weight=1/13≈0.08
          Distinctive terms strongly signal the correct chapter.
        """
        ch_vocab: dict[int, dict[str, int]] = {}

        for _, row in self.pdf.iterrows():
            ch    = int(row.get("chapter_num", 0))
            ctype = str(row.get("content_type", ""))
            text  = str(row.get("text", ""))
            if not (1 <= ch <= 13):
                continue
            if ctype not in self._VOCAB_CONTENT_TYPES:
                continue
            for kw in extract_keywords(text):
                if ch not in ch_vocab:
                    ch_vocab[ch] = {}
                ch_vocab[ch][kw] = ch_vocab[ch].get(kw, 0) + 1

        # Inject synthetic vocabulary for terms absent from OCR text
        # (e.g. 'ওহম' doesn't appear verbatim; book uses 'ও\'ম' OCR form)
        for ch, injections in _BN_VOCAB_INJECTIONS.items():
            if ch not in ch_vocab:
                ch_vocab[ch] = {}
            for kw, freq in injections.items():
                # Only inject if not already present with higher real-text frequency
                if ch_vocab[ch].get(kw, 0) < freq:
                    ch_vocab[ch][kw] = freq

        # Count how many chapters each keyword appears in
        kw_ch_count: dict[str, int] = {}
        for ch_kws in ch_vocab.values():
            for kw in ch_kws:
                kw_ch_count[kw] = kw_ch_count.get(kw, 0) + 1

        return ch_vocab, kw_ch_count

    def _infer_chapter_from_keywords(
        self,
        keywords: list[str],
        min_score: float = 0.25,
        dominance: float = 1.4,   # lowered 1.6→1.4 for borderline cases
    ) -> Optional[int]:
        """
        IDF-weighted chapter scoring from query keywords.

        A keyword that appears in 1 chapter gets weight 1.0.
        A keyword in all 13 chapters gets weight ~0.08.
        Distinctive physics terms (স্নেলের, কুলম্ব, ওহম, etc.) strongly
        pull the score toward the correct chapter.

        Returns chapter int if one clearly dominates, else None.
        """
        if not keywords or not self._ch_vocab:
            return None

        total_ch = len(self._ch_vocab)
        scores: dict[int, float] = {}

        for ch, vocab in self._ch_vocab.items():
            sc = 0.0
            for kw in keywords:
                if kw in vocab:
                    # sqrt(freq) dampens extreme counts while rewarding the chapter
                    # where the term appears most (e.g. 'ভরবেগ' appears 6× in ch=3,
                    # only 1× in ch=5 → ch=3 scores 2.45× higher)
                    n = self._kw_ch_count.get(kw, total_ch)
                    sc += (vocab.get(kw, 0) ** 0.5) / n
            if sc > 0:
                scores[ch] = sc

        if not scores:
            return None

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_ch, best_sc = ranked[0]

        if best_sc < min_score:
            return None

        if len(ranked) >= 2:
            _, second_sc = ranked[1]
            if best_sc < second_sc * dominance:
                return None  # Too ambiguous

        return best_ch

    def _infer_chapters_multi(
        self,
        keywords: list[str],
        min_score: float = 0.25,  # raised from 0.10 — prevents noise from shankhipto unit mentions
        max_chapters: int = 5,    # increased 3→5 to catch tied cases (e.g. রৈখিক ch=6 tied with ch=2,3,5)
    ) -> list[int]:
        """
        When no single chapter dominates, return top 2-3 candidate chapters.
        Narrows vector search scope WITHOUT hard-filtering to one chapter.

        Example:
          "পূর্ণ অভ্যন্তরীণ প্রতিফলন" → ch=6,8,9 tied in IDF
          → narrow vector search to ch={6,8,9}
          → within those 3, ch=9 vector wins (has actual TIR content)
        """
        if not keywords or not self._ch_vocab:
            return []
        total_ch = len(self._ch_vocab)
        scores: dict[int, float] = {}
        for ch, vocab in self._ch_vocab.items():
            sc = sum(
                (vocab.get(kw, 0) ** 0.5) / self._kw_ch_count.get(kw, total_ch)
                for kw in keywords if kw in vocab
            )
            if sc >= min_score:
                scores[ch] = sc
        if not scores:
            return []
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_chapters]
        return [ch for ch, _ in top]

    def _score_all_chapters(self, keywords: list[str]) -> dict[int, float]:
        """
        IDF score for each chapter, normalized to [0, 1].
        Used as a SOFT BOOST on cosine similarity scores when the
        chapter classifier can't pick a clear winner.

        Even a 0.1 bonus on cosine scores can push correct-chapter
        results above attractor chunks from wrong chapters.
        """
        if not keywords or not self._ch_vocab:
            return {}
        total_ch = len(self._ch_vocab)
        scores: dict[int, float] = {}
        for ch, vocab in self._ch_vocab.items():
            sc = sum(
                (vocab.get(kw, 0) ** 0.5) / self._kw_ch_count.get(kw, total_ch)
                for kw in keywords if kw in vocab
            )
            if sc > 0:
                scores[ch] = sc
        if not scores:
            return {}
        max_sc = max(scores.values())
        return {ch: sc / max_sc for ch, sc in scores.items()}  # normalize 0-1


    # ── Public ──────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        q_vec: list[float],
        top_k: int = 8,
    ) -> tuple[list[dict], "QueryHints"]:
        hints  = extract_hints(query)
        intent = hints.intent

        dispatch = {
            QueryIntent.CHAPTER_OVERVIEW: self._chapter_overview,
            QueryIntent.SECTION_LOOKUP:   self._section_lookup,
            QueryIntent.PAGE_LOOKUP:      lambda h: self._page_lookup(h, q_vec, top_k),
            QueryIntent.NICHE_KORO:       self._niche_koro,
            QueryIntent.ONUSONDHAN:       self._onusondhan,
            QueryIntent.NOMUNA_PROSHNO:   self._nomuna_proshno,
            QueryIntent.SHANKHIPTO:       self._shankhipto,
            QueryIntent.CREATIVE_Q:       self._creative_q,
            QueryIntent.GENERAL:          lambda h: self._vector_search(query, q_vec, h, top_k),
        }
        results = dispatch[intent](hints)

        # Multi-intent: e.g. "নমুনা প্রশ্ন ২ এবং সংক্ষিপ্ত প্রশ্ন ৩"
        if intent == QueryIntent.NOMUNA_PROSHNO and "সংক্ষিপ্ত" in query:
            sh_results = self._shankhipto(hints)
            seen_ids = {r["chunk_id"] for r in results}
            for r in sh_results:
                if r["chunk_id"] not in seen_ids:
                    results.append(r)
                    seen_ids.add(r["chunk_id"])

        # Symmetric: SHANKHIPTO query also mentions নমুনা → also fetch MCQ
        if intent == QueryIntent.SHANKHIPTO and "নমুনা" in query:
            nom_results = self._nomuna_proshno(hints)
            seen_ids = {r["chunk_id"] for r in results}
            for r in nom_results:
                if r["chunk_id"] not in seen_ids:
                    results.append(r)
                    seen_ids.add(r["chunk_id"])

        return results, hints

    # ── Strategies ──────────────────────────────────────────

    def _chapter_overview(self, h: QueryHints) -> list[dict]:
        if h.chapter is None:
            return []
        rows = self.pdf[self.pdf["chapter_num"] == h.chapter].sort_values("book_page_start")
        seen: set[str] = set()
        out  = []
        for _, row in rows.iterrows():
            ctype = str(row.get("content_type", ""))
            if ctype == "figure":
                continue
            sec = str(row.get("section_num", ""))
            key = sec if sec else f"p{row['book_page_start']}"
            if key in seen:
                continue
            seen.add(key)
            out.append(self._prow(row, 1.0, "chapter_overview"))
        return out[:35]

    def _section_lookup(self, h: QueryHints) -> list[dict]:
        if not h.section:
            return []

        sec = h.section   # e.g. "5.3" (already extracted from "5.3.1")

        # ── Exact match (e.g. stored section_num = "5.3") ──────────────
        mask = self.pdf["section_num"].astype(str).str.startswith(sec)
        if h.chapter:
            mask &= (self.pdf["chapter_num"] == h.chapter)
        rows = self.pdf[mask].sort_values("book_page_start")
        results = [self._prow(r, 1.0, "section_lookup") for _, r in rows.iterrows()]

        # ── Sub-section extension ────────────────────────────────────────
        # When user explicitly asks for a sub-section (5.3.1 / 8.8.4 etc.),
        # directly search extracted_book.md for the heading — reliable even
        # when the DB chunks are mislabeled (OCR pipeline limitation).
        if h.is_subsection:
            # Reconstruct the original 3-part label from the query
            raw = h.raw
            m3 = re.search(r"\b(\d+\.\d+\.\d+)\b", raw)
            if m3:
                sub_label = m3.group(1)   # e.g. "8.8.4"
                bidx = get_book_index()
                sub_text = bidx.find_subsection(sub_label)
                if sub_text:
                    sub_result = {
                        "chunk_id":    f"subsec_{sub_label}",
                        "parent_id":   f"subsec_{sub_label}",
                        "text":        sub_text,
                        "content_type": "subsection",
                        "chapter":      h.chapter or 0,
                        "page_start":   0,
                        "page_end":     0,
                        "section":      sec,
                        "score":        1.0,
                        "source":       "book_index_subsection",
                    }
                    # Prepend so LLM sees it first
                    results.insert(0, sub_result)

            # Extend page range in DB for additional context
            all_pages = [r["page_start"] for r in results if r["page_start"] > 0]
            if all_pages:
                max_matched_pg = max(all_pages)
                extended_mask = (
                    (self.pdf["book_page_start"] > max_matched_pg) &
                    (self.pdf["book_page_start"] <= max_matched_pg + 6)
                )
                if h.chapter:
                    extended_mask &= (self.pdf["chapter_num"] == h.chapter)
                ext_rows = self.pdf[extended_mask].sort_values("book_page_start")
                ext_results = [self._prow(r, 0.85, "section_lookup_ext")
                               for _, r in ext_rows.iterrows()]
                seen_ids = {r["chunk_id"] for r in results}
                for r in ext_results:
                    if r["chunk_id"] not in seen_ids:
                        results.append(r)
                        seen_ids.add(r["chunk_id"])

        if not results and h.chapter:
            # Final fallback: keyword search within chapter
            kw_mask = (
                (self.pdf["chapter_num"] == h.chapter) &
                self.pdf["section_num"].astype(str).str.contains(
                    sec.split(".")[0], na=False
                )
            )
            rows2 = self.pdf[kw_mask].sort_values("book_page_start")
            results = [self._prow(r, 0.7, "section_fallback")
                       for _, r in rows2.iterrows()]

        return results[:15]


    def _page_lookup(self, h: QueryHints, q_vec, top_k) -> list[dict]:
        if h.page is None:
            return []
        pg   = h.page
        mask = (self.pdf["book_page_start"] <= pg) & (self.pdf["book_page_end"] >= pg - 1)
        if h.chapter:
            mask &= (self.pdf["chapter_num"] == h.chapter)
        rows = self.pdf[mask].sort_values("book_page_start")
        if len(rows) == 0:
            # relax by ±1 page
            mask2 = (self.pdf["book_page_start"] <= pg+1) & (self.pdf["book_page_end"] >= pg-2)
            rows  = self.pdf[mask2].sort_values("book_page_start")
        return [self._prow(r, 1.0, "page_lookup") for _, r in rows.iterrows()][:10]

    def _niche_koro(self, h: QueryHints) -> list[dict]:
        mask = (self.pdf["content_type"] == "niche_koro")
        if h.chapter:
            mask &= (self.pdf["chapter_num"] == h.chapter)
        if h.page:
            pg = h.page
            mask &= (self.pdf["book_page_start"] <= pg+2) & (self.pdf["book_page_end"] >= pg-2)
        rows = self.pdf[mask].sort_values("book_page_start")
        return [self._prow(r, 1.0, "niche_koro") for _, r in rows.iterrows()][:5]

    def _onusondhan(self, h: QueryHints) -> list[dict]:
        # Primary: BookIndex (has অনুসন্ধান for all chapters with exact text)
        try:
            from src.book_index import get_book_index
            bidx = get_book_index()
            entries = bidx.get_onusondhan(
                h.chapter or 0,
                page=h.page,
            )
            if entries:
                results = []
                for e in entries:
                    results.append({
                        "intent":       "onusondhan",
                        "score":        1.0,
                        "chunk_id":     f"book_onusondan_{e['id']}",
                        "parent_id":    f"book_onusondan_{e['id']}",
                        "chapter":      h.chapter or 0,
                        "section":      "",
                        "page_start":   e.get("book_page", 0),
                        "page_end":     e.get("book_page", 0) + 2,
                        "content_type": "onusondhan",
                        "text":         e["text"],
                        "child_text":   e["text"][:200],
                        "page_span":    f"{e.get('book_page',0)}–{e.get('book_page',0)+2}",
                    })
                return results[:5]
        except Exception:
            pass

        # Fallback: vector DB
        mask = (self.pdf["content_type"] == "onusondhan")
        if h.chapter:
            mask &= (self.pdf["chapter_num"] == h.chapter)
        if h.page:
            pg = h.page
            mask &= (self.pdf["book_page_start"] <= pg+1) & (self.pdf["book_page_end"] >= pg-1)
        rows = self.pdf[mask].sort_values("book_page_start")
        results = []
        for _, row in rows.iterrows():
            r = self._prow(row, 1.0, "onusondhan")
            r["page_span"] = f"{row['book_page_start']}–{row['book_page_end']}"
            results.append(r)
        return results[:5]

    def _nomuna_proshno(self, h: QueryHints) -> list[dict]:
        # Primary: BookIndex — has MCQ for ALL 13 chapters
        if h.chapter:
            try:
                from src.book_index import get_book_index
                bidx = get_book_index()
                text = bidx.get_mcq(h.chapter)
                if text:
                    result = [{
                        "intent":       "nomuna_proshno",
                        "score":        1.0,
                        "chunk_id":     f"book_mcq_ch{h.chapter}",
                        "parent_id":    f"book_mcq_ch{h.chapter}",
                        "chapter":      h.chapter,
                        "section":      "",
                        "page_start":   0,
                        "page_end":     0,
                        "content_type": "mcq_nomuna",
                        "text":         text,
                        "child_text":   text[:200],
                    }]
                    # Filter by question number if specified
                    if h.question_num:
                        qn   = h.question_num
                        bn_q = "০১২৩৪৫৬৭৮৯"[qn] if qn <= 9 else str(qn)
                        # Return full text but highlight the question number
                        if f"{qn}." in text or f"{bn_q}." in text:
                            return result
                    return result
            except Exception:
                pass

        # Fallback: vector DB
        mask = self.pdf["content_type"].isin(["mcq", "mcq_nomuna"])
        if h.chapter:
            mask &= (self.pdf["chapter_num"] == h.chapter)
        rows = self.pdf[mask].sort_values("book_page_start")
        if len(rows) == 0 and h.chapter:
            mask2 = (
                (self.pdf["chapter_num"] == h.chapter) &
                self.pdf["text"].str.contains("নমুনা প্রশ্ন|বহুনির্বাচনি", na=False)
            )
            rows = self.pdf[mask2].sort_values("book_page_start")
        return [self._prow(r, 1.0, "nomuna_proshno") for _, r in rows.iterrows()][:12]

    def _shankhipto(self, h: QueryHints) -> list[dict]:
        # Primary: BookIndex — has সংক্ষিপ্ত for ALL 13 chapters
        if h.chapter:
            try:
                from src.book_index import get_book_index
                text = get_book_index().get_shankhipto(h.chapter)
                if text:
                    return [{
                        "intent":       "shankhipto",
                        "score":        1.0,
                        "chunk_id":     f"book_shankhipto_ch{h.chapter}",
                        "parent_id":    f"book_shankhipto_ch{h.chapter}",
                        "chapter":      h.chapter,
                        "section":      "",
                        "page_start":   0,
                        "page_end":     0,
                        "content_type": "shankhipto",
                        "text":         text,
                        "child_text":   text[:200],
                    }]
            except Exception:
                pass

        # Fallback: vector DB
        mask = (self.pdf["content_type"] == "shankhipto")
        if h.chapter:
            mask &= (self.pdf["chapter_num"] == h.chapter)
        rows = self.pdf[mask].sort_values("book_page_start")
        if len(rows) == 0 and h.chapter:
            mask2 = (
                (self.pdf["chapter_num"] == h.chapter) &
                self.pdf["text"].str.contains("সংক্ষিপ্ত", na=False)
            )
            rows = self.pdf[mask2].sort_values("book_page_start")
        return [self._prow(r, 1.0, "shankhipto") for _, r in rows.iterrows()][:5]

    def _creative_q(self, h: QueryHints) -> list[dict]:
        # Primary: BookIndex — has সৃজনশীল for ALL 13 chapters
        if h.chapter:
            try:
                from src.book_index import get_book_index
                text = get_book_index().get_creative(h.chapter)
                if text:
                    return [{
                        "intent":       "creative_question",
                        "score":        1.0,
                        "chunk_id":     f"book_creative_ch{h.chapter}",
                        "parent_id":    f"book_creative_ch{h.chapter}",
                        "chapter":      h.chapter,
                        "section":      "",
                        "page_start":   0,
                        "page_end":     0,
                        "content_type": "creative_question",
                        "text":         text,
                        "child_text":   text[:200],
                    }]
            except Exception:
                pass

        # Fallback: vector DB
        mask = (self.pdf["content_type"] == "creative_question")
        if h.chapter:
            mask &= (self.pdf["chapter_num"] == h.chapter)
        rows = self.pdf[mask].sort_values("book_page_start")
        return [self._prow(r, 1.0, "creative_question") for _, r in rows.iterrows()][:5]

    # Content types that are exercise sections — should NOT appear for concept queries
    _EXERCISE_CTYPES = frozenset({
        "mcq_nomuna", "creative_question", "shankhipto", "onusondhan", "niche_koro",
    })
    # Query keywords that indicate the student IS asking for exercises
    _EXERCISE_QUERY_KW = frozenset({
        "প্রশ্ন", "MCQ", "mcq", "নমুনা", "সৃজনশীল", "সংক্ষিপ্ত", "বহুনির্বাচনি",
        "exercise", "quiz", "কুইজ", "অনুশীলন",
    })

    def _vector_search(
        self,
        query: str,
        q_vec: list[float],
        h:     QueryHints,
        top_k: int,
    ) -> list[dict]:
        keywords = extract_keywords(query)

        # Detect if student is asking for exercise content
        is_exercise_query = any(kw in query for kw in self._EXERCISE_QUERY_KW)

        # If no chapter hint, try vocabulary-based inference (hard filter);
        # Always compute soft IDF scores for boosting (helps even when no winner).
        _multi_narrow: list[int] = []   # candidate chapters when single fails
        if not h.chapter and not h.section and not h.page:
            inferred = self._infer_chapter_from_keywords(keywords)
            if inferred:
                h.chapter = inferred
            else:
                # No single winner → try multi-chapter narrowing.
                # E.g. "পূর্ণ অভ্যন্তরীণ প্রতিফলন" → ch=6,8,9 tied in IDF
                # → narrow to {6,8,9}, then vector search picks ch=9 winner.
                _multi_narrow = self._infer_chapters_multi(keywords)
        idf_ch_scores = (
            self._score_all_chapters(keywords)
            if not h.chapter   # only boost when doing full-corpus search
            else {}
        )

        # Pre-filter child_df
        mask = pd.Series([True] * len(self.cdf), index=self.cdf.index)
        if h.chapter:
            mask &= (self.cdf["chapter_num"] == h.chapter)
        elif _multi_narrow:
            # Multi-chapter narrowing: restrict to candidate chapters
            mask &= self.cdf["chapter_num"].isin(_multi_narrow)
        if h.section:
            mask &= self.cdf["section_num"].astype(str).str.startswith(h.section)
        if h.page:
            pg = h.page
            mask &= (self.cdf["book_page_start"] <= pg) & (self.cdf["book_page_end"] >= pg-2)

        # For concept/formula queries: exclude exercise content types so that
        # theory sections surface above MCQ/creative/shankhipto sections.
        if not is_exercise_query:
            mask &= ~self.cdf["content_type"].isin(self._EXERCISE_CTYPES)

        sub_df   = self.cdf[mask].reset_index(drop=True)
        sub_vecs = self.cvec[mask.values]

        # Relax exercise filter if too few results
        if len(sub_df) < 5 and h.chapter:
            mask2    = self.cdf["chapter_num"] == h.chapter
            sub_df   = self.cdf[mask2].reset_index(drop=True)
            sub_vecs = self.cvec[mask2.values]

        if len(sub_df) == 0:
            sub_df   = self.cdf
            sub_vecs = self.cvec

        q_arr = np.array(q_vec, dtype=np.float32)
        CK    = min(25, len(sub_df))
        dots  = sub_vecs @ q_arr
        norms = np.linalg.norm(sub_vecs, axis=1) * (np.linalg.norm(q_arr) + 1e-10)
        sims  = dots / (norms + 1e-10)
        cidx  = np.argsort(-sims)[:CK]

        # parent text lookup for keyword boost
        plookup = dict(zip(self.pdf["chunk_id"], self.pdf["text"].astype(str)))

        # Build keyword forms: include stripped variants for text matching.
        # e.g. query keyword "সমান্তরালে" (suffix তে) → also check "সমান্তরাল" in chunk text.
        _kw_forms: set[str] = set(keywords)
        for kw in keywords:
            s = _strip_suffix(kw)
            if s != kw and len(s) >= 3:
                _kw_forms.add(s)

        scored = []
        for idx in cidx:
            base = float(sims[idx])
            row  = sub_df.iloc[idx]
            ct   = str(row["text"])
            pid  = str(row["parent_id"])
            pt   = plookup.get(pid, "")
            cb   = sum(0.15 for kw in _kw_forms if kw in ct)
            pb   = sum(0.10 for kw in _kw_forms if kw in pt)
            # IDF soft boost: nudges results from likely chapters upward
            # (max +0.12 so it influences ranking without overriding cosine)
            ch   = int(row["chapter_num"])
            idf_boost = idf_ch_scores.get(ch, 0) * 0.12
            scored.append((idx, base + cb + pb + idf_boost))

        scored.sort(key=lambda x: x[1], reverse=True)

        seen: set[str] = set()
        results = []
        ch_count: dict[int, int] = {}          # for diversity cap
        # Diversity cap: 2 per chapter when searching all 13 chapters,
        # more generous when already narrowed by h.chapter or multi_narrow.
        _is_focused = bool(h.chapter or _multi_narrow)
        MAX_PER_CH = top_k if _is_focused else 2

        for idx, score in scored:
            row = sub_df.iloc[idx]
            pid = str(row["parent_id"])
            ch  = int(row["chapter_num"])

            if pid in seen:
                continue

            # Chapter diversity: prevent one chapter dominating full-corpus results
            if not _is_focused and ch_count.get(ch, 0) >= MAX_PER_CH:
                continue

            # Skip obvious OCR garbage chunks (formula-only text, no Bengali)
            child_text = str(row["text"])
            bn_chars = sum(1 for c in child_text if '\u0980' <= c <= '\u09FF')
            if bn_chars < 10 and len(child_text) < 80:
                continue   # nearly no Bengali — likely OCR garbage

            seen.add(pid)
            ch_count[ch] = ch_count.get(ch, 0) + 1   # track results per chapter
            pidx = self._pid.get(pid)
            prow = self.pdf.iloc[pidx] if pidx is not None else None
            ptext = str(prow["text"]) if prow is not None else str(row["text"])
            results.append({
                "intent":       "general",
                "score":        score,
                "chunk_id":     str(row["chunk_id"]),
                "parent_id":    pid,
                "chapter":      int(row["chapter_num"]),
                "section":      str(row["section_num"]),
                "page_start":   int(row["book_page_start"]),
                "page_end":     int(row["book_page_end"]),
                "content_type": str(row["content_type"]),
                "text":         ptext,
                "child_text":   str(row["text"]),
            })
            if len(results) >= top_k:
                break

        # ── FTS Hybrid Merge ─────────────────────────────────────────────────
        # Run FTS (keyword) search alongside dense and merge results.
        # FTS finds exact keyword matches that cosine similarity misses.
        # Only inject FTS chunks from the correct chapter (if known) to
        # avoid cross-chapter noise.
        if self._fts_table is not None and query.strip():
            try:
                _fts_limit = 8
                _fts_raw = (
                    self._fts_table
                    .search(query, query_type="fts")
                    .limit(_fts_limit)
                    .to_pandas()
                )
                _seen_in_dense = set(r["parent_id"] for r in results)
                _fts_added = 0
                for _, frow in _fts_raw.iterrows():
                    fpid  = str(frow["chunk_id"])
                    fch   = int(frow["chapter_num"])
                    # Filter: if chapter known, only accept from that chapter
                    if h.chapter and fch != h.chapter:
                        continue
                    if fpid in _seen_in_dense:
                        continue
                    ftext = str(frow["text"])
                    # Skip OCR garbage
                    bn_ch = sum(1 for c in ftext if "\u0980" <= c <= "\u09FF")
                    if bn_ch < 10:
                        continue
                    _seen_in_dense.add(fpid)
                    results.append({
                        "intent":       "fts_hybrid",
                        "score":        0.6,   # between dense (~0.7+) and rescue (0.5)
                        "chunk_id":     fpid,
                        "parent_id":    fpid,
                        "chapter":      fch,
                        "section":      str(frow.get("section_num", "")),
                        "page_start":   int(frow["book_page_start"]),
                        "page_end":     int(frow["book_page_end"]),
                        "content_type": str(frow.get("content_type", "")),
                        "text":         ftext,
                        "child_text":   ftext[:200],
                    })
                    _fts_added += 1
            except Exception:
                pass   # FTS failure is non-fatal — dense results always returned

        # ── Keyword Rescue Search ─────────────────────────────────────────────
        # AFTER vector search: check if any query keywords are completely absent
        # from all returned chunk texts. If so, do a substring search in the
        # chapter's parent_chunks and inject matching chunks.
        # Generic dense+sparse hybrid — no chapter hardcoding.
        # Only activates when chapter is known (avoids noisy cross-chapter rescue).
        if h.chapter and keywords:
            returned_text = " ".join(r.get("text", "") for r in results)
            # Use original keywords (not suffix-stripped) for the missing check.
            # Suffix forms can false-match: e.g. 'স্ক্র' matches 'স্কেল',
            # making the real keyword 'স্ক্রু' appear present when it isn't.
            missing_kws = [
                kw for kw in keywords
                if len(kw) >= 3 and kw not in returned_text
            ]
            if missing_kws:
                ch_mask = self.pdf["chapter_num"] == h.chapter
                ch_pdf  = self.pdf[ch_mask]
                # Prefer theory/subsection chunks; skip learning_objectives.
                _RESCUE_PRIORITY = {
                    "theory": 0, "subsection": 1, "niche_koro": 2,
                    "shankhipto": 3, "creative_question": 4,
                    "mcq_nomuna": 5, "onusondhan": 6, "learning_objectives": 99,
                }
                rescue_seen: set[str] = set(r["parent_id"] for r in results)
                rescue_added = 0
                for kw in missing_kws:
                    if rescue_added >= 3:
                        break
                    matching = ch_pdf[ch_pdf["text"].str.contains(kw, na=False, regex=False)]
                    matching = matching.copy()
                    matching["_prio"] = matching["content_type"].map(
                        lambda ct: _RESCUE_PRIORITY.get(str(ct), 50)
                    )
                    matching = matching.sort_values("_prio")
                    for _, mrow in matching.iterrows():
                        pid = str(mrow["chunk_id"])
                        if pid in rescue_seen:
                            continue
                        mtext = str(mrow["text"])
                        bn_chars = sum(1 for c in mtext if '\u0980' <= c <= '\u09FF')
                        if bn_chars < 10:
                            continue
                        rescue_seen.add(pid)
                        results.append({
                            "intent":       "keyword_rescue",
                            "score":        0.5,
                            "chunk_id":     pid,
                            "parent_id":    pid,
                            "chapter":      int(mrow["chapter_num"]),
                            "section":      str(mrow.get("section_num", "")),
                            "page_start":   int(mrow["book_page_start"]),
                            "page_end":     int(mrow["book_page_end"]),
                            "content_type": str(mrow.get("content_type", "")),
                            "text":         mtext,
                            "child_text":   mtext[:200],
                        })
                        rescue_added += 1
                        break

        return results

    # ── Helper ──────────────────────────────────────────────

    def _prow(self, row, score: float, intent: str) -> dict:
        return {
            "intent":       intent,
            "score":        score,
            "chunk_id":     str(row.get("chunk_id", "")),
            "parent_id":    str(row.get("chunk_id", "")),
            "chapter":      int(row["chapter_num"]),
            "section":      str(row.get("section_num", "")),
            "page_start":   int(row["book_page_start"]),
            "page_end":     int(row["book_page_end"]),
            "content_type": str(row.get("content_type", "")),
            "text":         str(row.get("text", "")),
            "child_text":   str(row.get("text", ""))[:200],
        }


# ─────────────────────────────────────────────────────────
# Context builder for LLM
# ─────────────────────────────────────────────────────────
def build_context(results: list[dict], hints: "QueryHints") -> str:
    if not results:
        return "(কোনো relevant content পাওয়া যায়নি।)"

    parts = []
    seen: set[str] = set()

    for r in results:
        text = r.get("text", "").strip()
        if not text or text[:80] in seen:
            continue
        seen.add(text[:80])

        ch    = r.get("chapter", "?")
        sec   = r.get("section", "")
        ps    = r.get("page_start", "?")
        pe    = r.get("page_end", "")
        ctype = r.get("content_type", "")

        pg_str  = f"পৃষ্ঠা {ps}" if (not pe or pe == ps) else f"পৃষ্ঠা {ps}–{pe}"
        sec_str = f" | ধারা {sec}" if sec else ""
        header  = f"[অধ্যায় {ch}{sec_str} | {pg_str} | {ctype}]"

        if r.get("page_span"):
            header += f" ⟨অনুসন্ধান পৃষ্ঠা {r['page_span']}⟩"

        parts.append(f"{header}\n{text[:5000]}")


    text_context = "\n\n---\n\n".join(parts)

    # ── OCR Normalization ─────────────────────────────────────
    # Normalize OCR artifacts in the output text so the LLM sees clean Bengali.
    # e.g. "ও'ম" → "ওহম" (OCR apostrophe artifact in physics book).
    # Uses the same _BN_OCR_NORMALIZE table that is applied to query keywords.
    for ocr_form, clean_form in _BN_OCR_NORMALIZE.items():
        text_context = text_context.replace(ocr_form, clean_form)

    # ── Figure injection ─────────────────────────────────────
    # Results এর page range থেকে relevant figure descriptions inject করো
    try:
        from src.figure_context import inject_figures_into_context
        text_context = inject_figures_into_context(
            text_context,
            results,
            max_total_figs=4,
        )
    except Exception:
        pass   # figure injection optional — never break retrieval

    return text_context


# ─────────────────────────────────────────────────────────
# Singleton factory
# ─────────────────────────────────────────────────────────
_retriever_instance: Optional["PhysicsTutorRetriever"] = None

def get_retriever() -> "PhysicsTutorRetriever":
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = PhysicsTutorRetriever()
    return _retriever_instance
