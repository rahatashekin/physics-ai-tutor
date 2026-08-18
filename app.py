"""
Physics Tutor Agent — Main Streamlit App
==========================================
বাংলাদেশের Class 9-10 পদার্থবিজ্ঞান বইয়ের উপর ভিত্তি করে
adaptive RAG chatbot — powered by Gemini 2.5 Pro (Vertex AI)
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

import numpy as np
import streamlit as st
import vertexai
from dotenv import load_dotenv
from google import genai as google_genai
from vertexai.generative_models import Content, GenerativeModel, Part

sys.path.insert(0, str(Path(__file__).parent))

from agent.prompts import build_system_prompt, QUIZ_PROMPT_SUFFIX, HINT_PROMPT_SUFFIX
from agent.tools import (
    get_chapter_formulas,
    analyze_user_image,
    generate_physics_diagram,
    detect_diagram_type,
)
from memory.user_profile import UserProfile
from utils.latex_formatter import format_latex_for_streamlit
from src.retrieval import PhysicsTutorRetriever, build_context, extract_hints
from src.figure_context import inject_figures_into_context, get_figures_for_pages

load_dotenv()

# ----------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Physics Tutor — Class 9-10",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------
# Custom CSS — loaded from static/styles.css (docling pattern: separate concerns)
# ----------------------------------------------------------------
def load_css():
    """Load CSS from static/styles.css — mirrors docling cookbook pattern."""
    css_path = Path(__file__).parent / 'static' / 'styles.css'
    if css_path.exists():
        st.markdown(f'<style>{css_path.read_text()}</style>', unsafe_allow_html=True)

load_css()


# ----------------------------------------------------------------
# Initialize Session State
# ----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = UserProfile.new_session()
if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = False
if "hint_mode" not in st.session_state:
    st.session_state.hint_mode = False
if "gemini_model" not in st.session_state:
    vertexai.init(
        project=os.getenv("GCP_PROJECT", "project-3e580a5b-256c-4c6d-a0a"),
        location=os.getenv("GCP_LOCATION", "us-central1"),
    )
    st.session_state.gemini_model = GenerativeModel(
        model_name="gemini-2.5-pro",
    )

if "retriever" not in st.session_state:
    import lancedb
    _db        = lancedb.connect("data/lancedb")
    _child_df  = _db.open_table("child_chunks").to_pandas()
    _child_vec = np.array(_child_df["vector"].tolist(), dtype=np.float32)
    _parent_df = _db.open_table("parent_chunks").to_pandas()
    st.session_state.retriever    = PhysicsTutorRetriever(_child_df, _child_vec, _parent_df)
    st.session_state.embed_client = google_genai.Client(
        vertexai=True,
        project=os.getenv("GCP_PROJECT", "project-3e580a5b-256c-4c6d-a0a"),
        location=os.getenv("GCP_LOCATION", "us-central1"),
    )

gemini_model: GenerativeModel = st.session_state.gemini_model
retriever: PhysicsTutorRetriever = st.session_state.retriever
embed_client = st.session_state.embed_client
profile: UserProfile = st.session_state.profile

LEVEL_EMOJI = {1: "🌱", 2: "📗", 3: "📘", 4: "📙", 5: "🏆"}
STYLE_LABELS = {
    "auto": "🤖 স্বয়ংক্রিয়",
    "analogy": "🔗 উপমা",
    "formula": "📐 সূত্র",
    "example": "💡 উদাহরণ",
    "step_by_step": "🪜 ধাপে ধাপে",
}

# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------
with st.sidebar:
    # ── Brand ─────────────────────────────────────────────
    st.markdown("""<div style="padding: 0.4rem 0 0.2rem 0;">
        <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.01em;">⚛️ Physics Tutor</div>
        <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">Class 9-10 · NCTB Bangladesh</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # ── New Chat (prominent CTA) ───────────────────────────
    if st.button("✦ নতুন কথোপকথন", use_container_width=True, key="new_chat_btn"):
        st.session_state.messages = []
        st.session_state.profile = UserProfile.new_session()
        st.rerun()

    st.divider()

    # ── Understanding Level ────────────────────────────────
    level = profile.understanding_level
    level_names = {1: "শিক্ষানবিশ", 2: "শিখছি", 3: "মোটামুটি", 4: "ভালো", 5: "বিশেষজ্ঞ"}
    st.markdown(f"""<div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.07em;
        color:#94a3b8; font-weight:600; margin-bottom:6px;">বোঝার স্তর</div>""",
        unsafe_allow_html=True)
    st.markdown(f"""<div class="level-badge">{LEVEL_EMOJI[level]} {level_names[level]} — {level}/5</div>""",
        unsafe_allow_html=True)
    st.progress((level - 1) / 4)

    st.divider()

    # ── Settings (collapsed by default) ───────────────────
    with st.expander("⚙️  সেটিংস", expanded=False):
        # Explanation style
        st.markdown("**ব্যাখ্যার ধরন**")
        style_choice = st.selectbox(
            "style",
            options=list(STYLE_LABELS.keys()),
            format_func=lambda x: STYLE_LABELS[x],
            index=list(STYLE_LABELS.keys()).index(profile.preferred_style),
            label_visibility="collapsed",
            key="style_select",
        )
        if style_choice != profile.preferred_style:
            profile.preferred_style = style_choice

        st.markdown("**মোড**")
        col1, col2 = st.columns(2)
        with col1:
            quiz_toggle = st.toggle("Quiz", value=st.session_state.quiz_mode, key="quiz_t")
            st.session_state.quiz_mode = quiz_toggle
        with col2:
            hint_toggle = st.toggle("Hint", value=st.session_state.hint_mode, key="hint_t")
            st.session_state.hint_mode = hint_toggle

        # Stats
        total_q = profile.total_questions
        understood = profile.total_understood
        if total_q > 0:
            rate = understood / total_q * 100
            st.markdown("**পরিসংখ্যান**")
            c1, c2 = st.columns(2)
            c1.metric("প্রশ্ন", total_q)
            c2.metric("বোঝার হার", f"{rate:.0f}%")

        # Formula quick-reference
        st.markdown("**সূত্র খোঁজো**")
        formula_query = st.text_input("অধ্যায়/বিষয়:", placeholder="যেমন: গতি, বিদ্যুৎ...", key="fq")
        if formula_query and st.button("সূত্র দেখাও", key="fq_btn"):
            with st.spinner("খুঁজছি..."):
                formulas = get_chapter_formulas(formula_query)
            st.markdown(format_latex_for_streamlit(formulas))

    # Active mode indicators (outside expander — always visible)
    if st.session_state.get("quiz_mode"):
        st.markdown("""
        <div style="background:rgba(234,179,8,0.15); border-left:3px solid #eab308;
             border-radius:0 8px 8px 0; padding:6px 10px; font-size:0.78rem;
             color:#fef9c3; margin-top:6px;">📝 Quiz Mode চালু</div>""",
            unsafe_allow_html=True)
    if st.session_state.get("hint_mode"):
        st.markdown("""
        <div style="background:rgba(59,130,246,0.15); border-left:3px solid #3b82f6;
             border-radius:0 8px 8px 0; padding:6px 10px; font-size:0.78rem;
             color:#bfdbfe; margin-top:6px;">💡 Hint Mode চালু</div>""",
            unsafe_allow_html=True)

    # Footer — simple, no absolute positioning
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.caption("Powered by Gemini 2.5 Pro ✨")



# ── Chat history — docling/5-chat.py pattern ─────────────────
# Native st.chat_message() ensures LaTeX ($...$), markdown (###, **bold**)
# render consistently for BOTH streaming and stored messages.
# Mirrors: ai-cookbook/knowledge/docling/5-chat.py lines 104-107
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user" and message.get("image_bytes"):
            st.image(message["image_bytes"], width=240)
        if message["role"] == "assistant" and message.get("diagram"):
            st.image(message["diagram"])
        st.markdown(message["content"])  # LaTeX ✅  markdown ✅  consistent ✅

# ── Welcome message — docling/5-chat.py pattern ──────────────
# Mirrors: ai-cookbook/knowledge/docling/5-chat.py structure
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""আস্সালামু আলাইকুম! আমি তোমার পদার্থবিজ্ঞান টিউটর। 🎉

আমি তোমাকে Class 9-10 Physics বুঝতে সাহায্য করব।

**আমি যা করতে পারি:**

📖 বইয়ের যেকোনো বিষয় ব্যাখ্যা করতে পারি
📐 সূত্র সহজ ভাষায় বোঝাতে পারি
📊 গ্রাফ ও diagram তৈরি করতে পারি
📷 তোমার সমস্যার ছবি দেখে সাহায্য করতে পারি
🧪 Quiz দিতে পারি, hint দিতে পারি

**কীভাবে জিজ্ঞেস করবে?**
বাংলায়, English এ, বা Banglish এ — যেভাবে ইচ্ছা! নিচে প্রশ্ন লেখো 👇""")

# ── Hidden file uploader (functional, CSS hides the UI) ─────────
_uploaded_widget = st.file_uploader(
    "ছবি",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="collapsed",
    key="img_uploader",
)
uploaded_image = st.session_state.get("img_uploader", None)

# Spacer so last message doesn't hide behind input bar
st.markdown("<div style='height:75px'></div>", unsafe_allow_html=True)

# ── "+" attach button — moved to document.body via JS ──────────
# KEY FIX: Streamlit's container has CSS transform/overflow that breaks
# position:fixed. We move the button to document.body so position:fixed
# is truly viewport-relative.
st.markdown("""
<style>
#g-attach:hover {
    background: #e0e7ff !important;
    border-color: #6366f1 !important;
    color: #4f46e5 !important;
}
#g-img-indicator {
    position: fixed;
    display: none;
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 20px;
    padding: 4px 12px 4px 8px;
    font-size: 0.78rem;
    color: #166534;
    z-index: 2147483647;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
}
#g-img-x { cursor:pointer; opacity:0.7; margin-left:2px; }
#g-img-x:hover { opacity:1; }
</style>

<button id="g-attach"
    data-gid="g-attach"
    title="ছবি attach করো"
    onclick="gTriggerFile()"
    style="
        position:fixed;
        bottom:18px; left:270px;
        width:34px; height:34px;
        border-radius:50%;
        background:#f1f5f9;
        border:1.5px solid #e2e8f0;
        color:#475569;
        cursor:pointer;
        display:flex;
        align-items:center;
        justify-content:center;
        z-index:2147483647;
        padding:0;
        box-shadow:0 1px 4px rgba(0,0,0,0.12);
        transition:background .15s,border-color .15s,color .15s;
    ">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2.5"
         stroke-linecap="round" stroke-linejoin="round">
        <line x1="12" y1="5" x2="12" y2="19"/>
        <line x1="5" y1="12" x2="19" y2="12"/>
    </svg>
</button>

<div id="g-img-indicator" data-gid="g-img-indicator">
    📷 <span id="g-img-name">ছবি attached</span>
    <span id="g-img-x" onclick="gClearImg()">✕</span>
</div>

<div id="g-hint" data-gid="g-hint" style="
    position:fixed;
    bottom:3px; left:270px; right:24px;
    text-align:center;
    font-size:0.68rem;
    color:#94a3b8;
    font-family:'Hind Siliguri','Inter',sans-serif;
    pointer-events:none;
    z-index:2147483646;
">💡 ছবির প্রশ্নের জন্য <strong style="color:#6366f1">+</strong> click করো</div>

<script>
(function() {
    var GIDS = ['g-attach', 'g-hint', 'g-img-indicator'];

    function setup() {
        // 1. Remove old body-level copies from previous Streamlit reruns
        GIDS.forEach(function(gid) {
            document.querySelectorAll('[data-gid="' + gid + '"]').forEach(function(el) {
                if (el.parentNode === document.body) {
                    el.parentNode.removeChild(el);
                }
            });
        });

        // 2. Move current elements (in Streamlit container) to document.body
        //    This fixes position:fixed inside Streamlit's transformed container
        GIDS.forEach(function(gid) {
            var el = document.querySelector('[data-gid="' + gid + '"]');
            if (el) document.body.appendChild(el);
        });

        // 3. Position precisely inside the chat input bar
        posBtn();
    }

    function posBtn() {
        var sb  = document.querySelector('[data-testid="stSidebar"]');
        var inp = document.querySelector('[data-testid="stChatInput"]');
        var ta  = document.querySelector('[data-testid="stChatInput"] textarea');
        var btn = document.querySelector('body > [data-gid="g-attach"]');
        var hint= document.querySelector('body > [data-gid="g-hint"]');
        var ind = document.querySelector('body > [data-gid="g-img-indicator"]');
        if (!inp || !btn) return;

        var ir  = inp.getBoundingClientRect();
        var sbR = sb ? sb.getBoundingClientRect().right : 0;

        // Use textarea for accurate vertical center (stChatInput extends to bottom edge)
        var centerY = ta
            ? ta.getBoundingClientRect().top + ta.getBoundingClientRect().height / 2
            : ir.top + ir.height / 2;

        // bottom: distance from viewport bottom to button's bottom edge
        var bBot = Math.round(window.innerHeight - centerY - 17); // 17 = half of 34px button
        btn.style.bottom = Math.max(bBot, 8) + 'px';
        btn.style.left   = (ir.left + 10) + 'px';

        // Hint: always at very bottom of viewport
        if (hint) {
            hint.style.bottom = '2px';
            hint.style.left   = (sbR + 16) + 'px';
            hint.style.right  = '24px';
        }
        // Indicator: just above the input box top
        if (ind) {
            ind.style.bottom = (window.innerHeight - ir.top + 6) + 'px';
            ind.style.left   = (ir.left + 10) + 'px';
        }
    }

    // Run setup immediately and on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setup);
    } else {
        setup();
    }
    setTimeout(setup, 200);   // fallback for Streamlit late render
    setInterval(posBtn, 300); // keep position updated on sidebar resize

    // Trigger the hidden Streamlit file uploader
    window.gTriggerFile = function() {
        var fi = document.querySelector('[data-testid="stFileUploader"] input[type="file"]')
               || document.querySelector('input[type="file"]');
        if (!fi) { alert('ফাইল input পাওয়া যায়নি'); return; }
        fi.click();
        fi.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                var nm = this.files[0].name;
                var nameEl = document.getElementById('g-img-name');
                var ind    = document.querySelector('body > [data-gid="g-img-indicator"]');
                if (nameEl) nameEl.textContent = nm.length > 18 ? nm.slice(0,18)+'...' : nm;
                if (ind)    ind.style.display = 'flex';
            }
        }, { once: true });
    };

    window.gClearImg = function() {
        var ind = document.querySelector('body > [data-gid="g-img-indicator"]');
        if (ind) ind.style.display = 'none';
        var fi = document.querySelector('[data-testid="stFileUploader"] input[type="file"]')
               || document.querySelector('input[type="file"]');
        if (fi) try { fi.value=''; fi.dispatchEvent(new Event('change',{bubbles:true})); } catch(e){}
    };
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Chat Input & Response — native widget, fully functional
# ─────────────────────────────────────────────────────────────────

if prompt := st.chat_input("পদার্থবিজ্ঞান নিয়ে প্রশ্ন করো..."):

    # --- Process uploaded image if any ---
    image_bytes = None
    image_analysis = ""
    if uploaded_image:
        image_bytes = uploaded_image.read()
        with st.spinner("ছবি বিশ্লেষণ করছি..."):
            image_analysis = analyze_user_image(image_bytes, prompt, gemini_model)

    # --- Display user message — docling/5-chat.py pattern (lines 169-171) ---
    with st.chat_message("user"):
        if image_bytes:
            st.image(image_bytes, width=240)
        st.markdown(prompt)

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "image_bytes": image_bytes,
    })


    # --- Detect comprehension signal ---
    signal = UserProfile.detect_signal_from_message(prompt)
    if signal:
        profile.record_signal(signal)
        profile.save()

    # --- Retrieve context using new intent-based retrieval ---
    with st.status("📚 বই থেকে খুঁজছি...", expanded=False) as status:
        search_query = prompt
        if image_analysis:
            search_query = f"{prompt} {image_analysis}"

        # Embed query
        _emb_result = embed_client.models.embed_content(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-004"),
            contents=[search_query],
        )
        _q_vec = list(_emb_result.embeddings[0].values)

        # Intent-based retrieval
        _results, _hints = retriever.retrieve(search_query, _q_vec, top_k=8)
        book_context = build_context(_results, _hints)

        _intent = _hints.intent.value
        _ch     = f" | অধ্যায় {_hints.chapter}" if _hints.chapter else ""
        status.update(
            label=f"✅ পাওয়া গেছে [{_intent}{_ch}] — {len(_results)} chunks",
            state="complete"
        )

    # Inject figure descriptions into context so LLM can describe diagrams
    _relevant_figs: list[dict] = []
    _seen_fig_keys: set[str] = set()
    for _r in _results:
        for _fig in get_figures_for_pages(
            _r.get("page_start", 0), _r.get("page_end", _r.get("page_start", 0)), max_figs=2
        ):
            _fk = _fig.get("key", "")
            if _fk and _fk not in _seen_fig_keys:
                _seen_fig_keys.add(_fk)
                _relevant_figs.append(_fig)
    if _relevant_figs:
        book_context = inject_figures_into_context(book_context, _results, max_total_figs=4)


    # --- Check if diagram needed ---
    diagram_bytes = None
    diagram_type = detect_diagram_type(prompt)
    if diagram_type:
        with st.spinner(f"📊 {diagram_type} diagram তৈরি করছি..."):
            diagram_bytes = generate_physics_diagram(diagram_type, {})

    # --- Build system prompt ---
    profile_context = profile.get_prompt_context()
    system_prompt = build_system_prompt(profile_context, book_context)

    if profile.preferred_style != "auto":
        style_instructions = {
            "analogy": "উপমা এবং তুলনা দিয়ে ব্যাখ্যা করো।",
            "formula": "সূত্র দিয়ে শুরু করো, তারপর ব্যাখ্যা করো।",
            "example": "বাস্তব উদাহরণ দিয়ে শুরু করো।",
            "step_by_step": "ধাপে ধাপে সংখ্যা দিয়ে ব্যাখ্যা করো।",
        }
        system_prompt += f"\n\nব্যাখ্যার ধরন: {style_instructions.get(profile.preferred_style, '')}"

    if st.session_state.quiz_mode:
        system_prompt += QUIZ_PROMPT_SUFFIX
    if st.session_state.hint_mode:
        system_prompt += HINT_PROMPT_SUFFIX
    if image_analysis:
        system_prompt += f"\n\n## ছাত্রের পাঠানো ছবির বিশ্লেষণ\n{image_analysis}"

    # --- Build messages (last 10 for context window management) ---
    # Mirrors: ai-cookbook/knowledge/docling/5-chat.py get_chat_response() pattern
    chat_messages = []
    for msg in st.session_state.messages[-10:]:
        chat_messages.append({"role": msg["role"], "content": msg["content"]})

    # --- Stream response — docling/5-chat.py pattern ─────────────
    # st.chat_message("assistant") + st.write_stream() ensures:
    # streaming and stored message rendering are identical (LaTeX ✅ markdown ✅)
    with st.chat_message("assistant"):
        # Show diagram first if generated
        if diagram_bytes:
            st.image(diagram_bytes)

        # Build proper multi-turn Gemini chat with Content/Part objects
        # (better than flat conversation string — preserves role boundaries)
        tutor_model = GenerativeModel(
            model_name="gemini-2.5-pro",
            system_instruction=system_prompt,
        )
        chat_history = [
            Content(role="user" if m["role"] == "user" else "model",
                    parts=[Part.from_text(m["content"])])
            for m in chat_messages[:-1]
        ]
        chat = tutor_model.start_chat(history=chat_history)
        last_msg = chat_messages[-1]["content"] if chat_messages else prompt

        def gemini_stream():
            try:
                response = chat.send_message(
                    last_msg,
                    generation_config={"temperature": 0.5},
                    stream=True,
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                yield f"\n\n⚠️ দুঃখিত, উত্তর দিতে সমস্যা হয়েছে: {str(e)}"

        # Stream directly inside chat_message — mirrors docling/5-chat.py line 90
        full_response = st.write_stream(gemini_stream())

        # Book figures expander (inside same assistant message)
        if _relevant_figs:
            displayable = [
                f for f in _relevant_figs
                if Path(f.get("image_path", "")).exists()
            ][:4]
            if displayable:
                with st.expander(f"📖 বইয়ের সংশ্লিষ্ট চিত্র ({len(displayable)}টি)", expanded=False):
                    cols = st.columns(min(len(displayable), 2))
                    for i, fig in enumerate(displayable):
                        cap = fig.get("caption", "") or f"চিত্র | অধ্যায় {fig.get('chapter_num','?')} | পৃষ্ঠা {fig.get('book_page','?')}"
                        with cols[i % 2]:
                            st.image(fig["image_path"], caption=cap, use_container_width=True)

    # Store assistant message — mirrors docling/5-chat.py line 180-181
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "diagram": diagram_bytes,
    })

    profile.save()
    st.rerun()  # refresh sidebar session list
