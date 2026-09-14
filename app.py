"""
Physics Tutor Agent v3 — Streamlit App (Thin UI)
=================================================
বাংলাদেশের Class 9-10 পদার্থবিজ্ঞান বইয়ের উপর ভিত্তি করে
adaptive AI tutor — powered by Gemini 2.5 Pro (Vertex AI)

Architecture: UI only — all logic in services/
Dave Pattern: GENERAL FALLBACK (Streamlit, not FastAPI)
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

import numpy as np
import streamlit as st
import vertexai
from google import genai as google_genai
from vertexai.generative_models import Content, GenerativeModel, Part

sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from services.gemini_service import GeminiService
from services.genome_service import GenomeService
from services.prompt_service import PromptService
from services.routing_service import RoutingService
from services.diagram_service import detect_diagram_type, generate_physics_diagram
from services.debounce_service import MessageBuffer
from src.retrieval import PhysicsTutorRetriever, build_context
from src.figure_context import inject_figures_into_context, get_figures_for_pages
from utils.latex_formatter import format_latex_for_streamlit


# -------------------------------------------------------------- #
#   Page Config                                                    #
# -------------------------------------------------------------- #

st.set_page_config(
    page_title="Physics Tutor — Class 9-10",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------- #
#   CSS                                                            #
# -------------------------------------------------------------- #

def load_css():
    css_path = Path(__file__).parent / "static" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

load_css()


# -------------------------------------------------------------- #
#   Constants                                                      #
# -------------------------------------------------------------- #

LEVEL_EMOJI = {1: "🌱", 2: "📗", 3: "📘", 4: "📙", 5: "🏆"}
LEVEL_NAMES = {1: "শিক্ষানবিশ", 2: "শিখছি", 3: "মোটামুটি", 4: "ভালো", 5: "বিশেষজ্ঞ"}

STYLE_LABELS = {
    "auto": "🤖 স্বয়ংক্রিয়",
    "analogy": "🔗 উপমা",
    "formula": "📐 সূত্র",
    "example": "💡 উদাহরণ",
    "step_by_step": "🪜 ধাপে ধাপে",
}

GUIDANCE_LABELS = {
    "auto": "🤖 স্বয়ংক্রিয়",
    "direct": "📖 সরাসরি শেখাও",
    "self": "🧠 নিজে চেষ্টা করতে চাই",
}

PERSONALITY_LABELS = {
    "auto": "🤖 স্বয়ংক্রিয়",
    "friendly": "😊 বন্ধুসুলভ",
    "formal": "👔 আনুষ্ঠানিক",
    "storyteller": "📖 গল্পকার",
    "competitive": "🏆 প্রতিযোগী",
}

STUDENT_ID = "default_student"


# -------------------------------------------------------------- #
#   Initialize Session State — Services + Data                     #
# -------------------------------------------------------------- #

def init_services():
    """Initialize all services once per session."""
    if "services_initialized" not in st.session_state:
        # Vertex AI init
        vertexai.init(
            project=settings.gcp_project,
            location=settings.gcp_location,
        )

        # Services
        st.session_state.gemini_svc = GeminiService(settings)
        st.session_state.genome_svc = GenomeService(settings)
        st.session_state.prompt_svc = PromptService()
        from services.voice_service import VoiceService
        st.session_state.voice_svc = VoiceService(settings)

        # RAG — existing retrieval system (untouched)
        import lancedb
        _db = lancedb.connect(settings.lancedb_path)
        _child_df = _db.open_table("child_chunks").to_pandas()
        _child_vec = np.array(_child_df["vector"].tolist(), dtype=np.float32)
        _parent_df = _db.open_table("parent_chunks").to_pandas()
        st.session_state.retriever = PhysicsTutorRetriever(_child_df, _child_vec, _parent_df)
        st.session_state.embed_client = google_genai.Client(
            vertexai=True,
            project=settings.gcp_project,
            location=settings.gcp_location,
        )

        # Routing service (orchestrates everything)
        st.session_state.routing_svc = RoutingService(
            gemini=st.session_state.gemini_svc,
            genome_svc=st.session_state.genome_svc,
            prompt_svc=st.session_state.prompt_svc,
            retriever=None,  # We handle RAG in app.py for figure injection
            settings=settings,
        )

        st.session_state.services_initialized = True


def init_session_data():
    """Initialize per-session data."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "genome" not in st.session_state:
        genome_svc = st.session_state.genome_svc
        st.session_state.genome = genome_svc.load_or_create(STUDENT_ID)
        genome_svc.start_session(st.session_state.genome)
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "last_topics" not in st.session_state:
        st.session_state.last_topics = []
    if "last_hint_level" not in st.session_state:
        st.session_state.last_hint_level = None


init_services()
init_session_data()

# Convenience aliases
genome_svc: GenomeService = st.session_state.genome_svc
gemini_svc: GeminiService = st.session_state.gemini_svc
routing_svc: RoutingService = st.session_state.routing_svc
prompt_svc: PromptService = st.session_state.prompt_svc
voice_svc = st.session_state.voice_svc
retriever: PhysicsTutorRetriever = st.session_state.retriever
embed_client = st.session_state.embed_client
genome = st.session_state.genome


# -------------------------------------------------------------- #
#   Sidebar                                                        #
# -------------------------------------------------------------- #

with st.sidebar:
    # Brand
    st.markdown("""<div style="padding: 0.4rem 0 0.2rem 0;">
        <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.01em;">⚛️ পদার্থবিদ Tutor</div>
        <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">Class 9-10 · NCTB Bangladesh</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # New Chat
    if st.button("✦ নতুন কথোপকথন", use_container_width=True, key="new_chat_btn"):
        st.session_state.messages = []
        st.session_state.last_response = None
        st.session_state.last_topics = []
        st.session_state.last_hint_level = None
        genome_svc.save(genome)
        st.session_state.genome = genome_svc.load_or_create(STUDENT_ID)
        genome_svc.start_session(st.session_state.genome)
        st.rerun()

    st.divider()

    # Understanding Level (read-only — system-controlled)
    level = genome.understanding_level
    st.markdown(f"""<div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.07em;
        color:#94a3b8; font-weight:600; margin-bottom:6px;">বোঝার স্তর</div>""",
        unsafe_allow_html=True)
    st.markdown(f"""<div class="level-badge">{LEVEL_EMOJI[level]} {LEVEL_NAMES[level]} — {level}/5</div>""",
        unsafe_allow_html=True)
    st.progress((level - 1) / 4)

    st.divider()

    # Settings (3 student-controllable dropdowns)
    with st.expander("⚙️  সেটিংস", expanded=False):
        # 1. Explanation Style
        st.markdown("**ব্যাখ্যার ধরন**")
        style_choice = st.selectbox(
            "style",
            options=list(STYLE_LABELS.keys()),
            format_func=lambda x: STYLE_LABELS[x],
            index=list(STYLE_LABELS.keys()).index(genome.preferences.style),
            label_visibility="collapsed",
            key="style_select",
        )
        if style_choice != genome.preferences.style:
            genome.preferences.style = style_choice

        # 2. Guidance Level
        st.markdown("**সাহায্যের মাত্রা**")
        guidance_choice = st.selectbox(
            "guidance",
            options=list(GUIDANCE_LABELS.keys()),
            format_func=lambda x: GUIDANCE_LABELS[x],
            index=list(GUIDANCE_LABELS.keys()).index(genome.preferences.guidance),
            label_visibility="collapsed",
            key="guidance_select",
        )
        if guidance_choice != genome.preferences.guidance:
            genome.preferences.guidance = guidance_choice

        # 3. Personality
        st.markdown("**টিউটরের ভাষা**")
        personality_choice = st.selectbox(
            "personality",
            options=list(PERSONALITY_LABELS.keys()),
            format_func=lambda x: PERSONALITY_LABELS[x],
            index=list(PERSONALITY_LABELS.keys()).index(genome.preferences.personality),
            label_visibility="collapsed",
            key="personality_select",
        )
        if personality_choice != genome.preferences.personality:
            genome.preferences.personality = personality_choice

        # Stats
        if genome.total_questions > 0:
            st.markdown("**পরিসংখ্যান**")
            c1, c2 = st.columns(2)
            c1.metric("প্রশ্ন", genome.total_questions)
            c2.metric("সেশন", genome.session_count)

    # Footer
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.caption("Powered by Gemini 2.5 Pro ✨")


# -------------------------------------------------------------- #
#   Chat History Display                                           #
# -------------------------------------------------------------- #

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user" and message.get("image_bytes"):
            st.image(message["image_bytes"], width=240)
        
        if message["role"] == "assistant":
            if message.get("diagram"):
                st.image(message["diagram"])
            st.markdown(message["content"])
            
            # TTS Output (Voice)
            cols = st.columns([1, 10])
            with cols[0]:
                if st.button("🔊", key=f"tts_btn_{idx}", help="শুনতে ক্লিক করো"):
                    with st.spinner("অডিও তৈরি হচ্ছে..."):
                        if "audio_bytes" not in message:
                            message["audio_bytes"] = voice_svc.generate_speech(message["content"])
            
            # Render audio player if we have audio bytes
            if message.get("audio_bytes"):
                st.audio(message["audio_bytes"], format="audio/mp3", autoplay=True)
        else:
            st.markdown(message["content"])

# Welcome message
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""আস্সালামু আলাইকুম! আমি **পদার্থবিদ** — তোমার পদার্থবিজ্ঞান টিউটর। 🎉

আমি তোমাকে Class 9-10 Physics বুঝতে সাহায্য করব।

**আমি যা করতে পারি:**

📖 বইয়ের যেকোনো বিষয় ব্যাখ্যা করতে পারি
📐 সূত্র সহজ ভাষায় বোঝাতে পারি
📷 তোমার সমস্যার ছবি দেখে সাহায্য করতে পারি
🧪 তোমার বোঝার মাত্রা বুঝে নিজে থেকে adapt করি

**কীভাবে জিজ্ঞেস করবে?**
বাংলায়, English এ, বা Banglish এ — যেভাবে ইচ্ছা! নিচে প্রশ্ন লেখো 👇""")



#   Chat Input — Debounce Buffer                                   #
#   Pattern: n8n Message Queue (store → wait → flush)              #
# -------------------------------------------------------------- #

# Step 1: Capture new input → add to buffer
if chat_submission := st.chat_input(
    "পদার্থবিজ্ঞান নিয়ে প্রশ্ন করো...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg"],
    accept_audio=True,
):
    # Streamlit 1.37+ returns a dict-like object when accept_file is True
    prompt_text = chat_submission.text
    
    # Process uploaded image if any
    _img_bytes = None
    if chat_submission.files:
        _img_bytes = chat_submission.files[0].read()

    # Process voice input (STT) if any
    if hasattr(chat_submission, "audio") and chat_submission.audio:
        with st.spinner("🎤 কথা শুনছি..."):
            audio_bytes = chat_submission.audio.read()
            transcribed_text = voice_svc.transcribe(audio_bytes)
            if transcribed_text:
                # If there was also typed text, append the voice text
                if prompt_text:
                    prompt_text += f"\n[Voice]: {transcribed_text}"
                else:
                    prompt_text = transcribed_text
            else:
                st.toast("দুঃখিত, কথা বুঝতে পারিনি। আবার চেষ্টা করো।", icon="⚠️")

    # If both are empty, do nothing
    if not prompt_text and not _img_bytes:
        st.stop()

    # Display user message immediately
    with st.chat_message("user"):
        if _img_bytes:
            st.image(_img_bytes, width=240)
        if prompt_text:
            st.markdown(prompt_text)

    # Store in chat history for display
    st.session_state.messages.append({
        "role": "user",
        "content": prompt_text,
        "image_bytes": _img_bytes,
    })

    # Add to debounce buffer (n8n: action="store")
    count = MessageBuffer.add(st.session_state, prompt_text, _img_bytes)

    # Rerun to start countdown
    st.rerun()


# -------------------------------------------------------------- #
#   Debounce Countdown — Wait for More Messages                    #
#   n8n equivalent: action="read" → skip if latest ≠ my_ts         #
# -------------------------------------------------------------- #

if MessageBuffer.has_pending(st.session_state):
    if not MessageBuffer.should_flush(st.session_state):
        # Still waiting — show countdown
        remaining = MessageBuffer.remaining_seconds(st.session_state)
        count = MessageBuffer.peek_count(st.session_state)
        plural = "টি message" if count > 1 else "টি message"
        st.markdown(
            f"""<div style="text-align:center; padding:12px; margin:8px 0;
            background:linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
            border-radius:12px; border:1px solid #4338ca;">
            <div style="font-size:1.5rem;">⏳</div>
            <div style="color:#c7d2fe; font-size:0.85rem; margin-top:4px;">
                {count}{plural} পেয়েছি — আরো কিছু বলতে চাইলে এখনই লেখো!
            </div>
            <div style="color:#818cf8; font-size:0.75rem; margin-top:2px;">
                {remaining:.0f} সেকেন্ড পর process করবো...
            </div>
            </div>""",
            unsafe_allow_html=True,
        )
        import time as _time
        _time.sleep(1)
        st.rerun()

    else:
        # ──────────────────────────────────────────────────────── #
        #   FLUSH! Process All Buffered Messages as Unified Context #
        #   n8n equivalent: action="read" → latest == my_ts → FLUSH #
        # ──────────────────────────────────────────────────────── #

        flush_result = MessageBuffer.flush(st.session_state)
        unified_text = flush_result.unified_text
        image_bytes = flush_result.image_bytes

        # Multi-message indicator
        if flush_result.message_count > 1:
            st.info(
                f"📦 {flush_result.message_count}টি message একসাথে process করছি..."
            )

        # --- Process image if present ---
        image_context = None
        if image_bytes:
            with st.spinner("ছবি বিশ্লেষণ করছি..."):
                image_context = gemini_svc.analyze_image(image_bytes, unified_text)

        # --- Step 13-14: Evaluate PREVIOUS interaction ---
        if st.session_state.last_response and len(st.session_state.messages) >= 3:
            signal_result = gemini_svc.evaluate_signal(
                student_reply=unified_text,
                tutor_previous=st.session_state.last_response,
                current_topics=st.session_state.last_topics,
            )
            genome_svc.record_signal(
                genome=genome,
                signal_result=signal_result,
                topics=st.session_state.last_topics,
                hint_level=st.session_state.last_hint_level,
            )

        # --- Steps 1-3: Analyze message (LLM Call 1) ---
        analysis = gemini_svc.analyze_message(unified_text, image_context)

        # --- Steps 5-9: Deterministic routing ---
        routing = routing_svc._route(analysis, genome)

        # --- Step 10: RAG Retrieval ---
        book_context = ""
        _relevant_figs: list[dict] = []
        with st.status("📚 বই থেকে খুঁজছি...", expanded=False) as status:
            search_query = unified_text
            if image_context:
                search_query = f"{unified_text} {image_context}"

            _emb_result = embed_client.models.embed_content(
                model=settings.embedding_model,
                contents=[search_query],
            )
            _q_vec = list(_emb_result.embeddings[0].values)

            _results, _hints = retriever.retrieve(search_query, _q_vec, top_k=8)
            book_context = build_context(_results, _hints)

            _intent = _hints.intent.value
            _ch = f" | অধ্যায় {_hints.chapter}" if _hints.chapter else ""
            status.update(
                label=f"✅ পাওয়া গেছে [{_intent}{_ch}] — {len(_results)} chunks",
                state="complete",
            )

        # Inject figure descriptions
        _seen_fig_keys: set[str] = set()
        for _r in _results:
            for _fig in get_figures_for_pages(
                _r.get("page_start", 0),
                _r.get("page_end", _r.get("page_start", 0)),
                max_figs=2,
            ):
                _fk = _fig.get("key", "")
                if _fk and _fk not in _seen_fig_keys:
                    _seen_fig_keys.add(_fk)
                    _relevant_figs.append(_fig)
        if _relevant_figs:
            book_context = inject_figures_into_context(
                book_context, _results, max_total_figs=4
            )

        # --- Check if diagram needed ---
        diagram_bytes = None
        diagram_type = detect_diagram_type(unified_text)
        if diagram_type:
            with st.spinner(f"📊 {diagram_type} diagram তৈরি করছি..."):
                diagram_bytes = generate_physics_diagram(diagram_type, {})

        # --- Step 11: Prompt Assembly ---
        genome_context = genome_svc.get_genome_context(genome)
        system_prompt = prompt_svc.build_system_prompt(
            routing=routing,
            genome_context=genome_context,
            book_context=book_context,
            image_context=image_context,
        )

        # --- Step 12: Generate Response (LLM Call 2 — streaming) ---
        chat_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-settings.max_conversation_history:]
        ]

        with st.chat_message("assistant"):
            # Show diagram first if generated
            if diagram_bytes:
                st.image(diagram_bytes)

            response_stream = gemini_svc.generate_response_stream(
                system_prompt=system_prompt,
                chat_messages=chat_messages,
                user_message=unified_text,
            )
            full_response = st.write_stream(response_stream)

            # Book figures expander
            if _relevant_figs:
                displayable = [
                    f for f in _relevant_figs
                    if Path(f.get("image_path", "")).exists()
                ][:4]
                if displayable:
                    with st.expander(
                        f"📖 বইয়ের সংশ্লিষ্ট চিত্র ({len(displayable)}টি)",
                        expanded=False,
                    ):
                        cols = st.columns(min(len(displayable), 2))
                        for i, fig in enumerate(displayable):
                            cap = fig.get("caption", "") or (
                                f"চিত্র | অধ্যায় {fig.get('chapter_num', '?')}"
                                f" | পৃষ্ঠা {fig.get('book_page', '?')}"
                            )
                            with cols[i % 2]:
                                st.image(
                                    fig["image_path"],
                                    caption=cap,
                                    use_container_width=True,
                                )

        # Store response
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "diagram": diagram_bytes,
        })

        # Store for next-turn evaluation
        st.session_state.last_response = full_response
        st.session_state.last_topics = analysis.topics
        st.session_state.last_hint_level = routing.hint_level

        # Save genome
        genome_svc.save(genome)
        st.rerun()

