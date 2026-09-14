# -------------------------------------------------------------- #
#   Debounce Service — Message Buffer for Rapid Multi-Input        #
#   Adapted from: n8n "Message Queue (Debounce Buffer)" workflow   #
#   Pattern: store → wait → flush unified context                  #
# -------------------------------------------------------------- #
#                                                                  #
#   Problem:                                                       #
#     Student: "এটা দেখো"        → msg 1                           #
#     Student: 📷 ছবি             → msg 2 (2s পর)                   #
#     Student: "3 নম্বর প্রশ্ন"   → msg 3 (1s পর)                   #
#     ❌ 3টা আলাদা reply? — বাজে!                                   #
#     ✅ 3s wait → সব collect → 1 unified reply                    #
#                                                                  #
#   n8n original:                                                  #
#     action="store" → queue তে push (text/image/audio/ts)         #
#     action="read"  → latest ≠ my_ts? SKIP : FLUSH                #
#     TTL = 120s                                                   #
#                                                                  #
#   Python adaptation:                                             #
#     add() → buffer তে push                                      #
#     should_flush() → elapsed >= DEBOUNCE_SECONDS?                #
#     flush() → unified text + last image → ready for pipeline     #
# -------------------------------------------------------------- #

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# -------------------------------------------------------------- #
#   Configuration                                                  #
# -------------------------------------------------------------- #

DEBOUNCE_SECONDS = 12.0  # Wait 12s after last message before processing


# -------------------------------------------------------------- #
#   Flush Result — Unified Context                                 #
# -------------------------------------------------------------- #

@dataclass
class FlushResult:
    """Unified context from multiple buffered messages.
    
    Mirrors n8n SalesForge's "Unified Context Builder":
    - unified_text: all texts joined (like unified_query)
    - image_bytes: last image (like image_url)
    - message_count: how many messages were collected
    - individual_texts: each text separately (for display)
    """
    unified_text: str
    image_bytes: Optional[bytes]
    message_count: int
    individual_texts: list[str] = field(default_factory=list)


# -------------------------------------------------------------- #
#   MessageBuffer — Streamlit Session State Based                  #
# -------------------------------------------------------------- #

class MessageBuffer:
    """Debounce buffer for rapid student messages.
    
    Uses Streamlit session_state as storage (like n8n's staticData).
    
    Usage in app.py:
        # On chat input:
        MessageBuffer.add(st.session_state, text, image_bytes)
        st.rerun()
        
        # On every rerun:
        if MessageBuffer.has_pending(st.session_state):
            if MessageBuffer.should_flush(st.session_state):
                result = MessageBuffer.flush(st.session_state)
                # process result.unified_text ...
            else:
                remaining = MessageBuffer.remaining_seconds(st.session_state)
                st.info(f"⏳ {remaining:.0f}s...")
                time.sleep(1)
                st.rerun()
    """

    # Session state keys (namespaced to avoid collision)
    _KEY_BUFFER = "_debounce_buffer"
    _KEY_LATEST = "_debounce_latest"

    @staticmethod
    def add(
        session_state,
        text: str,
        image_bytes: Optional[bytes] = None,
    ) -> int:
        """Store a message in the buffer. Returns buffer size.
        
        n8n equivalent: action="store"
        """
        if MessageBuffer._KEY_BUFFER not in session_state:
            session_state[MessageBuffer._KEY_BUFFER] = []

        session_state[MessageBuffer._KEY_BUFFER].append({
            "text": text or "",
            "image_bytes": image_bytes,
            "ts": time.time(),
        })
        session_state[MessageBuffer._KEY_LATEST] = time.time()

        return len(session_state[MessageBuffer._KEY_BUFFER])

    @staticmethod
    def has_pending(session_state) -> bool:
        """Check if there are buffered messages waiting."""
        return bool(session_state.get(MessageBuffer._KEY_BUFFER, []))

    @staticmethod
    def should_flush(session_state) -> bool:
        """Check if debounce window has elapsed.
        
        n8n equivalent: action="read" → latest == my_ts → flush
        """
        latest = session_state.get(MessageBuffer._KEY_LATEST)
        if latest is None:
            return False
        return (time.time() - latest) >= DEBOUNCE_SECONDS

    @staticmethod
    def remaining_seconds(session_state) -> float:
        """Seconds remaining before flush."""
        latest = session_state.get(MessageBuffer._KEY_LATEST)
        if latest is None:
            return 0
        return max(0.0, DEBOUNCE_SECONDS - (time.time() - latest))

    @staticmethod
    def flush(session_state) -> FlushResult:
        """Flush all buffered messages into a unified context.
        
        n8n equivalent: action="read" → return messages → delete queue
        
        Unified Context Builder logic:
        - Multiple texts → join with newline
        - Multiple images → use the LAST one (most recent)
        - Returns FlushResult ready for pipeline
        """
        messages = session_state.get(MessageBuffer._KEY_BUFFER, [])

        # Clear buffer (like n8n: delete staticData.q[sender])
        session_state[MessageBuffer._KEY_BUFFER] = []
        session_state[MessageBuffer._KEY_LATEST] = None

        if not messages:
            return FlushResult(
                unified_text="",
                image_bytes=None,
                message_count=0,
            )

        # Collect texts (filter empty)
        texts = [m["text"] for m in messages if m["text"].strip()]

        # Use last image (if multiple images, latest is most relevant)
        image_bytes = None
        for m in reversed(messages):
            if m.get("image_bytes"):
                image_bytes = m["image_bytes"]
                break

        # Build unified text
        if len(texts) == 1:
            unified_text = texts[0]
        elif len(texts) > 1:
            # Join multiple messages into one context
            unified_text = "\n".join(texts)
        else:
            unified_text = ""

        return FlushResult(
            unified_text=unified_text,
            image_bytes=image_bytes,
            message_count=len(messages),
            individual_texts=texts,
        )

    @staticmethod
    def peek_count(session_state) -> int:
        """How many messages are in the buffer right now."""
        return len(session_state.get(MessageBuffer._KEY_BUFFER, []))
