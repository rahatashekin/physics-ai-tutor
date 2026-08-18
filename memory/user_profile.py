"""
User Profile Manager
=====================
Student এর session profile manage করে।
Understanding level, preferred explanation style, topic history track করে।
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

PROFILES_DIR = Path("profiles")
PROFILES_DIR.mkdir(exist_ok=True)

ExplanationStyle = Literal["analogy", "formula", "example", "step_by_step", "auto"]
UnderstandingSignal = Literal["confused", "partial", "understood", "mastered"]


@dataclass
class TopicHistory:
    attempts: int = 0
    understood: int = 0

    @property
    def success_rate(self) -> float:
        return self.understood / self.attempts if self.attempts > 0 else 0.0


@dataclass
class UserProfile:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    understanding_level: int = 3          # 1 (beginner) → 5 (advanced)
    preferred_style: ExplanationStyle = "auto"
    recent_signals: list[str] = field(default_factory=list)   # last 5 signals
    topic_history: dict[str, dict] = field(default_factory=dict)
    total_questions: int = 0
    total_understood: int = 0

    # ----------------------------------------------------------------
    # Signal Processing — adaptive learning core
    # ----------------------------------------------------------------

    def record_signal(self, signal: UnderstandingSignal, topic: str = "general") -> None:
        """Record a comprehension signal and update level + style."""
        # Keep last 5 signals
        self.recent_signals.append(signal)
        if len(self.recent_signals) > 5:
            self.recent_signals.pop(0)

        # Update topic history
        if topic not in self.topic_history:
            self.topic_history[topic] = {"attempts": 0, "understood": 0}
        self.topic_history[topic]["attempts"] += 1

        self.total_questions += 1

        if signal in ("understood", "mastered"):
            self.topic_history[topic]["understood"] += 1
            self.total_understood += 1

        # Adapt level based on recent signals
        self._adapt_level()

    def _adapt_level(self) -> None:
        """Adjust understanding level based on recent signal pattern."""
        if len(self.recent_signals) < 2:
            return

        last3 = self.recent_signals[-3:] if len(self.recent_signals) >= 3 else self.recent_signals

        confused_count = last3.count("confused")
        understood_count = last3.count("understood") + last3.count("mastered")

        if confused_count >= 2 and self.understanding_level > 1:
            self.understanding_level -= 1   # Simplify
        elif understood_count >= 2 and self.understanding_level < 5:
            self.understanding_level += 1   # Increase complexity

    # ----------------------------------------------------------------
    # Profile-based prompt context
    # ----------------------------------------------------------------

    def get_prompt_context(self) -> str:
        """Return a string describing the user's profile for the system prompt."""
        level_desc = {
            1: "একেবারে শুরু করা শিক্ষার্থী — খুব সহজ ভাষায়, বাস্তব উদাহরণ দিয়ে বোঝাতে হবে",
            2: "মোটামুটি বোঝে — সহজ ভাষায়, ধাপে ধাপে",
            3: "গড় Class 9-10 শিক্ষার্থী — স্বাভাবিক ব্যাখ্যা",
            4: "ভালো বোঝে — সূত্র এবং বিশ্লেষণ দিতে পারো",
            5: "উন্নত শিক্ষার্থী — গভীর ব্যাখ্যা এবং কঠিন সমস্যা দাও",
        }

        struggle_topics = [
            topic for topic, hist in self.topic_history.items()
            if hist["attempts"] > 0 and hist["understood"] / hist["attempts"] < 0.5
        ]

        context = f"শিক্ষার্থীর বোঝার স্তর: {level_desc[self.understanding_level]}"

        if self.preferred_style != "auto":
            style_desc = {
                "analogy": "উপমা/তুলনা দিয়ে বোঝাতে পছন্দ করে",
                "formula": "সরাসরি সূত্র দিয়ে শুরু করতে পছন্দ করে",
                "example": "উদাহরণ দিয়ে শুরু করতে পছন্দ করে",
                "step_by_step": "ধাপে ধাপে বোঝাতে পছন্দ করে",
            }
            context += f"\nপছন্দের ব্যাখ্যার ধরন: {style_desc.get(self.preferred_style, '')}"

        if struggle_topics:
            context += f"\nযে বিষয়গুলোতে কষ্ট হচ্ছে: {', '.join(struggle_topics[:3])}"

        return context

    # ----------------------------------------------------------------
    # Detect signal from user message
    # ----------------------------------------------------------------

    @staticmethod
    def detect_signal_from_message(message: str) -> UnderstandingSignal | None:
        """Parse user message to detect comprehension signal."""
        msg = message.lower()

        confused_keywords = [
            "বুঝলাম না", "বুঝি না", "বুঝতে পারছি না", "কঠিন", "আরেকটু সহজ",
            "confusing", "don't understand", "not clear", "বুইঝা পারতেছি না"
        ]
        understood_keywords = [
            "বুঝলাম", "বুঝেছি", "আচ্ছা", "ওকে", "ধন্যবাদ",
            "understood", "got it", "clear", "thanks", "বুইজা গেছি", "বুজছি"
        ]
        mastered_keywords = [
            "পারলাম", "সমাধান করলাম", "correct", "সঠিক", "এখন পুরো বুঝলাম"
        ]

        if any(kw in msg for kw in confused_keywords):
            return "confused"
        if any(kw in msg for kw in mastered_keywords):
            return "mastered"
        if any(kw in msg for kw in understood_keywords):
            return "understood"
        return None

    # ----------------------------------------------------------------
    # Persistence (session file)
    # ----------------------------------------------------------------

    def save(self) -> None:
        path = PROFILES_DIR / f"{self.session_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, session_id: str) -> "UserProfile":
        path = PROFILES_DIR / f"{session_id}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profile = cls(**data)
            return profile
        return cls(session_id=session_id)

    @classmethod
    def new_session(cls) -> "UserProfile":
        return cls()
