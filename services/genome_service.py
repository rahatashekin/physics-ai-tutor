# -------------------------------------------------------------- #
#   Genome Service — Student Genome CRUD + Signal Processing       #
#   Dave Pattern: ADAPT — Service class wrapping data operations   #
#   Implementation Type: ADAPT                                     #
# -------------------------------------------------------------- #

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from config import Settings
from models.genome import (
    Misconception,
    Signal,
    StudentGenome,
    StudentPreferences,
    TopicMastery,
)
from models.routing import SignalResult

logger = logging.getLogger(__name__)

# Hint level → mastery credit mapping (VISION.md Hint Cascade)
HINT_CREDIT_MAP: dict[str, float] = {
    "L0": 1.0,   # Solved alone (Socratic only) → full credit
    "L1": 0.8,   # Directional hint → high credit
    "L2": 0.5,   # Structural hint → medium credit
    "L3": 0.3,   # Step-by-step → low credit
    "L4": 0.1,   # Direct answer → minimal credit
}

MAX_SIGNALS: int = 20


class GenomeService:
    """Manages Student Genome lifecycle: load, update, save.

    All genome operations are deterministic (code, not AI).
    JSON file storage for MVP — PostgreSQL in production.
    """

    def __init__(self, settings: Settings) -> None:
        self._genome_dir = Path(settings.genome_dir)
        self._genome_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------- #
    #   CRUD Operations                                                #
    # -------------------------------------------------------------- #

    def load_or_create(self, student_id: str) -> StudentGenome:
        """Load existing genome from JSON, or create a new one."""
        filepath = self._genome_dir / f"{student_id}.json"
        if filepath.exists():
            try:
                data = filepath.read_text(encoding="utf-8")
                genome = StudentGenome.model_validate_json(data)
                logger.info(f"Loaded genome for {student_id}")
                return genome
            except Exception as e:
                logger.warning(f"Failed to load genome {student_id}: {e}, creating new")

        genome = StudentGenome(student_id=student_id)
        logger.info(f"Created new genome for {student_id}")
        return genome

    def save(self, genome: StudentGenome) -> None:
        """Persist genome to JSON file."""
        filepath = self._genome_dir / f"{genome.student_id}.json"
        data = genome.model_dump_json(indent=2)
        filepath.write_text(data, encoding="utf-8")
        logger.debug(f"Saved genome for {genome.student_id}")

    # -------------------------------------------------------------- #
    #   Mastery Update (Step 14 — deterministic)                       #
    # -------------------------------------------------------------- #

    def update_mastery(
        self,
        genome: StudentGenome,
        topic: str,
        signal_type: str,
        hint_level: str | None,
    ) -> None:
        """Update topic mastery based on signal and hint level used.

        Credit-weighted: solving alone (L0) = full credit,
        solving with heavy hints (L4) = minimal credit.
        """
        if not topic:
            return

        # Get or create topic mastery
        if topic not in genome.mastery:
            genome.mastery[topic] = TopicMastery()
        mastery = genome.mastery[topic]

        mastery.attempts += 1
        mastery.last_seen = datetime.now()

        if signal_type == "mastered":
            credit = HINT_CREDIT_MAP.get(hint_level, 0.5) if hint_level else 0.8
            mastery.score = min(1.0, mastery.score + credit * 0.15)
            mastery.correct_streak += 1
        elif signal_type == "understood":
            credit = HINT_CREDIT_MAP.get(hint_level, 0.5) if hint_level else 0.6
            mastery.score = min(1.0, mastery.score + credit * 0.10)
            mastery.correct_streak += 1
        elif signal_type == "partial":
            mastery.score = min(1.0, mastery.score + 0.03)
            mastery.correct_streak = 0
        elif signal_type == "confused":
            mastery.score = max(0.0, mastery.score - 0.05)
            mastery.correct_streak = 0

        # Update average hint level
        if hint_level is not None:
            level_num = int(hint_level[1])  # "L2" → 2
            mastery.avg_hint_level = (
                (mastery.avg_hint_level * (mastery.attempts - 1) + level_num)
                / mastery.attempts
            )

    # -------------------------------------------------------------- #
    #   Signal Recording                                               #
    # -------------------------------------------------------------- #

    def record_signal(
        self,
        genome: StudentGenome,
        signal_result: SignalResult,
        topics: list[str],
        hint_level: str | None,
    ) -> None:
        """Record comprehension signal and update genome accordingly."""

        # Record signal
        topic_str = topics[0] if topics else ""
        hint_num = int(hint_level[1]) if hint_level else None

        signal = Signal(
            type=signal_result.signal,
            topic=topic_str,
            hint_level_used=hint_num,
        )
        genome.signals.append(signal)

        # Keep only last N signals
        if len(genome.signals) > MAX_SIGNALS:
            genome.signals = genome.signals[-MAX_SIGNALS:]

        # Update mastery for each topic
        for topic in topics:
            self.update_mastery(genome, topic, signal_result.signal, hint_level)

        # Record misconception if detected
        if signal_result.misconception and signal_result.misconception_topic:
            misconception = Misconception(
                topic=signal_result.misconception_topic,
                belief=signal_result.misconception,
            )
            genome.misconceptions.append(misconception)

        # Adapt understanding level
        self._adapt_level(genome)

        # Increment counters
        genome.total_questions += 1
        genome.messages_since_last_check += 1
        genome.message_count_current_topic += 1

    # -------------------------------------------------------------- #
    #   Level Adaptation (deterministic)                               #
    # -------------------------------------------------------------- #

    def _adapt_level(self, genome: StudentGenome) -> None:
        """Adapt understanding level based on recent signals.

        If last 5 signals are mostly confused → level down.
        If last 5 signals are mostly mastered → level up.
        """
        if len(genome.signals) < 3:
            return

        recent = genome.signals[-5:]
        confused_count = sum(1 for s in recent if s.type == "confused")
        mastered_count = sum(1 for s in recent if s.type in ("mastered", "understood"))

        if confused_count >= 3 and genome.understanding_level > 1:
            genome.understanding_level -= 1
            logger.info(f"Level DOWN → {genome.understanding_level}")
        elif mastered_count >= 4 and genome.understanding_level < 5:
            genome.understanding_level += 1
            logger.info(f"Level UP → {genome.understanding_level}")

    # -------------------------------------------------------------- #
    #   Genome Context for Prompt Injection                            #
    # -------------------------------------------------------------- #

    def get_genome_context(self, genome: StudentGenome) -> str:
        """Convert genome to prompt-injectable text summary.

        This goes into the system prompt so the LLM knows the student.
        """
        parts = []

        # Understanding level
        level_names = {1: "একদম নতুন", 2: "শিক্ষানবিশ", 3: "মাঝামাঝি", 4: "ভালো", 5: "অগ্রসর"}
        parts.append(f"Student Level: {level_names.get(genome.understanding_level, 'মাঝামাঝি')}")

        # Topic mastery summary (top 5 recent)
        if genome.mastery:
            sorted_topics = sorted(
                genome.mastery.items(),
                key=lambda x: x[1].last_seen or datetime.min,
                reverse=True,
            )[:5]
            mastery_lines = []
            for topic, m in sorted_topics:
                pct = int(m.score * 100)
                mastery_lines.append(f"  - {topic}: {pct}% mastered ({m.attempts} attempts)")
            parts.append("Recent Topics:\n" + "\n".join(mastery_lines))

        # Active misconceptions
        active_misc = [m for m in genome.misconceptions if not m.corrected]
        if active_misc:
            misc_lines = [f"  - {m.topic}: {m.belief}" for m in active_misc[-3:]]
            parts.append("Active Misconceptions:\n" + "\n".join(misc_lines))

        # Recent signal trend
        if genome.signals:
            recent = genome.signals[-5:]
            trend = [s.type for s in recent]
            parts.append(f"Recent Signals: {' → '.join(trend)}")

        # Session info
        parts.append(f"Session #{genome.session_count + 1}, Total Questions: {genome.total_questions}")

        return "\n".join(parts)

    # -------------------------------------------------------------- #
    #   Topic Mastery Lookup                                           #
    # -------------------------------------------------------------- #

    def get_topic_mastery(self, genome: StudentGenome, topic: str) -> float:
        """Get mastery score for a topic. Returns 0.0 if new topic."""
        if topic in genome.mastery:
            return genome.mastery[topic].score
        return 0.0

    # -------------------------------------------------------------- #
    #   Session Management                                             #
    # -------------------------------------------------------------- #

    def start_session(self, genome: StudentGenome) -> None:
        """Mark start of a new session."""
        genome.session_count += 1
        genome.last_visit = datetime.now()
        genome.messages_since_last_check = 0
        genome.message_count_current_topic = 0

    def reset_topic_counter(self, genome: StudentGenome) -> None:
        """Reset topic counter when student changes topic."""
        genome.message_count_current_topic = 0

    def reset_check_counter(self, genome: StudentGenome) -> None:
        """Reset check counter after a comprehension check."""
        genome.messages_since_last_check = 0
