# -------------------------------------------------------------- #
#   Routing Service — 16-Step DAG Pipeline                         #
#   Dave Pattern: ADAPT — DAG backbone + LLM edges                 #
#   Implementation Type: ADAPT                                     #
# -------------------------------------------------------------- #

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Generator, Literal

from config import Settings
from models.genome import StudentGenome
from models.routing import AnalysisResult, RoutingDecision, SignalResult
from services.genome_service import GenomeService
from services.prompt_service import PromptService

if TYPE_CHECKING:
    from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)


class RoutingService:
    """The core intelligence pipeline: Student Message → Tutor Response.

    Architecture: DAG backbone (deterministic routing) + LLM edges.
    Dave's production recipe: code controls flow, LLM at leaf nodes only.

    Pipeline:
      🧠 LLM Call 1 → analyze (override, topic, intent)
      🔧 CODE       → route (mode, strategy, hint, style, personality)
      🔧 CODE       → retrieve (RAG)
      🔧 CODE       → assemble prompt
      🧠 LLM Call 2 → generate response
    Post-response (on next message):
      🧠 LLM Call 3 → evaluate signal
      🔧 CODE       → update genome
    """

    def __init__(
        self,
        gemini: GeminiService,
        genome_svc: GenomeService,
        prompt_svc: PromptService,
        retriever=None,
        embed_client=None,
        settings: Settings | None = None,
    ) -> None:
        self._gemini = gemini
        self._genome_svc = genome_svc
        self._prompt_svc = prompt_svc
        self._retriever = retriever
        self._embed_client = embed_client
        self._settings = settings

    # -------------------------------------------------------------- #
    #   Main Pipeline                                                  #
    # -------------------------------------------------------------- #

    def process_message(
        self,
        message: str,
        genome: StudentGenome,
        chat_messages: list[dict[str, str]],
        image_context: str | None = None,
        previous_response: str | None = None,
    ) -> tuple[RoutingDecision, str, Generator[str, None, None]]:
        """Execute the full routing pipeline.

        Returns:
            (routing_decision, book_context, response_stream)
        """
        # ---------------------------------------------------------- #
        #   Step 13-14: Evaluate PREVIOUS interaction (if exists)      #
        # ---------------------------------------------------------- #
        if previous_response and len(chat_messages) >= 2:
            self._evaluate_previous(
                student_reply=message,
                tutor_previous=previous_response,
                genome=genome,
                previous_topics=getattr(genome, "_last_topics", []),
                previous_hint=getattr(genome, "_last_hint_level", None),
            )

        # ---------------------------------------------------------- #
        #   Step 1-3: 🧠 LLM Call 1 — Analyze Message                 #
        # ---------------------------------------------------------- #
        analysis = self._gemini.analyze_message(message, image_context)
        logger.info(
            f"Analysis: override={analysis.override}, "
            f"intent={analysis.intent}, topics={analysis.topics}"
        )

        # ---------------------------------------------------------- #
        #   Steps 4-9: 🔧 Deterministic Routing                       #
        # ---------------------------------------------------------- #
        routing = self._route(analysis, genome)
        logger.info(
            f"Routing: mode={routing.mode}, strategy={routing.strategy}, "
            f"hint={routing.hint_level}"
        )

        # Store for next-turn evaluation
        genome._last_topics = analysis.topics  # type: ignore[attr-defined]
        genome._last_hint_level = routing.hint_level  # type: ignore[attr-defined]

        # ---------------------------------------------------------- #
        #   Step 10: 🔧 RAG Retrieval                                  #
        # ---------------------------------------------------------- #
        book_context = self._retrieve(message, analysis)

        # ---------------------------------------------------------- #
        #   Step 11: 🔧 Prompt Assembly                                #
        # ---------------------------------------------------------- #
        genome_context = self._genome_svc.get_genome_context(genome)
        system_prompt = self._prompt_svc.build_system_prompt(
            routing=routing,
            genome_context=genome_context,
            book_context=book_context,
            image_context=image_context,
        )

        # ---------------------------------------------------------- #
        #   Step 12: 🧠 LLM Call 2 — Generate Response (streaming)    #
        # ---------------------------------------------------------- #
        response_stream = self._gemini.generate_response_stream(
            system_prompt=system_prompt,
            chat_messages=chat_messages,
            user_message=message,
        )

        # Update counters
        if analysis.topics and (
            not hasattr(genome, "_current_topics")
            or set(analysis.topics) != set(getattr(genome, "_current_topics", []))
        ):
            self._genome_svc.reset_topic_counter(genome)
            genome._current_topics = analysis.topics  # type: ignore[attr-defined]

        return routing, book_context, response_stream

    # -------------------------------------------------------------- #
    #   Deterministic Routing (Steps 4-9)                              #
    # -------------------------------------------------------------- #

    def _route(self, analysis: AnalysisResult, genome: StudentGenome) -> RoutingDecision:
        """Pure deterministic routing — no AI, just IF/ELSE."""

        # Step 5: Mode Selection
        mode = self._select_mode(analysis, genome.preferences.guidance)

        # Step 6: Strategy Selection
        primary_topic = analysis.topics[0] if analysis.topics else ""
        strategy = self._select_strategy(genome, primary_topic, analysis.intent)

        # Step 7: Hint Level
        mastery = self._genome_svc.get_topic_mastery(genome, primary_topic)
        hint_level = self._determine_hint_level(
            mode, strategy, genome.preferences.guidance, mastery,
        )

        # Step 8: Preferences Resolution
        resolved_style, resolved_personality = self._resolve_preferences(
            genome=genome,
            analysis=analysis,
            strategy=strategy,
        )

        # Step 9: Interrupt Check
        interrupt = self._check_interrupt(genome)

        # Handle overrides
        if analysis.override == "DIRECT_ANSWER":
            hint_level = "L4"
        elif analysis.override == "WANT_HINT":
            mode = "PRACTICE"
            if hint_level is None or hint_level > "L1":
                hint_level = "L1"
        elif analysis.override == "WANT_QUIZ":
            mode = "PRACTICE"
            strategy = "ASSESSMENT"
        elif analysis.override.startswith("STYLE_"):
            style_map = {
                "STYLE_ANALOGY": "analogy",
                "STYLE_FORMULA": "formula",
                "STYLE_EXAMPLE": "example",
                "STYLE_STEP_BY_STEP": "step_by_step",
            }
            resolved_style = style_map.get(analysis.override, resolved_style)

        return RoutingDecision(
            mode=mode,
            strategy=strategy,
            hint_level=hint_level,
            resolved_style=resolved_style,
            resolved_personality=resolved_personality,
            interrupt=interrupt,
        )

    # -------------------------------------------------------------- #
    #   Step 5: Mode Selection                                         #
    # -------------------------------------------------------------- #

    def _select_mode(
        self,
        analysis: AnalysisResult,
        guidance: str,
    ) -> Literal["TEACH", "PRACTICE"]:
        """Determine fundamental mode from intent + guidance preference."""

        if guidance == "direct":
            return "TEACH"
        elif guidance == "self":
            return "PRACTICE"

        # Auto mode — based on intent
        if analysis.intent in ("CONCEPT", "CLARIFY", "GENERAL", "GREETING"):
            return "TEACH"
        elif analysis.intent in ("PROBLEM", "PRACTICE_REQUEST"):
            return "PRACTICE"

        return "TEACH"  # default

    # -------------------------------------------------------------- #
    #   Step 6: Strategy Selection                                     #
    # -------------------------------------------------------------- #

    def _select_strategy(
        self,
        genome: StudentGenome,
        topic: str,
        intent: str,
    ) -> str:
        """Select teaching strategy based on genome signals and mastery."""

        mastery = self._genome_svc.get_topic_mastery(genome, topic)

        # Count recent signals
        recent = genome.signals[-5:] if genome.signals else []
        confused_count = sum(1 for s in recent if s.type == "confused")
        understood_count = sum(1 for s in recent if s.type in ("understood", "mastered"))

        # Check days since last visit
        from datetime import datetime
        days_since_last = 0
        if genome.last_visit:
            days_since_last = (datetime.now() - genome.last_visit).days

        # Strategy selection — deterministic thresholds
        # Order matters: more specific conditions first
        if days_since_last >= 2 and mastery > 0:
            return "RECALL"
        elif mastery == 0.0:
            return "SCAFFOLDING"  # Brand new topic
        elif confused_count >= 2:
            return "RESCUE"  # Struggling (has some mastery but confused)
        elif mastery < 0.2:
            return "RESCUE"  # Very low mastery
        elif mastery > 0.7 and understood_count >= 3:
            return "CHALLENGE"
        elif genome.message_count_current_topic >= 5:
            return "ASSESSMENT"
        elif intent == "CLARIFY":
            return "EXPLORATION"
        else:
            return "SCAFFOLDING"

    # -------------------------------------------------------------- #
    #   Step 7: Hint Level Determination                               #
    # -------------------------------------------------------------- #

    def _determine_hint_level(
        self,
        mode: str,
        strategy: str,
        guidance: str,
        mastery: float,
    ) -> str | None:
        """Determine hint cascade level. None in TEACH mode."""

        if mode == "TEACH":
            return None

        # Guidance preference overrides
        if guidance == "self":
            return "L0"  # always Socratic
        elif guidance == "direct":
            return "L3"  # near-direct

        # Auto — genome-based
        if mastery > 0.7:
            level = "L0"
        elif mastery > 0.4:
            level = "L1"
        elif mastery > 0.2:
            level = "L2"
        else:
            level = "L3"

        # Strategy adjustments
        if strategy == "RESCUE":
            # Move toward more help
            level_num = int(level[1])
            level = f"L{min(4, level_num + 1)}"
        elif strategy == "CHALLENGE":
            # Move toward less help
            level_num = int(level[1])
            level = f"L{max(0, level_num - 1)}"

        return level

    # -------------------------------------------------------------- #
    #   Step 8: Preferences Resolution                                 #
    # -------------------------------------------------------------- #

    def _resolve_preferences(
        self,
        genome: StudentGenome,
        analysis: AnalysisResult,
        strategy: str,
    ) -> tuple[str, str]:
        """Resolve style and personality ? auto or student override."""

        # Style resolution
        if genome.preferences.style != "auto":
            resolved_style = genome.preferences.style
        elif genome.effective_style != "auto":
            resolved_style = genome.effective_style
        else:
            # Dynamic switching when Auto
            if analysis.override == "STYLE_DIAGRAM" or analysis.intent == "WANT_DIAGRAM":
                resolved_style = "diagram"
            elif strategy == "RESCUE":
                resolved_style = "step_by_step"
            elif analysis.intent == "CLARIFY":
                resolved_style = "analogy"
            elif analysis.intent == "PROBLEM":
                resolved_style = "formula"
            else:
                resolved_style = "example"

        # Personality resolution
        if genome.preferences.personality != "auto":
            resolved_personality = genome.preferences.personality
        elif genome.observed_personality != "auto":
            resolved_personality = genome.observed_personality
        else:
            # Dynamic switching when Auto
            if strategy == "CHALLENGE":
                resolved_personality = "competitive"
            elif analysis.intent == "CONCEPT" and genome.understanding_level < 3:
                resolved_personality = "storyteller"
            elif genome.understanding_level == 5:
                resolved_personality = "formal"
            else:
                resolved_personality = "friendly"

        return resolved_style, resolved_personality

    # -------------------------------------------------------------- #
    #   Step 9: Interrupt Check                                        #
    # -------------------------------------------------------------- #

    def _check_interrupt(self, genome: StudentGenome) -> str | None:
        """Check if we should interrupt with a comprehension check."""

        # Active misconception → correct it
        active_misc = [m for m in genome.misconceptions if not m.corrected]
        if active_misc:
            return "MISCONCEPTION_CORRECT"

        # Every 5 messages → quick check
        if genome.messages_since_last_check >= 5:
            return "QUICK_CHECK"

        return None

    # -------------------------------------------------------------- #
    #   Step 10: RAG Retrieval                                         #
    # -------------------------------------------------------------- #

    def _retrieve(self, message: str, analysis: AnalysisResult) -> str:
        """Retrieve relevant book content using existing RAG system."""

        if not self._retriever:
            return ""

        try:
            results = self._retriever.retrieve(
                query=message,
                query_embedding=None,  # retriever handles embedding internally
                top_k=8,
            )
            if results:
                context_parts = []
                for r in results[:5]:
                    text = r.get("text", r.get("content", ""))
                    if text:
                        context_parts.append(text)
                return "\n\n---\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")

        return ""

    # -------------------------------------------------------------- #
    #   Step 13-14: Evaluate Previous & Update Genome                  #
    # -------------------------------------------------------------- #

    def _evaluate_previous(
        self,
        student_reply: str,
        tutor_previous: str,
        genome: StudentGenome,
        previous_topics: list[str],
        previous_hint: str | None,
    ) -> None:
        """Evaluate student's reply to previous tutor message.

        🧠 LLM Call 3 → detect signal
        🔧 Code → update genome
        """
        signal_result = self._gemini.evaluate_signal(
            student_reply=student_reply,
            tutor_previous=tutor_previous,
            current_topics=previous_topics,
        )

        logger.info(
            f"Signal: {signal_result.signal}, "
            f"misconception={signal_result.misconception}"
        )

        self._genome_svc.record_signal(
            genome=genome,
            signal_result=signal_result,
            topics=previous_topics,
            hint_level=previous_hint,
        )
