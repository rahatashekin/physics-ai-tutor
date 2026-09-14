# -------------------------------------------------------------- #
#   Prompt Service — Modular Prompt Assembly                       #
#   Dave Pattern: ADAPT — State machine prompts                    #
#   Implementation Type: ADAPT                                     #
# -------------------------------------------------------------- #

from __future__ import annotations

from models.routing import RoutingDecision
from agent.prompts import (
    BASE_SYSTEM_PROMPT,
    TEACH_MODE_PROMPT,
    PRACTICE_MODE_PROMPT,
    STRATEGY_PROMPTS,
    HINT_LEVEL_PROMPTS,
    STYLE_PROMPTS,
    PERSONALITY_PROMPTS,
    INTERRUPT_PROMPTS,
)


class PromptService:
    """Assembles the final system prompt from modular components.

    Dave's state machine prompt pattern: different prompt modules
    are selected based on routing decisions (code-driven).
    The LLM never sees the routing logic — only the final assembled prompt.
    """

    def build_system_prompt(
        self,
        routing: RoutingDecision,
        genome_context: str,
        book_context: str,
        image_context: str | None = None,
    ) -> str:
        """Assemble complete system prompt from routing decisions.

        Structure (Dave's canonical order):
          1. Role + Base instructions
          2. Mode section (TEACH or PRACTICE)
          3. Strategy section
          4. Hint level section (Practice only)
          5. Style section
          6. Personality section
          7. Interrupt section (if any)
          8. Student genome context
          9. Book/RAG context
          10. Image context (if any)
        """
        sections: list[str] = []

        # 1. Base prompt (role, language, general behavior)
        sections.append(BASE_SYSTEM_PROMPT)

        # 2. Mode
        if routing.mode == "TEACH":
            sections.append(TEACH_MODE_PROMPT)
        else:
            sections.append(PRACTICE_MODE_PROMPT)

        # 3. Strategy
        strategy_prompt = STRATEGY_PROMPTS.get(routing.strategy, "")
        if strategy_prompt:
            sections.append(strategy_prompt)

        # 4. Hint Level (Practice mode only)
        if routing.hint_level and routing.hint_level in HINT_LEVEL_PROMPTS:
            sections.append(HINT_LEVEL_PROMPTS[routing.hint_level])

        # 5. Explanation Style
        style_prompt = STYLE_PROMPTS.get(routing.resolved_style, "")
        if style_prompt:
            sections.append(style_prompt)

        # 6. Personality
        personality_prompt = PERSONALITY_PROMPTS.get(routing.resolved_personality, "")
        if personality_prompt:
            sections.append(personality_prompt)

        # 7. Interrupt (if any)
        if routing.interrupt and routing.interrupt in INTERRUPT_PROMPTS:
            sections.append(INTERRUPT_PROMPTS[routing.interrupt])

        # 8. Student genome context
        if genome_context:
            sections.append(f"## Student Profile\n{genome_context}")

        # 9. Book/RAG context
        if book_context:
            sections.append(f"## Reference Material (NCTB Textbook)\n{book_context}")

        # 10. Image context
        if image_context:
            sections.append(f"## Student's Uploaded Image Analysis\n{image_context}")

        return "\n\n".join(sections)
