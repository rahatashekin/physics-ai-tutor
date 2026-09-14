# -------------------------------------------------------------- #
#   Routing Models — Structured Output for LLM Calls               #
#   Dave Pattern: COPY — Structured Output + Literal routing       #
#   Implementation Type: COPY (DR-013)                             #
# -------------------------------------------------------------- #

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# -------------------------------------------------------------- #
#   LLM Call 1: Analyze (Steps 1-3)                                #
# -------------------------------------------------------------- #

class AnalysisResult(BaseModel):
    """Structured output from LLM Call 1 — message analysis.

    Combines 3 detection tasks in a single call:
      Step 1: Override detection
      Step 2: Topic extraction
      Step 3: Intent classification
    """

    override: Literal[
        "NONE",
        "DIRECT_ANSWER",
        "WANT_HINT",
        "WANT_QUIZ",
        "STYLE_ANALOGY",
        "STYLE_FORMULA",
        "STYLE_EXAMPLE",
        "STYLE_STEP_BY_STEP",
        "STYLE_DIAGRAM",
    ] = Field(
        description=(
            "Student ki explicitly kisu request kortese? "
            "NONE = regular message, no override. "
            "DIRECT_ANSWER = direct answer ba samadhan chaitese. "
            "WANT_HINT = hint chaitese. "
            "WANT_QUIZ = quiz/practice chaitese. "
            "STYLE_* = specific explanation style chaitese (e.g. STYLE_DIAGRAM for drawing a picture). "
        ),
    )
    topics: list[str] = Field(
        description=(
            "Message e thaka identified physics topics (Bangla names). "
            "Example: ['Mohakorsho', 'Boll']. "
            "Empty list if no specific topic (e.g., greeting)."
        ),
    )
    intent: Literal[
        "CONCEPT",
        "PROBLEM",
        "CLARIFY",
        "PRACTICE_REQUEST",
        "WANT_DIAGRAM",
        "GENERAL",
        "GREETING",
    ] = Field(
        description=(
            "Student er intent: "
            "CONCEPT = concept bujhte chaitese ('Bojhao'). "
            "PROBLEM = specific problem solve ('F=10N, a=?'). "
            "CLARIFY = kono topic clear korar request ('Bujhi nai'). "
            "PRACTICE_REQUEST = practice/problem er request. "
            "WANT_DIAGRAM = chobi, chitra ba diagram eke bujhanor request ('Chobi eke bujhao'). "
            "GENERAL = general question ('physics ki'). "
            "GREETING = salam or casual ('ki khobor')."
        ),
    )


# -------------------------------------------------------------- #
#   LLM Call 3: Evaluate Signal (Step 13)                          #
# -------------------------------------------------------------- #

class SignalResult(BaseModel):
    """Structured output from LLM Call 3 — signal detection.

    Analyzes student's reply to determine comprehension level
    and detect any misconceptions.
    """

    signal: Literal["confused", "partial", "understood", "mastered"] = Field(
        description=(
            "Student কতটুকু বুঝেছে: "
            "confused = বুঝেনি, হতাশ, ভুল উত্তর। "
            "partial = কিছুটা বুঝেছে, trying but uncertain। "
            "understood = সঠিক বুঝেছে, confidence আছে। "
            "mastered = নিজে solve করতে পারছে, deeply understood।"
        ),
    )
    misconception: str | None = Field(
        None,
        description=(
            "যদি ভুল ধারণা detect হয় — student কী ভুল বুঝেছে। "
            "Example: 'ওজন আর ভর একই মনে করছে'। "
            "null if no misconception detected."
        ),
    )
    misconception_topic: str | None = Field(
        None,
        description=(
            "কোন topic এ misconception। "
            "Example: 'ভর ও ওজন'। "
            "null if no misconception."
        ),
    )


# -------------------------------------------------------------- #
#   Deterministic Routing Decision (Steps 5-9)                     #
# -------------------------------------------------------------- #

class RoutingDecision(BaseModel):
    """Output of the deterministic routing pipeline.

    Created by code (IF/ELSE), not by LLM.
    Contains all decisions needed to assemble the system prompt.
    """

    mode: Literal["TEACH", "PRACTICE"] = Field(
        description="Fundamental mode — teach concepts or practice problems",
    )
    strategy: Literal[
        "SCAFFOLDING",
        "CHALLENGE",
        "RESCUE",
        "ASSESSMENT",
        "RECALL",
        "EXPLORATION",
    ] = Field(
        description="Teaching strategy based on mastery + signals",
    )
    hint_level: Literal["L0", "L1", "L2", "L3", "L4"] | None = Field(
        None,
        description="Hint cascade level (Practice mode only). None in TEACH mode.",
    )
    resolved_style: str = Field(
        description="Final explanation style (from preference or genome)",
    )
    resolved_personality: str = Field(
        description="Final personality tone (from preference or genome)",
    )
    interrupt: Literal[
        "QUICK_CHECK",
        "MISCONCEPTION_CORRECT",
        "RECALL_PROBE",
    ] | None = Field(
        None,
        description="Interrupt type to inject (or None for normal flow)",
    )
