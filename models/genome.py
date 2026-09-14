# -------------------------------------------------------------- #
#   Student Genome — Pydantic Models                               #
#   Dave Pattern: ADAPT — Pydantic-first data modeling (DR-002)    #
#   Implementation Type: ADAPT (educational context)               #
# -------------------------------------------------------------- #

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# -------------------------------------------------------------- #
#   Sub-Models                                                     #
# -------------------------------------------------------------- #

class TopicMastery(BaseModel):
    """Per-topic mastery tracking with hint-weighted scoring."""

    score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="0.0 = new topic, 1.0 = fully mastered",
    )
    attempts: int = Field(0, description="Total interaction count on this topic")
    correct_streak: int = Field(0, description="Consecutive correct answers")
    avg_hint_level: float = Field(
        0.0, ge=0.0, le=4.0,
        description="Average hint level used (lower = more independent)",
    )
    last_seen: datetime | None = Field(None, description="Last interaction timestamp")


class Misconception(BaseModel):
    """A detected misconception in the student's understanding."""

    topic: str = Field(description="কোন topic এ ভুল ধারণা")
    belief: str = Field(description="Student এর ভুল ধারণা কী")
    detected_at: datetime = Field(default_factory=datetime.now)
    corrected: bool = Field(False, description="Tutor কি correct করেছে?")


class Signal(BaseModel):
    """A comprehension signal detected from student's reply."""

    type: Literal["confused", "partial", "understood", "mastered"] = Field(
        description="Student এর comprehension level",
    )
    topic: str = Field(default="", description="কোন topic এ signal")
    timestamp: datetime = Field(default_factory=datetime.now)
    hint_level_used: int | None = Field(
        None, description="কোন hint level এ signal এসেছে",
    )


class StudentPreferences(BaseModel):
    """Student-controlled tutor settings. Default = স্বয়ংক্রিয়."""

    style: Literal[
        "auto", "analogy", "formula", "example", "step_by_step",
    ] = Field("auto", description="ব্যাখ্যার ধরন — auto means genome-driven")
    guidance: Literal[
        "auto", "direct", "self",
    ] = Field("auto", description="সাহায্যের মাত্রা — auto means genome-driven")
    personality: Literal[
        "auto", "friendly", "formal", "storyteller", "competitive",
    ] = Field("auto", description="টিউটরের ভাষা — auto means genome-driven")


# -------------------------------------------------------------- #
#   Student Genome — Main Model                                    #
# -------------------------------------------------------------- #

class StudentGenome(BaseModel):
    """Complete student learning profile — the 'DNA' of a learner.

    5 Layers:
      1. Identity — who the student is
      2. Knowledge Map — mastery per topic
      3. Learning Profile — style, hint patterns, misconceptions
      4. Signals — recent comprehension signals
      5. Memory — session history, preferences
    """

    # Identity
    student_id: str = Field(description="Unique student identifier")
    name: str = Field("", description="Student name (optional)")

    # Knowledge Map
    mastery: dict[str, TopicMastery] = Field(
        default_factory=dict,
        description="Topic → mastery tracking. Key = topic name (Bangla).",
    )

    # Learning Profile
    misconceptions: list[Misconception] = Field(
        default_factory=list,
        description="Active misconceptions (uncorrected first)",
    )
    understanding_level: int = Field(
        3, ge=1, le=5,
        description="Current understanding level (1=beginner, 5=advanced)",
    )
    effective_style: str = Field(
        "auto",
        description="Auto-detected best explanation style for this student",
    )
    observed_personality: str = Field(
        "auto",
        description="Auto-detected best personality tone for this student",
    )

    # Signals
    signals: list[Signal] = Field(
        default_factory=list,
        description="Recent comprehension signals (last 20 kept)",
    )

    # Memory
    preferences: StudentPreferences = Field(
        default_factory=StudentPreferences,
        description="Student-controlled tutor settings",
    )
    session_count: int = Field(0, description="Total sessions")
    total_questions: int = Field(0, description="Total questions asked")
    last_visit: datetime | None = Field(None, description="Last session timestamp")
    message_count_current_topic: int = Field(
        0, description="Messages in current topic (for interrupt scheduling)",
    )
    messages_since_last_check: int = Field(
        0, description="Messages since last comprehension check",
    )
