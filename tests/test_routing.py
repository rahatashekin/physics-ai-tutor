# -------------------------------------------------------------- #
#   Routing Logic Unit Tests                                       #
#   Dave Pattern: COPY — L1 Unit Tests (assertions on output)      #
# -------------------------------------------------------------- #

"""Tests for deterministic routing logic (no LLM calls needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.genome import StudentGenome, TopicMastery, Signal, StudentPreferences
from models.routing import AnalysisResult, RoutingDecision
from services.genome_service import GenomeService
from services.prompt_service import PromptService
from config import settings


def make_genome(**kwargs) -> StudentGenome:
    """Helper to create a genome with defaults."""
    defaults = {"student_id": "test"}
    defaults.update(kwargs)
    return StudentGenome(**defaults)


def make_analysis(**kwargs) -> AnalysisResult:
    """Helper to create an AnalysisResult with defaults."""
    defaults = {"override": "NONE", "topics": [], "intent": "CONCEPT"}
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


# -------------------------------------------------------------- #
#   Mode Selection Tests                                           #
# -------------------------------------------------------------- #

def test_concept_intent_gives_teach_mode():
    """CONCEPT intent → TEACH mode."""
    from services.routing_service import RoutingService
    rs = RoutingService.__new__(RoutingService)
    analysis = make_analysis(intent="CONCEPT")
    mode = rs._select_mode(analysis, "auto")
    assert mode == "TEACH", f"Expected TEACH, got {mode}"
    print("✅ test_concept_intent_gives_teach_mode")


def test_problem_intent_gives_practice_mode():
    """PROBLEM intent → PRACTICE mode."""
    from services.routing_service import RoutingService
    rs = RoutingService.__new__(RoutingService)
    analysis = make_analysis(intent="PROBLEM")
    mode = rs._select_mode(analysis, "auto")
    assert mode == "PRACTICE", f"Expected PRACTICE, got {mode}"
    print("✅ test_problem_intent_gives_practice_mode")


def test_direct_guidance_forces_teach():
    """guidance=direct → always TEACH."""
    from services.routing_service import RoutingService
    rs = RoutingService.__new__(RoutingService)
    analysis = make_analysis(intent="PROBLEM")
    mode = rs._select_mode(analysis, "direct")
    assert mode == "TEACH", f"Expected TEACH with direct guidance, got {mode}"
    print("✅ test_direct_guidance_forces_teach")


def test_self_guidance_forces_practice():
    """guidance=self → always PRACTICE."""
    from services.routing_service import RoutingService
    rs = RoutingService.__new__(RoutingService)
    analysis = make_analysis(intent="CONCEPT")
    mode = rs._select_mode(analysis, "self")
    assert mode == "PRACTICE", f"Expected PRACTICE with self guidance, got {mode}"
    print("✅ test_self_guidance_forces_practice")


# -------------------------------------------------------------- #
#   Strategy Selection Tests                                       #
# -------------------------------------------------------------- #

def test_new_topic_gives_scaffolding():
    """New topic (0 mastery) → SCAFFOLDING."""
    from services.routing_service import RoutingService
    gs = GenomeService(settings)
    rs = RoutingService.__new__(RoutingService)
    rs._genome_svc = gs
    genome = make_genome()
    strategy = rs._select_strategy(genome, "force", "CONCEPT")
    assert strategy == "SCAFFOLDING", f"Expected SCAFFOLDING, got {strategy}"
    print("✅ test_new_topic_gives_scaffolding")


def test_confused_signals_give_rescue():
    """3+ confused signals → RESCUE."""
    from services.routing_service import RoutingService
    from datetime import datetime
    gs = GenomeService(settings)
    rs = RoutingService.__new__(RoutingService)
    rs._genome_svc = gs
    genome = make_genome()
    genome.mastery["force"] = TopicMastery(score=0.3, attempts=5)
    for _ in range(3):
        genome.signals.append(Signal(type="confused", topic="force"))
    strategy = rs._select_strategy(genome, "force", "CONCEPT")
    assert strategy == "RESCUE", f"Expected RESCUE, got {strategy}"
    print("✅ test_confused_signals_give_rescue")


def test_high_mastery_gives_challenge():
    """High mastery + understood signals → CHALLENGE."""
    from services.routing_service import RoutingService
    gs = GenomeService(settings)
    rs = RoutingService.__new__(RoutingService)
    rs._genome_svc = gs
    genome = make_genome()
    genome.mastery["force"] = TopicMastery(score=0.8, attempts=10)
    for _ in range(4):
        genome.signals.append(Signal(type="understood", topic="force"))
    strategy = rs._select_strategy(genome, "force", "CONCEPT")
    assert strategy == "CHALLENGE", f"Expected CHALLENGE, got {strategy}"
    print("✅ test_high_mastery_gives_challenge")


# -------------------------------------------------------------- #
#   Hint Level Tests                                               #
# -------------------------------------------------------------- #

def test_teach_mode_no_hints():
    """TEACH mode → no hint level."""
    from services.routing_service import RoutingService
    rs = RoutingService.__new__(RoutingService)
    hint = rs._determine_hint_level("TEACH", "SCAFFOLDING", "auto", 0.5)
    assert hint is None, f"Expected None, got {hint}"
    print("✅ test_teach_mode_no_hints")


def test_self_guidance_gives_l0():
    """guidance=self → always L0 (Socratic)."""
    from services.routing_service import RoutingService
    rs = RoutingService.__new__(RoutingService)
    hint = rs._determine_hint_level("PRACTICE", "SCAFFOLDING", "self", 0.5)
    assert hint == "L0", f"Expected L0, got {hint}"
    print("✅ test_self_guidance_gives_l0")


def test_high_mastery_less_hints():
    """High mastery → L0 (less help)."""
    from services.routing_service import RoutingService
    rs = RoutingService.__new__(RoutingService)
    hint = rs._determine_hint_level("PRACTICE", "SCAFFOLDING", "auto", 0.8)
    assert hint == "L0", f"Expected L0, got {hint}"
    print("✅ test_high_mastery_less_hints")


def test_low_mastery_more_hints():
    """Low mastery → L3 (more help)."""
    from services.routing_service import RoutingService
    rs = RoutingService.__new__(RoutingService)
    hint = rs._determine_hint_level("PRACTICE", "SCAFFOLDING", "auto", 0.1)
    assert hint == "L3", f"Expected L3, got {hint}"
    print("✅ test_low_mastery_more_hints")


# -------------------------------------------------------------- #
#   Prompt Assembly Tests                                          #
# -------------------------------------------------------------- #

def test_prompt_assembly_has_all_sections():
    """Assembled prompt contains all expected sections."""
    ps = PromptService()
    rd = RoutingDecision(
        mode="PRACTICE",
        strategy="RESCUE",
        hint_level="L2",
        resolved_style="analogy",
        resolved_personality="friendly",
        interrupt="QUICK_CHECK",
    )
    prompt = ps.build_system_prompt(rd, "Level: 2", "F=ma textbook context")
    assert "PRACTICE" in prompt or "অনুশীলন" in prompt, "Missing mode section"
    assert "রেসকিউ" in prompt, "Missing strategy section"
    assert "L2" in prompt or "কাঠামো" in prompt, "Missing hint section"
    assert "উপমা" in prompt, "Missing style section"
    assert "বন্ধুসুলভ" in prompt, "Missing personality section"
    assert "Quick Check" in prompt or "বোঝা-যাচাই" in prompt, "Missing interrupt section"
    assert "Level: 2" in prompt, "Missing genome context"
    assert "F=ma" in prompt, "Missing book context"
    print("✅ test_prompt_assembly_has_all_sections")


# -------------------------------------------------------------- #
#   Genome Service Tests                                           #
# -------------------------------------------------------------- #

def test_mastery_increases_on_understood():
    """Understood signal → mastery score increases."""
    from models.routing import SignalResult
    gs = GenomeService(settings)
    genome = make_genome()
    sr = SignalResult(signal="understood", misconception=None, misconception_topic=None)
    gs.record_signal(genome, sr, ["force"], "L1")
    assert genome.mastery["force"].score > 0, "Mastery should increase"
    assert genome.total_questions == 1, "Question count should increase"
    print("✅ test_mastery_increases_on_understood")


def test_mastery_decreases_on_confused():
    """Confused signal on existing topic → mastery decreases."""
    from models.routing import SignalResult
    gs = GenomeService(settings)
    genome = make_genome()
    genome.mastery["force"] = TopicMastery(score=0.5, attempts=5)
    sr = SignalResult(signal="confused", misconception=None, misconception_topic=None)
    gs.record_signal(genome, sr, ["force"], None)
    assert genome.mastery["force"].score < 0.5, "Mastery should decrease"
    print("✅ test_mastery_decreases_on_confused")


def test_misconception_recorded():
    """Misconception in signal → recorded in genome."""
    from models.routing import SignalResult
    gs = GenomeService(settings)
    genome = make_genome()
    sr = SignalResult(
        signal="confused",
        misconception="ভর আর ওজন একই মনে করছে",
        misconception_topic="ভর ও ওজন",
    )
    gs.record_signal(genome, sr, ["ভর ও ওজন"], None)
    assert len(genome.misconceptions) == 1, "Should have 1 misconception"
    assert genome.misconceptions[0].belief == "ভর আর ওজন একই মনে করছে"
    print("✅ test_misconception_recorded")


# -------------------------------------------------------------- #
#   Run All Tests                                                  #
# -------------------------------------------------------------- #

if __name__ == "__main__":
    print("\n🧪 Running routing & genome tests...\n")

    test_concept_intent_gives_teach_mode()
    test_problem_intent_gives_practice_mode()
    test_direct_guidance_forces_teach()
    test_self_guidance_forces_practice()

    test_new_topic_gives_scaffolding()
    test_confused_signals_give_rescue()
    test_high_mastery_gives_challenge()

    test_teach_mode_no_hints()
    test_self_guidance_gives_l0()
    test_high_mastery_less_hints()
    test_low_mastery_more_hints()

    test_prompt_assembly_has_all_sections()

    test_mastery_increases_on_understood()
    test_mastery_decreases_on_confused()
    test_misconception_recorded()

    print(f"\n🎉 All 15 tests passed!\n")
