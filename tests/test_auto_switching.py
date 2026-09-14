import sys
import vertexai
from google import genai as google_genai
from datetime import datetime

from config import Settings
from services.gemini_service import GeminiService
from services.genome_service import GenomeService
from services.routing_service import RoutingService
from services.prompt_service import PromptService
from models.genome import TopicMastery, Signal

settings = Settings()
vertexai.init(project=settings.gcp_project, location=settings.gcp_location)

gemini_svc = GeminiService(settings)
genome_svc = GenomeService(settings)
prompt_svc = PromptService()
routing_svc = RoutingService(gemini=gemini_svc, genome_svc=genome_svc, prompt_svc=prompt_svc, retriever=None, settings=settings)

STUDENT_ID = "auto_switch_tester"
genome = genome_svc.load_or_create(STUDENT_ID)

# Ensure Auto is set
genome.preferences.style = "auto"
genome.preferences.guidance = "auto"
genome.preferences.personality = "auto"
genome.effective_style = "auto"
genome.observed_personality = "auto"

def setup_competitive(g):
    g.understanding_level = 4
    g.mastery["বলের অংক"] = TopicMastery(score=0.9, attempts=5, correct_streak=3, avg_hint_level=1.0)
    for _ in range(4):
        g.signals.append(Signal(type="mastered", topic="বলের অংক", timestamp=datetime.now()))

test_cases = [
    {
        "name": "Trigger: Storyteller (Concept + Low Level)",
        "message": "ভাইয়া, নিউটনের প্রথম সূত্রটা আমি কিছুই বুঝতে পারছি না। একদম সহজ করে বলো।",
        "setup_genome": lambda g: setattr(g, 'understanding_level', 1),
    },
    {
        "name": "Trigger: Formula (Problem Intent)",
        "message": "৫ কেজি ভরের বস্তুর ত্বরণ ২ হলে বল কত হবে?",
        "setup_genome": lambda g: setattr(g, 'understanding_level', 3),
    },
    {
        "name": "Trigger: Analogy (Clarify Intent)",
        "message": "তুমি যা বললা সেটা একটু কনফিউজিং। নিউটনের তৃতীয় সূত্রটা আরেকবার বোঝাবা?",
        "setup_genome": lambda g: setattr(g, 'understanding_level', 3),
    },
    {
        "name": "Trigger: Competitive (Challenge Strategy)",
        "message": "ভাইয়া, আমি বলের অংকগুলো সব পারি। আমাকে একটা কঠিন অংক দাও তো দেখি!",
        "setup_genome": setup_competitive,
    }
]

print("Starting Auto-Switching Validation...\n" + "-"*50)

for tc in test_cases:
    print(f"\n[TEST CASE]: {tc['name']}")
    print(f"[STUDENT]: {tc['message']}")
    
    # Setup
    genome.mastery = {}
    genome.signals = []
    tc['setup_genome'](genome)
    
    # Analyze
    analysis = gemini_svc.analyze_message(tc['message'], None)
    
    # Route
    routing = routing_svc._route(analysis, genome)
    
    print(f" [AI: Intent detected]: {analysis.intent}")
    print(f" [SYSTEM: Strategy selected]: {routing.strategy}")
    print(f" => [AUTO-SWITCH: Style]: {routing.resolved_style}")
    print(f" => [AUTO-SWITCH: Personality]: {routing.resolved_personality}")
    print(f" => [AUTO-SWITCH: Guidance/Mode]: {routing.mode} (Hint: {routing.hint_level})")
