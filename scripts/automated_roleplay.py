import sys
import os
import vertexai
from google import genai as google_genai

from config import Settings
from services.gemini_service import GeminiService
from services.genome_service import GenomeService
from services.routing_service import RoutingService
from services.prompt_service import PromptService
from src.retrieval import PhysicsTutorRetriever, build_context

settings = Settings()
vertexai.init(project=settings.gcp_project, location=settings.gcp_location)

gemini_svc = GeminiService(settings)
genome_svc = GenomeService(settings)
prompt_svc = PromptService()

import lancedb
import numpy as np
_db = lancedb.connect(settings.lancedb_path)
_child_df = _db.open_table("child_chunks").to_pandas()
_child_vec = np.array(_child_df["vector"].tolist(), dtype=np.float32)
_parent_df = _db.open_table("parent_chunks").to_pandas()
retriever = PhysicsTutorRetriever(_child_df, _child_vec, _parent_df)

embed_client = google_genai.Client(vertexai=True, project=settings.gcp_project, location=settings.gcp_location)
routing_svc = RoutingService(gemini=gemini_svc, genome_svc=genome_svc, prompt_svc=prompt_svc, retriever=None, settings=settings)

STUDENT_ID = "roleplay_student_sifat"
# Reset genome for a fresh start
genome = genome_svc.load_or_create(STUDENT_ID)
genome.mastery = {}
genome.signals = []
genome.misconceptions = []
genome_svc.save(genome)

chat_messages = []
last_response = ""
last_topics = []
last_hint_level = None

turns = [
    "ভাইয়া, আমি সিফাত। নিউটনের গতির সূত্রগুলো তো পড়লাম, কিন্তু প্রথম সূত্রটা ঠিক বুঝলাম না। জড়তা মানে কী?",
    "ও আচ্ছা! বাসের এক্সাম্পলটা জোস ছিল। এবার বুঝছি। আচ্ছা, ধরো একটা গাড়ির ভর ১০০০ কেজি, আর সেটার ত্বরণ ৫। তাহলে বল কত হবে? আমাকে এটা করতে দাও তো।",
    "বল বের করার সূত্র তো F = ma তাই না? তাহলে ১০০০ কে ৫ দিয়ে ভাগ করতে হবে?",
    "ওহ শিট! ভাগ না, গুণ হবে! F = 1000 * 5 = 5000 নিউটন। ঠিক বলছি?",
    "ভাইয়া একটা জিনিস কনফিউশন হচ্ছে। বস্তুর ভর আর ওজন তো আসলে একই জিনিস, তাই না? খালি নাম আলাদা?",
    "অনেক ঘুরিয়ে বলছো ভাই, আমার এত সময় নাই, সরাসরি উত্তরটা বলে দাও প্লিজ!"
]

def run_turn(student_message):
    global last_response, last_topics, last_hint_level, genome
    print(f"\n[STUDENT]: {student_message}")
    
    # 1. Evaluate previous signal
    if last_response and len(chat_messages) >= 2:
        signal_result = gemini_svc.evaluate_signal(
            student_reply=student_message,
            tutor_previous=last_response,
            current_topics=last_topics,
        )
        print(f" [AI: Evaluation] Signal: {signal_result.signal} | Misconception: {signal_result.misconception}")
        genome_svc.record_signal(
            genome=genome,
            signal_result=signal_result,
            topics=last_topics,
            hint_level=last_hint_level,
        )

    # 2. Analyze new message
    analysis = gemini_svc.analyze_message(student_message, None)
    print(f" [AI: Analysis] Intent: {analysis.intent} | Topics: {analysis.topics} | Override: {analysis.override}")
    
    # 3. Routing
    routing = routing_svc._route(analysis, genome)
    print(f" [SYSTEM: Routing] Mode: {routing.mode} | Strategy: {routing.strategy} | Hint Level: {routing.hint_level}")
    
    # 4. Retrieval
    _emb_result = embed_client.models.embed_content(model=settings.embedding_model, contents=[student_message])
    _q_vec = list(_emb_result.embeddings[0].values)
    _results, _hints = retriever.retrieve(student_message, _q_vec, top_k=5)
    book_context = build_context(_results, _hints)
    
    # 5. Prompt Assembly
    genome_context = genome_svc.get_genome_context(genome)
    system_prompt = prompt_svc.build_system_prompt(routing=routing, genome_context=genome_context, book_context=book_context, image_context=None)
    
    chat_messages.append({"role": "user", "content": student_message})
    
    # 6. Generation
    response_stream = gemini_svc.generate_response_stream(system_prompt=system_prompt, chat_messages=chat_messages, user_message=student_message)
    
    full_response = ""
    print(f"\n[TUTOR RESPONSE]:\n")
    for chunk in response_stream:
        print(chunk, end="", flush=True)
        full_response += chunk
    print("\n" + "="*80)
    
    chat_messages.append({"role": "assistant", "content": full_response})
    last_response = full_response
    last_topics = analysis.topics
    last_hint_level = routing.hint_level
    genome_svc.save(genome)
    
    # Print genome state
    print(f" [DEBUG] Genome Mastery: { {k: v.score for k, v in genome.mastery.items()} }")
    print(f" [DEBUG] Genome Misconceptions: { [m.belief for m in genome.misconceptions] }")
    print("="*80)

for turn in turns:
    run_turn(turn)
