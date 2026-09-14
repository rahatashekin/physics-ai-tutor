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
import logging

logging.basicConfig(level=logging.DEBUG)

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

embed_client = google_genai.Client(
    vertexai=True,
    project=settings.gcp_project,
    location=settings.gcp_location,
)

routing_svc = RoutingService(gemini=gemini_svc, genome_svc=genome_svc, prompt_svc=prompt_svc, retriever=None, settings=settings)

STUDENT_ID = "test_eval_student"
genome = genome_svc.load_or_create(STUDENT_ID)
chat_messages = []

def run_turn(student_message):
    global genome
    print(f"\n[STUDENT]: {student_message}")
    
    analysis = gemini_svc.analyze_message(student_message, None)
    print(f" [AI: Analysis] Intent: {analysis.intent} | Topics: {analysis.topics} | Override: {analysis.override}")
    
    routing = routing_svc._route(analysis, genome)
    print(f" [SYSTEM: Routing] Mode: {routing.mode} | Strategy: {routing.strategy} | Style: {routing.resolved_style}")
    
    _emb_result = embed_client.models.embed_content(
        model=settings.embedding_model,
        contents=[student_message],
    )
    _q_vec = list(_emb_result.embeddings[0].values)
    _results, _hints = retriever.retrieve(student_message, _q_vec, top_k=5)
    book_context = build_context(_results, _hints)
    
    genome_context = genome_svc.get_genome_context(genome)
    system_prompt = prompt_svc.build_system_prompt(
        routing=routing,
        genome_context=genome_context,
        book_context=book_context,
        image_context=None,
    )
    
    chat_messages.append({"role": "user", "content": student_message})
    
    response_stream = gemini_svc.generate_response_stream(
        system_prompt=system_prompt,
        chat_messages=chat_messages,
        user_message=student_message,
    )
    
    full_response = ""
    print(f"\n[TUTOR RESPONSE]:\n")
    for chunk in response_stream:
        # print using utf-8 replacement for powershell
        print(chunk.encode("cp1252", errors="replace").decode("cp1252"), end="", flush=True)
        full_response += chunk
    print("\n" + "="*80)
    chat_messages.append({"role": "assistant", "content": full_response})


scenarios = [
    # Scenario 1: concept with diagram
    "Odhyay 3 er ghorshon (friction) niye kotha bolo, ekta sundor diagram eke bujhao please.",
    # Scenario 2: return a specific image from chapter 3
    "Chapter 3 theke specifically kono ekta figure ba image amake return koro, tader description onujayi nijei akbe."
]

for msg in scenarios:
    run_turn(msg)
