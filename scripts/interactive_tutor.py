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

print("TUTOR_READY")

while True:
    try:
        student_message = input().strip()
        if not student_message:
            continue
        if student_message == "EXIT":
            break
            
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
        print("TUTOR_READY")
        
        chat_messages.append({"role": "assistant", "content": full_response})
        last_response = full_response
        last_topics = analysis.topics
        last_hint_level = routing.hint_level
        genome_svc.save(genome)

    except EOFError:
        break
    except Exception as e:
        print(f"ERROR: {e}")
        print("TUTOR_READY")
