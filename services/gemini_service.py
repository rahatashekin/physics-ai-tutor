# -------------------------------------------------------------- #
#   Gemini Service â€” LLM API Wrapper                               #
#   Dave Pattern: ADAPT â€” Provider Adapter, structured output      #
#   Implementation Type: ADAPT (Gemini instead of OpenAI)          #
# -------------------------------------------------------------- #

from __future__ import annotations

import json
import logging
from typing import Generator

from google.genai import Client
from google.genai.types import GenerateContentConfig
from vertexai.generative_models import GenerativeModel, Content, Part

from config import Settings
from models.routing import AnalysisResult, SignalResult

logger = logging.getLogger(__name__)


class GeminiService:
    """Wraps all Gemini API calls behind a clean interface.

    Three LLM calls per message:
      1. analyze_message()  â€” Flash â€” Steps 1-3 (override, topic, intent)
      2. generate_response() â€” Pro â€” Step 12 (main tutor response, streaming)
      3. evaluate_signal()   â€” Flash â€” Step 13 (signal + misconception detection)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._flash_client = Client(vertexai=True)
        # Pro model for streaming uses Vertex AI SDK directly
        self._pro_model_name = settings.gemini_pro_model
        self._flash_model_name = settings.gemini_flash_model

    # -------------------------------------------------------------- #
    #   LLM Call 1: Analyze Message (Steps 1-3)                        #
    # -------------------------------------------------------------- #

    def analyze_message(
        self,
        message: str,
        image_context: str | None = None,
    ) -> AnalysisResult:
        """Analyze student message: detect override, topics, and intent.

        Uses Gemini Flash for speed and cost efficiency.
        Returns structured AnalysisResult via JSON schema enforcement.
        """
        context_part = ""
        if image_context:
            context_part = f"\n\nImage context: {image_context}"

        prompt = f"""Analyze this student message from a physics tutoring session.

Student message: "{message}"{context_part}

Detect:
1. Override: Is the student explicitly requesting something? (direct answer, hint, quiz, specific style like analogy, formula, example, step-by-step, or diagram)
2. Topics: What physics topics are mentioned or implied? Use Bangla topic names.
3. Intent: What does the student want? (learn concept, solve problem, clarify, practice, want diagram, general, greeting)

Return a JSON object with these fields:
- override: one of NONE, DIRECT_ANSWER, WANT_HINT, WANT_QUIZ, STYLE_ANALOGY, STYLE_FORMULA, STYLE_EXAMPLE, STYLE_STEP_BY_STEP, STYLE_DIAGRAM
- topics: list of topic strings in Bangla
- intent: one of CONCEPT, PROBLEM, CLARIFY, PRACTICE_REQUEST, WANT_DIAGRAM, GENERAL, GREETING"""

        try:
            response = self._flash_client.models.generate_content(
                model=f"publishers/google/models/{self._flash_model_name}",
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=AnalysisResult.model_json_schema(),
                ),
            )
            return AnalysisResult.model_validate_json(response.text)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                override="NONE",
                topics=[],
                intent="GENERAL",
            )

    # -------------------------------------------------------------- #
    #   LLM Call 2: Generate Response (Step 12) â€” Streaming            #
    # -------------------------------------------------------------- #

    def draw_diagram(self, description: str) -> str:
        """Generates a diagram or picture based on the description.
        Call this function whenever the user asks you to draw a picture, diagram, or visualize something.
        """
        import base64
        import uuid
        try:
            response = self._flash_client.models.generate_content(
                model=f"publishers/google/models/gemini-2.5-flash-image",
                contents=description
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                    placeholder = f"DIAGRAM_PLACEHOLDER_{uuid.uuid4().hex[:8]}"
                    if not hasattr(self, '_image_cache'):
                        self._image_cache = {}
                    self._image_cache[placeholder] = b64
                    return f"Successfully generated diagram. Include exactly this in your response: ![Diagram]({placeholder})"
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return f"Failed to generate diagram: {e}"
        return "Failed to generate diagram."

    def generate_response_stream(
        self,
        system_prompt: str,
        chat_messages: list[dict[str, str]],
        user_message: str,
    ) -> Generator[str, None, None]:
        """Generate streaming tutor response using Gemini Pro.

        Uses Google GenAI SDK for streaming and Automatic Function Calling (AFC).
        System prompt contains ALL routing decisions baked in.
        """
        try:
            from google.genai import types

            # Build chat history from previous messages
            history = []
            for msg in chat_messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                history.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
                )

            # We use the flash client instance which is initialized with vertexai=True
            # But we override the model name to use the pro model for teaching
            chat = self._flash_client.chats.create(
                model=f"publishers/google/models/{self._pro_model_name}",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.5,
                    tools=[self.draw_diagram]
                ),
                history=history
            )

            response_stream = chat.send_message_stream(user_message)
            
            buffer = ""
            import re
            import base64
            
            for chunk in response_stream:
                if chunk.text:
                    buffer += chunk.text
                    
                    # Check for partial placeholder at the end of the buffer
                    if "DIAGRAM_PLACEHOLDER" in buffer[-30:]:
                        idx = buffer.rfind("DIAGRAM_PLACEHOLDER")
                        if len(buffer) - idx < 30:
                            # Might be incomplete, yield everything before it
                            if idx > 0:
                                yield buffer[:idx]
                                buffer = buffer[idx:]
                            continue
                            
                    if "BOOK_FIGURE:" in buffer:
                        idx = buffer.rfind("BOOK_FIGURE:")
                        if ")" not in buffer[idx:]:
                            if idx > 0:
                                yield buffer[:idx]
                                buffer = buffer[idx:]
                            continue
                        
                        matches = re.finditer(r"BOOK_FIGURE:([^)]+)", buffer)
                        import urllib.parse
                        for match in matches:
                            path = match.group(1)
                            decoded_path = urllib.parse.unquote(path)
                            try:
                                with open(decoded_path, "rb") as f:
                                    b64 = base64.b64encode(f.read()).decode("utf-8")
                                buffer = buffer.replace(f"BOOK_FIGURE:{path}", f"data:image/png;base64,{b64}")
                            except Exception as e:
                                logger.error(f"Failed to load book figure {decoded_path}: {e}")
                                buffer = buffer.replace(f"BOOK_FIGURE:{path}", "https://via.placeholder.com/400?text=Image+Not+Found")
                    
                    # Replace complete placeholders
                    if hasattr(self, '_image_cache'):
                        for ph, b64 in self._image_cache.items():
                            if ph in buffer:
                                buffer = buffer.replace(ph, f"data:image/jpeg;base64,{b64}")
                    
                    yield buffer
                    buffer = ""
                    
            if buffer:
                # Final flush
                if hasattr(self, '_image_cache'):
                    for ph, b64 in list(self._image_cache.items()):
                        if ph in buffer:
                            buffer = buffer.replace(ph, f"data:image/jpeg;base64,{b64}")
                            del self._image_cache[ph]
                
                matches = re.finditer(r"BOOK_FIGURE:([^)]+)", buffer)
                import urllib.parse
                for match in matches:
                    path = match.group(1)
                    decoded_path = urllib.parse.unquote(path)
                    try:
                        with open(decoded_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        buffer = buffer.replace(f"BOOK_FIGURE:{path}", f"data:image/png;base64,{b64}")
                    except Exception as e:
                        logger.error(f"Failed to load book figure {decoded_path}: {e}")
                        buffer = buffer.replace(f"BOOK_FIGURE:{path}", "https://via.placeholder.com/400?text=Image+Not+Found")

                yield buffer

            # Yield any remaining images that the LLM forgot to output
            if hasattr(self, '_image_cache'):
                for ph, b64 in self._image_cache.items():
                    yield f"\n\n![Diagram](data:image/jpeg;base64,{b64})"
                self._image_cache.clear()

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            yield f"\n\n⚠ দুঃখিত, উত্তর দিতে সমস্যা হয়েছে: {e}"

    # -------------------------------------------------------------- #
    #   LLM Call 3: Evaluate Signal (Step 13)                          #
    # -------------------------------------------------------------- #

    def evaluate_signal(
        self,
        student_reply: str,
        tutor_previous: str,
        current_topics: list[str],
    ) -> SignalResult:
        """Detect comprehension signal and misconceptions from student reply.

        Uses Gemini Flash for speed and cost.
        Returns structured SignalResult.
        """
        topics_str = ", ".join(current_topics) if current_topics else "unknown"

        prompt = f"""You are evaluating a student's reply in a physics tutoring session.

Current topic(s): {topics_str}

Tutor's previous message:
"{tutor_previous}"

Student's reply:
"{student_reply}"

Analyze:
1. Signal: How well did the student understand?
   - confused: Did not understand, frustrated, wrong answer
   - partial: Partially understood, trying but uncertain
   - understood: Correctly understood, shows confidence
   - mastered: Can solve independently, deep understanding

2. Misconception: Did the student reveal any wrong understanding?
   - If yes, describe what they believe incorrectly
   - If no, set to null

Return a JSON object with:
- signal: one of confused, partial, understood, mastered
- misconception: string description or null
- misconception_topic: topic name (Bangla) or null"""

        try:
            response = self._flash_client.models.generate_content(
                model=f"publishers/google/models/{self._flash_model_name}",
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=SignalResult.model_json_schema(),
                ),
            )
            return SignalResult.model_validate_json(response.text)
        except Exception as e:
            logger.error(f"Signal evaluation failed: {e}")
            return SignalResult(
                signal="partial",
                misconception=None,
                misconception_topic=None,
            )

    # -------------------------------------------------------------- #
    #   Image Analysis                                                 #
    # -------------------------------------------------------------- #

    def analyze_image(self, image_bytes: bytes, student_message: str = "") -> str:
        """Analyze uploaded image (textbook photo, handwritten problem, etc.).

        Returns text description of the image content.
        """
        import base64

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt = f"""Analyze this image from a physics student.
Describe what you see: equations, diagrams, problem text, handwritten notes.
If it's a physics problem, extract the complete problem statement.
If the student also wrote a message, consider it: "{student_message}"

Respond in Bangla."""

        try:
            response = self._flash_client.models.generate_content(
                model=f"publishers/google/models/{self._flash_model_name}",
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": b64_image,
                                }
                            },
                        ],
                    }
                ],
                config=GenerateContentConfig(temperature=0.3),
            )
            return response.text
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return ""
