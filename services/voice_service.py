# -------------------------------------------------------------- #
#   Voice Service — Google Speech-to-Text Integration              #
#   Dave Pattern: COPY — Service class wrapping external API       #
#   Implementation Type: ADAPT                                     #
# -------------------------------------------------------------- #

from __future__ import annotations

import logging

from config import Settings

logger = logging.getLogger(__name__)


class VoiceService:
    """Transcribes Bengali speech to text using Google Speech-to-Text API.

    Used for voice input in the tutor — student speaks into mic,
    we transcribe and process as text message.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None

    def _get_client(self):
        """Lazy init — avoid import error if google-cloud-speech not installed."""
        if self._client is None:
            try:
                from google.cloud import speech
                self._client = speech.SpeechClient()
            except ImportError:
                logger.warning("google-cloud-speech not installed, voice disabled")
                return None
        return self._client

    def transcribe(self, audio_bytes: bytes, language: str = "bn-BD") -> str:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data (WAV or WebM format)
            language: BCP-47 language code. Default: Bangla (Bangladesh)

        Returns:
            Transcribed text, or empty string if failed.
        """
        client = self._get_client()
        if not client:
            return ""

        try:
            from google.cloud import speech

            audio = speech.RecognitionAudio(content=audio_bytes)
            config = speech.RecognitionConfig(
                language_code=language,
                alternative_language_codes=["en-US"],  # Fallback for English
                model="latest_long",
                enable_automatic_punctuation=True,
            )

            response = client.recognize(config=config, audio=audio)

            if response.results:
                transcript = response.results[0].alternatives[0].transcript
                confidence = response.results[0].alternatives[0].confidence
                logger.info(f"Voice transcribed: '{transcript[:50]}...' (confidence={confidence:.2f})")
                return transcript

            logger.warning("No speech detected in audio")
            return ""

        except Exception as e:
            logger.error(f"Voice transcription failed: {e}")
            return ""

    # -------------------------------------------------------------- #
    #   Text-to-Speech (TTS)                                           #
    # -------------------------------------------------------------- #

    def generate_speech(self, text: str, language: str = "bn-IN") -> bytes:
        """Convert text to speech audio bytes using Gemini 3.1 Flash TTS.

        Args:
            text: The text to convert to speech.
            language: BCP-47 language code. Default: bn-IN.

        Returns:
            WAV audio bytes, or empty bytes if failed.
        """
        if not text.strip():
            return b""

        try:
            from google import genai
            from google.genai import types
            
            # Using genai client via Vertex AI
            client = genai.Client(
                vertexai=True, 
                project=self._settings.gcp_project, 
                location=self._settings.gcp_location
            )

            # Gemini-TTS natural language prompt for emotional control
            prompt = f"Speak the following Bengali text in a very cordial, warm, friendly, and enthusiastic male tone, like an encouraging older brother: {text}"

            response = client.models.generate_content(
                model="gemini-3.1-flash-tts-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    speech_config=types.SpeechConfig(
                        language_code=language,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Charon"  # Charon is a great male voice
                            )
                        )
                    )
                )
            )
            
            # Extract raw PCM data
            pcm_data = response.candidates[0].content.parts[0].inline_data.data
            
            # We must convert raw PCM to WAV bytes so Streamlit can play it
            import wave
            import io
            
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm_data)
                
            audio_bytes = wav_io.getvalue()
            logger.info("Gemini-TTS generated successfully with Charon voice")
            return audio_bytes

        except Exception as e:
            logger.error(f"Gemini-TTS generation failed: {e}")
            return b""
