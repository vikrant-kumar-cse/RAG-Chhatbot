import os
import base64
import requests
from dotenv import load_dotenv
from gtts import gTTS
from io import BytesIO

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


class MultilingualVoiceHandler:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")

        self.groq_api_key = GROQ_API_KEY
        self.groq_url = "https://api.groq.com/openai/v1"
        
        # Supported Indian Languages with their codes
        self.supported_languages = {
            'hindi': 'hi',
            'bengali': 'bn',
            'tamil': 'ta',
            'telugu': 'te',
            'marathi': 'mr',
            'gujarati': 'gu',
            'kannada': 'kn',
            'malayalam': 'ml',
            'punjabi': 'pa',
            'urdu': 'ur',
            'odia': 'or',
            'assamese': 'as',
            'nepali': 'ne',
            'english': 'en',
            # Additional languages
            'sanskrit': 'sa',
            'sindhi': 'sd',
            'konkani': 'gom',
            'manipuri': 'mni',
            'kashmiri': 'ks',
            'dogri': 'doi',
            'bodo': 'brx',
            'santhali': 'sat',
            'maithili': 'mai'
        }

   
    # 🔊 SPEECH → TEXT (STT) - Supports Multilingual Auto-Detection
    # ---------------------------------------------------------------
    def speech_to_text(self, audio_data, language=None):
        """
        Convert audio (base64 encoded) to text using Groq STT (Whisper)
        Whisper automatically detects the language!
        
        Args:
            audio_data: base64 encoded audio data (wav/ogg/etc.)
            language: Optional - language code for hint (e.g., 'hi', 'ta', 'en')
        Returns:
            A dict with keys: 'text', 'language', and optionally 'confidence'
            or None on failure
        """
        try:
            if not audio_data:
                print("speech_to_text: no audio_data provided")
                return None

            audio_bytes = base64.b64decode(audio_data)

            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
            }

            stt_model = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")

            files = {
                "file": ("audio.wav", audio_bytes, "audio/wav"),
                "model": (None, stt_model),
                "response_format": (None, "verbose_json"),  # Get detailed response
            }
            
            # Add language hint if provided
            if language:
                files["language"] = (None, language)

            response = requests.post(
                f"{self.groq_url}/audio/transcriptions",
                headers=headers,
                files=files,
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()

                text = None
                detected_language = None
                confidence = None

                if isinstance(result, dict):
                    text = result.get("text") or ""
                    detected_language = result.get("language")  # Whisper detects language
                    
                    # Calculate confidence from segments
                    segments = result.get("segments")
                    if isinstance(segments, list) and segments:
                        confs = [
                            s.get("confidence")
                            for s in segments
                            if isinstance(s, dict) and s.get("confidence") is not None
                        ]
                        if confs:
                            try:
                                confidence = sum(confs) / len(confs)
                            except Exception:
                                confidence = None

                if text is None and isinstance(result, str):
                    text = result

                return {
                    "text": text or "",
                    "language": detected_language,
                    "confidence": confidence
                }

            try:
                print("Error in speech_to_text:", response.status_code, response.text)
            except Exception:
                print("Error in speech_to_text: non-200 response")

            return None

        except Exception as e:
            print(f"Error converting speech to text: {str(e)}")
            return None


    # 🗣 TEXT → SPEECH (TTS) - Multilingual with gTTS
    # ------------------------------------------------
    def text_to_speech(self, text, language='en', use_groq=False):
        """
        Convert text to speech supporting 22+ Indian languages
        
        Args:
            text: Text to convert to speech
            language: Language code (e.g., 'hi', 'ta', 'en', 'bn')
            use_groq: If True, try Groq TTS first (English only), then fallback to gTTS
        Returns:
            Audio data as base64 encoded string or None on failure
        """
        try:
            if not text:
                print("text_to_speech: no text provided")
                return None

            # Option 1: Try Groq TTS first (if requested and English)
            if use_groq and language == 'en':
                groq_audio = self._groq_tts(text)
                if groq_audio:
                    return groq_audio
                print("⚠️ Groq TTS failed, falling back to gTTS...")

            # Option 2: Use gTTS for multilingual support
            return self._gtts_tts(text, language)

        except Exception as e:
            print(f"Error in text_to_speech: {str(e)}")
            return None


    def _groq_tts(self, text):
        """Groq TTS (English only, premium voices)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            }

            candidate_models = [
                ("playai-tts", "Fritz-PlayAI"),
                ("playai-tts", "Bryan-PlayAI"),
                ("playai-tts", "Aria-PlayAI"),
            ]

            response_format = os.environ.get("GROQ_TTS_FORMAT", "wav")

            for tts_model, tts_voice in candidate_models:
                try:
                    data = {
                        "model": tts_model,
                        "input": text,
                        "voice": tts_voice,
                        "response_format": response_format,
                    }

                    response = requests.post(
                        f"{self.groq_url}/audio/speech",
                        headers=headers,
                        json=data,
                        timeout=60,
                    )

                    if response.status_code == 200:
                        audio_base64 = base64.b64encode(response.content).decode("utf-8")
                        print(f"✅ Groq TTS successful: model={tts_model}, voice={tts_voice}")
                        return audio_base64
                    else:
                        continue

                except Exception as e:
                    print(f"⚠️ Groq TTS error for model '{tts_model}': {str(e)}")
                    continue

            return None

        except Exception as e:
            print(f"Error in Groq TTS: {str(e)}")
            return None


    def _gtts_tts(self, text, language='en'):
        """Google TTS - Supports 22+ Indian languages"""
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Save to BytesIO buffer
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Convert to base64
            audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
            print(f"✅ gTTS successful: language={language}")
            return audio_base64

        except Exception as e:
            print(f"Error in gTTS: {str(e)}")
            return None


    def get_supported_languages(self):
        """Return list of supported languages"""
        return self.supported_languages


    def detect_language(self, audio_data):
        """
        Detect language from audio
        Returns the detected language code
        """
        result = self.speech_to_text(audio_data)
        if result and result.get('language'):
            return result['language']
        return None


# Global instance
voice_handler = MultilingualVoiceHandler()


# Example Usage:
if __name__ == "__main__":
    # Print supported languages
    print("Supported Languages:")
    for lang_name, lang_code in voice_handler.supported_languages.items():
        print(f"  {lang_name}: {lang_code}")
    
    # Example: Generate Hindi TTS
    # hindi_audio = voice_handler.text_to_speech("नमस्ते, मैं एक चिकित्सा सहायक हूँ।", language='hi')
    # print(f"Hindi audio generated: {bool(hindi_audio)}")
    
    # Example: Generate Tamil TTS
    # tamil_audio = voice_handler.text_to_speech("வணக்கம், நான் ஒரு மருத்துவ உதவியாளர்.", language='ta')
    # print(f"Tamil audio generated: {bool(tamil_audio)}")