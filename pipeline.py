"""
SpeechMind - End-to-end Pipeline

Flow:

    Audio
      ↓
    Audio preprocessing
      ↓
    IndicWhisper
      ↓
    Language + Transcript + ASR confidence
      ↓
    ┌───────────────────────┐
    │                       │
    ▼                       ▼
Text Emotion            Voice Emotion
    │                       │
    └───────────┬───────────┘
                ▼
          Late Fusion
                │
                ▼
        Final Emotion State
                │
                ▼
        English Translation
                │
                ▼
           Final Result

The text and voice branches are executed in parallel
after ASR to reduce processing time.

If the translation model can't be loaded (missing dependency, no
internet on first download, etc.), the pipeline degrades gracefully
and shows the original-language transcript instead of crashing.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from utils.audio_utils import preprocess

from models.asr import SpeechRecognizer
from models.text_emotion import TextEmotionClassifier
from models.voice_emotion import VoiceEmotionRecognizer
from models.translation import Translator

from fusion_layer import fuse


class SpeechMindPipeline:

    def __init__(self, load_translator: bool = True):

        print(
            "[Pipeline] Loading models "
            "(this can take a minute on first run)..."
        )

        # --------------------------------------------------
        # Speech recognition + language identification
        # --------------------------------------------------

        self.asr = SpeechRecognizer()

        # --------------------------------------------------
        # Text branch
        # --------------------------------------------------

        self.text_emotion = TextEmotionClassifier()

        # --------------------------------------------------
        # Voice branch
        # --------------------------------------------------

        self.voice_emotion = VoiceEmotionRecognizer()

        # --------------------------------------------------
        # Optional English translation
        #
        # This is allowed to fail without taking down the whole
        # pipeline - e.g. IndicTransToolkit isn't installed yet,
        # or the model can't be downloaded. In that case we fall
        # back to showing the original-language transcript.
        # --------------------------------------------------

        self.translator = None

        if load_translator:
            try:
                self.translator = Translator()
            except Exception as error:
                print(
                    "[Pipeline] WARNING: Translator could not be loaded "
                    f"({error}). Continuing without English translation - "
                    "the original-language transcript will be shown instead."
                )
                self.translator = None

        print("[Pipeline] All models loaded.")

    # ======================================================
    # MAIN PIPELINE
    # ======================================================

    def run(self, audio_path: str) -> dict:

        start_time = time.time()

        # --------------------------------------------------
        # Stage 0: Audio preprocessing
        # --------------------------------------------------

        print("[Pipeline] Preprocessing audio...")

        audio = preprocess(audio_path)

        # --------------------------------------------------
        # Stage 1: Speech recognition + language detection
        # --------------------------------------------------

        print("[Pipeline] Running speech recognition...")

        asr_result = self.asr.transcribe(audio)

        transcript = asr_result.get(
            "transcript",
            ""
        )

        language = asr_result.get(
            "language",
            "unknown"
        )

        asr_confidence = asr_result.get(
            "confidence",
            0.0
        )

        # Make sure confidence is a valid number.
        try:
            asr_confidence = float(asr_confidence)
        except (TypeError, ValueError):
            asr_confidence = 0.0

        asr_confidence = max(
            0.0,
            min(1.0, asr_confidence)
        )

        print(
            f"[Pipeline] Language: {language}"
        )

        print(
            f"[Pipeline] Transcript: {transcript}"
        )

        print(
            f"[Pipeline] ASR confidence: "
            f"{asr_confidence:.3f}"
        )

        # --------------------------------------------------
        # Stage 2: Text + Voice branches
        #
        # These are independent after ASR, so they run
        # concurrently.
        # --------------------------------------------------

        print(
            "[Pipeline] Running text and voice analysis..."
        )

        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:

            text_future = executor.submit(
                self.text_emotion.predict,
                transcript
            )

            voice_future = executor.submit(
                self.voice_emotion.predict,
                audio
            )

            text_result = text_future.result()
            voice_result = voice_future.result()

        print(
            "[Pipeline] Text emotion:",
            text_result.get("emotion", "unknown")
        )

        print(
            "[Pipeline] Voice emotion:",
            voice_result.get("emotion", "unknown")
        )

        # --------------------------------------------------
        # Stage 3: Late fusion
        # --------------------------------------------------

        print("[Pipeline] Running fusion...")

        fused = fuse(
            text_result=text_result,
            voice_result=voice_result,
            asr_confidence=asr_confidence
        )

        # --------------------------------------------------
        # Stage 4: English translation
        #
        # For the current version, this translates the
        # transcript into English for display.
        #
        # If self.translator is None (not installed, or failed
        # to load), we just show the original transcript instead
        # of crashing the app.
        # --------------------------------------------------

        print("[Pipeline] Preparing English meaning...")

        if self.translator is not None:

            try:

                english_meaning = (
                    self.translator.to_english(
                        transcript,
                        language
                    )
                )

            except Exception as error:

                print(
                    "[Pipeline] Translation failed:",
                    error
                )

                # Fallback to original transcript.
                english_meaning = transcript

        else:

            english_meaning = transcript

        # --------------------------------------------------
        # Stage 5: Processing time
        # --------------------------------------------------

        elapsed = round(
            time.time() - start_time,
            2
        )

        # --------------------------------------------------
        # Final result
        #
        # These keys are deliberately aligned with app.py.
        # --------------------------------------------------

        result = {

            # Speech recognition
            "language": language,

            "transcript": transcript,

            "asr_confidence": round(
                asr_confidence,
                3
            ),

            # English display
            "meaning_en": english_meaning,

            # Text emotion
            "text_emotion": fused[
                "text_emotion"
            ],

            # Voice emotion
            "voice_emotion": fused[
                "voice_emotion"
            ],

            # Combined emotional state
            "valence": fused[
                "valence"
            ],

            "arousal": fused[
                "arousal"
            ],

            "overall_interpretation": fused[
                "overall_interpretation"
            ],

            "confidence": fused[
                "confidence"
            ],

            # Warning consumed by app.py
            "low_confidence_warning": fused[
                "low_confidence_warning"
            ],

            # Useful for debugging/evaluation
            "overall_emotion": fused[
                "overall_emotion"
            ],

            "fusion_weights": fused[
                "fusion_weights"
            ],

            "processing_time_sec": elapsed,
        }

        return result


# ==========================================================
# COMMAND-LINE TEST
# ==========================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage: python pipeline.py "
            "<path_to_audio_file>"
        )

        sys.exit(1)

    audio_file = sys.argv[1]

    pipeline = SpeechMindPipeline()

    output = pipeline.run(
        audio_file
    )

    print("\n" + "=" * 50)
    print("SpeechMind Result")
    print("=" * 50)

    for key, value in output.items():

        print(
            f"{key:25s}: {value}"
        )
