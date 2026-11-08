"""
SpeechMind multilingual ASR router.

Model weights are loaded on demand:
- A user-selected Indian language uses IndicConformer directly.
- A user-selected English input uses Whisper directly.
- Auto-detect retains Whisper language detection, then routes supported Indic
  languages to IndicConformer.

Selecting the known language avoids the unnecessary language-detection pass
and avoids loading the unused ASR model.
"""
import torch
import whisper
from transformers import AutoModel

WHISPER_MODEL = "small"
INDIC_CONFORMER_MODEL = "ai4bharat/indic-conformer-600m-multilingual"

INDIC_LANGUAGES = {
    "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri",
    "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "kok": "Konkani",
    "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam", "mni": "Manipuri",
    "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
    "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu",
}

WHISPER_TO_INDIC = {
    "assamese": "as", "as": "as", "bengali": "bn", "bn": "bn",
    "bodo": "brx", "brx": "brx", "dogri": "doi", "doi": "doi",
    "gujarati": "gu", "gu": "gu", "hindi": "hi", "hi": "hi",
    "kannada": "kn", "kn": "kn", "konkani": "kok", "kok": "kok",
    "kashmiri": "ks", "ks": "ks", "maithili": "mai", "mai": "mai",
    "malayalam": "ml", "ml": "ml", "manipuri": "mni", "mni": "mni",
    "marathi": "mr", "mr": "mr", "nepali": "ne", "ne": "ne",
    "odia": "or", "oriya": "or", "or": "or", "punjabi": "pa", "pa": "pa",
    "sanskrit": "sa", "sa": "sa", "santali": "sat", "sat": "sat",
    "sindhi": "sd", "sd": "sd", "tamil": "ta", "ta": "ta",
    "telugu": "te", "te": "te", "urdu": "ur", "ur": "ur",
}

SUPPORTED_LANGUAGE_CODES = {"en", *INDIC_LANGUAGES.keys()}


class SpeechRecognizer:
    def __init__(self):
        self.model_name = f"openai/whisper-{WHISPER_MODEL}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper = None
        self.indic_model = None
        print(
            "[ASR] Router initialized. Whisper and IndicConformer weights "
            "will load only when their route is needed."
        )

    def _load_whisper(self):
        if self.whisper is None:
            print(f"[ASR] Loading Whisper-{WHISPER_MODEL} on demand...")
            self.whisper = whisper.load_model(
                WHISPER_MODEL, device=str(self.device)
            )
            print(f"[ASR] Whisper loaded on {self.device}.")
        return self.whisper

    def _load_indic_model(self):
        if self.indic_model is None:
            print("[ASR] Loading IndicConformer-600M on demand...")
            self.indic_model = AutoModel.from_pretrained(
                INDIC_CONFORMER_MODEL,
                trust_remote_code=True,
            )
            self.indic_model.eval()
            self.indic_model.to(self.device)
            print("[ASR] IndicConformer loaded.")
        return self.indic_model

    def transcribe(self, audio, language_hint=None) -> dict:
        if isinstance(audio, torch.Tensor):
            audio_numpy = audio.detach().cpu().numpy()
        else:
            audio_numpy = audio
        audio_numpy = torch.as_tensor(audio_numpy).detach().cpu().numpy().squeeze()
        if audio_numpy.size == 0:
            raise ValueError("ASR received empty audio.")

        hint = str(language_hint).lower().strip() if language_hint else None
        if hint in ("english", "en"):
            return self._transcribe_english(audio_numpy, forced=True)
        if hint in INDIC_LANGUAGES:
            return self._transcribe_indic(audio_numpy, hint, forced=True)
        if hint is not None and hint not in ("auto", "auto-detect", "unknown"):
            raise ValueError(
                f"Unsupported language hint '{language_hint}'. "
                "Use 'en' or a supported Indic language code."
            )

        # Auto-detect mode: one Whisper decode identifies language and produces
        # a transcript. For Indic languages the draft is replaced by
        # IndicConformer; explicit language selection can bypass this pass.
        model = self._load_whisper()
        print("[ASR] Auto-detecting language with Whisper...")
        whisper_result = model.transcribe(
            audio_numpy, task="transcribe", fp16=False, verbose=False
        )
        detected_language = str(
            whisper_result.get("language", "unknown")
        ).lower().strip()
        print(f"[ASR] Whisper detected: {detected_language}")

        indic_language = WHISPER_TO_INDIC.get(detected_language)
        if indic_language in INDIC_LANGUAGES:
            return self._transcribe_indic(
                audio_numpy, indic_language, forced=False
            )

        return {
            "transcript": whisper_result.get("text", "").strip(),
            "language": "en" if detected_language == "english" else detected_language,
            "confidence": self._estimate_whisper_confidence(whisper_result),
            "asr_model": "whisper-small",
            "language_route": "auto-whisper",
            "confidence_method": "whisper-segment-logprob-heuristic",
        }

    def _transcribe_english(self, audio, forced: bool):
        model = self._load_whisper()
        result = model.transcribe(
            audio,
            language="en",
            task="transcribe",
            fp16=False,
            verbose=False,
        )
        return {
            "transcript": result.get("text", "").strip(),
            "language": "en",
            "confidence": self._estimate_whisper_confidence(result),
            "asr_model": "whisper-small",
            "language_route": "selected-english" if forced else "auto-whisper",
            "confidence_method": "whisper-segment-logprob-heuristic",
        }

    def _transcribe_indic(self, audio, language_code: str, forced: bool):
        transcript = self._indic_transcribe(audio, language_code)
        return {
            "transcript": transcript,
            "language": language_code,
            "confidence": self._estimate_indic_confidence(transcript),
            "asr_model": "indic-conformer-600m",
            "language_route": "selected-indic" if forced else "auto-whisper-to-indic",
            "confidence_method": "nonempty-transcript-heuristic",
        }

    def _indic_transcribe(self, audio, language_code):
        model = self._load_indic_model()
        if isinstance(audio, torch.Tensor):
            wav = audio.detach().clone().float()
        else:
            wav = torch.tensor(audio, dtype=torch.float32)
        wav = wav.squeeze().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            transcription = model(wav, language_code, "ctc")
        if isinstance(transcription, (list, tuple)):
            transcription = transcription[0]
        return str(transcription).strip()

    @staticmethod
    def _estimate_whisper_confidence(whisper_result):
        """Heuristic score from Whisper segment log probabilities; not calibrated."""
        segments = whisper_result.get("segments", [])
        values = [
            segment.get("avg_logprob")
            for segment in segments
            if segment.get("avg_logprob") is not None
        ]
        if not values:
            return 0.5
        average = sum(values) / len(values)
        return round(float(torch.sigmoid(torch.tensor(average)).item()), 3)

    @staticmethod
    def _estimate_indic_confidence(transcript):
        """IndicConformer wrapper currently exposes no calibrated confidence."""
        return 0.7 if transcript else 0.0
