"""
SpeechMind - Speech Recognition + Language Identification.

Whisper is retained as the language-identification and non-Indic ASR model.
IndicConformer is loaded only when a supported Indic language is detected.
This avoids loading a 600M-parameter model for English recordings.
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


class SpeechRecognizer:
    def __init__(self):
        self.model_name = f"openai/whisper-{WHISPER_MODEL}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ASR] Loading Whisper-{WHISPER_MODEL}...")
        self.whisper = whisper.load_model(WHISPER_MODEL, device=str(self.device))
        self.indic_model = None
        print(f"[ASR] Whisper loaded on {self.device}; IndicConformer will load on demand.")

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

    def transcribe(self, audio) -> dict:
        if isinstance(audio, torch.Tensor):
            audio_numpy = audio.detach().cpu().numpy()
        else:
            audio_numpy = audio
        audio_numpy = audio_numpy.squeeze()
        if audio_numpy.size == 0:
            raise ValueError("ASR received empty audio.")

        # Keep the existing single Whisper pass for language detection. For
        # Indic speech, its draft transcript is discarded and IndicConformer
        # supplies the final transcript.
        print("[ASR] Detecting language with Whisper...")
        whisper_result = self.whisper.transcribe(
            audio_numpy, task="transcribe", fp16=False, verbose=False
        )
        detected_language = str(
            whisper_result.get("language", "unknown")
        ).lower().strip()
        print(f"[ASR] Whisper detected: {detected_language}")

        indic_language = WHISPER_TO_INDIC.get(detected_language)
        if detected_language in ("en", "english"):
            transcript = whisper_result.get("text", "").strip()
            return {
                "transcript": transcript,
                "language": "en",
                "confidence": self._estimate_whisper_confidence(whisper_result),
                "asr_model": "whisper-small",
            }

        if indic_language in INDIC_LANGUAGES:
            print(f"[ASR] Using IndicConformer for {INDIC_LANGUAGES[indic_language]}.")
            transcript = self._indic_transcribe(audio, indic_language)
            return {
                "transcript": transcript,
                "language": indic_language,
                "confidence": self._estimate_indic_confidence(transcript),
                "asr_model": "indic-conformer-600m",
            }

        return {
            "transcript": whisper_result.get("text", "").strip(),
            "language": detected_language,
            "confidence": self._estimate_whisper_confidence(whisper_result),
            "asr_model": "whisper-small",
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
