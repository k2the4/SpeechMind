"""
SpeechMind ASR and language identification.

Whisper performs a lightweight language-identification pass first. Supported
Indic languages go directly to IndicConformer, avoiding a complete Whisper
transcription that would otherwise be discarded. English and unsupported
languages use Whisper for transcription.
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

    def _detect_language(self, audio_numpy):
        """Identify language from Whisper's first 30-second window without decoding text."""
        padded = whisper.pad_or_trim(audio_numpy)
        mel = whisper.log_mel_spectrogram(
            padded, n_mels=self.whisper.dims.n_mels
        ).to(self.device)
        _, probabilities = self.whisper.detect_language(mel)
        language = max(probabilities, key=probabilities.get)
        return str(language).lower().strip(), float(probabilities[language])

    def _load_indic_model(self):
        if self.indic_model is None:
            print("[ASR] Loading IndicConformer-600M on demand...")
            self.indic_model = AutoModel.from_pretrained(
                INDIC_CONFORMER_MODEL, trust_remote_code=True
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
        if audio_numpy.ndim != 1 or audio_numpy.size == 0:
            raise ValueError("ASR expects non-empty mono audio.")

        try:
            detected_language, language_score = self._detect_language(audio_numpy)
        except Exception as error:
            # Keep transcription available if language-only detection fails
            # because of a Whisper-version/API difference.
            print(f"[ASR] Language-only detection failed; falling back to Whisper decode: {error}")
            whisper_result = self.whisper.transcribe(
                audio_numpy, task="transcribe", fp16=False, verbose=False
            )
            detected_language = str(whisper_result.get("language", "unknown")).lower().strip()
            language_score = None
            return self._whisper_result(whisper_result, detected_language, language_score)

        indic_language = WHISPER_TO_INDIC.get(detected_language)
        if indic_language in INDIC_LANGUAGES:
            print(f"[ASR] Detected {INDIC_LANGUAGES[indic_language]}; using IndicConformer directly.")
            transcript = self._indic_transcribe(audio_numpy, indic_language)
            return {
                "transcript": transcript,
                "language": indic_language,
                "confidence": self._estimate_indic_confidence(transcript),
                "language_detection_score": round(language_score, 3),
                "asr_model": "indic-conformer-600m",
            }

        whisper_result = self.whisper.transcribe(
            audio_numpy, task="transcribe", fp16=False, verbose=False
        )
        return self._whisper_result(
            whisper_result, detected_language, language_score
        )

    @staticmethod
    def _whisper_result(whisper_result, detected_language, language_score):
        transcript = whisper_result.get("text", "").strip()
        language = "en" if detected_language in ("en", "english") else detected_language
        result = {
            "transcript": transcript,
            "language": language,
            "confidence": SpeechRecognizer._estimate_whisper_confidence(whisper_result),
            "asr_model": "whisper-small",
        }
        if language_score is not None:
            result["language_detection_score"] = round(float(language_score), 3)
        return result

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
