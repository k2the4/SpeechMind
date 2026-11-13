"""Paired ASR benchmark for one or more recordings against a human reference.

Run from the project root:
  .venv\\Scripts\\python.exe scripts\\compare_asr_reference.py --audio "female.wav" --audio "male.wav"

The script sends the same preprocessed 16 kHz mono waveform to each selected
recognizer, supports long Srota recordings through provider-side chunking, and
writes one row per recording/model to a CSV report.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import unicodedata
from pathlib import Path

# When launched by path, Python puts scripts/ rather than the project root on
# sys.path. Add the root so normal PowerShell invocations resolve package imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import whisper
from transformers import AutoModel

from models.srota_provider import SrotaASRProvider
from models.hinglish_context import infer_hinglish_context_from_audio_tail
from utils.audio_utils import preprocess


DEFAULT_REFERENCE = (
    "मुझे इस project के लिए एक नया डिजाइन बनाना है "
    "but i am not sure about the layout"
)
CSV_FIELDS = [
    "speaker", "audio_file", "audio_duration_seconds", "model", "model_id",
    "reference", "hypothesis", "WER", "CER", "inference_seconds",
    "initialization_seconds", "elapsed_seconds", "device", "status", "error",
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or "")).casefold()
    text = re.sub(r"[^\\w\\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_token in enumerate(reference, start=1):
        current = [row]
        for col, hyp_token in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[col] + 1,
                previous[col - 1] + (ref_token != hyp_token),
            ))
        previous = current
    return previous[-1]


def error_rates(reference: str, hypothesis: str) -> dict[str, float]:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    ref_words = ref.split()
    hyp_words = hyp.split()
    wer = edit_distance(ref_words, hyp_words) / max(1, len(ref_words))
    cer = edit_distance(list(ref), list(hyp)) / max(1, len(ref))
    return {"WER": wer, "CER": cer}


def detect_whisper_language(audio: np.ndarray, model) -> tuple[str, float, dict]:
    """Run Whisper's encoder language classifier without autoregressive decoding."""
    padded = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(
        padded, n_mels=model.dims.n_mels
    ).to(model.device)
    _, probabilities = model.detect_language(mel)
    language = max(probabilities, key=probabilities.get)
    normalized = {
        str(code).lower().strip(): float(score)
        for code, score in probabilities.items()
    }
    return str(language).lower().strip(), float(probabilities[language]), normalized


def make_row(
    speaker: str,
    audio_path: Path,
    duration: float,
    model: str,
    model_id: str,
    reference: str,
    hypothesis: str,
    inference_seconds: float | None,
    initialization_seconds: float | None,
    elapsed_seconds: float | None,
    device: str,
    status: str = "ok",
    error: str = "",
) -> dict:
    scores = error_rates(reference, hypothesis)
    return {
        "speaker": speaker,
        "audio_file": str(audio_path),
        "audio_duration_seconds": round(duration, 3),
        "model": model,
        "model_id": model_id,
        "reference": reference,
        "hypothesis": hypothesis,
        "WER": round(scores["WER"], 6),
        "CER": round(scores["CER"], 6),
        "inference_seconds": (
            round(float(inference_seconds), 3)
            if inference_seconds is not None else ""
        ),
        "initialization_seconds": (
            round(float(initialization_seconds), 3)
            if initialization_seconds is not None else ""
        ),
        "elapsed_seconds": (
            round(float(elapsed_seconds), 3)
            if elapsed_seconds is not None else ""
        ),
        "device": device or "unknown",
        "status": status,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio", action="append",
        help="Audio path. Repeat this option to benchmark multiple recordings.",
    )
    parser.add_argument(
        "--speaker", action="append",
        help="Speaker label matching each --audio argument, in the same order.",
    )
    parser.add_argument(
        "--reference", default=DEFAULT_REFERENCE,
        help="Correct human transcript shared by all supplied recordings.",
    )
    parser.add_argument(
        "--context", default=None,
        help="Optional task context for Srota only; omit for baseline comparison.",
    )
    parser.add_argument(
        "--srota-model",
        default=SrotaASRProvider.MODEL_NAME,
        help="Srota model ID.",
    )
    parser.add_argument(
        "--whisper-languages",
        nargs="+",
        choices=("auto", "hi", "en"),
        default=("auto", "hi"),
        help="Whisper-small decoding modes. Default: auto hi.",
    )
    parser.add_argument(
        "--skip-whisper", action="store_true",
        help="Skip Whisper comparison transcripts.",
    )
    parser.add_argument(
        "--include-indicconformer", action="store_true",
        help="Also run IndicConformer-600M in forced-Hindi mode.",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0,
        help="Srota worker request timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--no-auto-context", action="store_true",
        help="Disable the automatic Whisper tail vocabulary scout.",
    )
    parser.add_argument(
        "--csv-output",
        default="data/benchmarks/asr_comparison.csv",
        help="CSV output path (default: data/benchmarks/asr_comparison.csv).",
    )
    parser.add_argument(
        "--append-csv", action="store_true",
        help="Append rows to an existing CSV instead of replacing it.",
    )
    args = parser.parse_args()

    audio_values = args.audio or [
        input("Enter the audio file path: ").strip().strip('"')
    ]
    audio_paths = [Path(value).expanduser().resolve() for value in audio_values]
    for audio_path in audio_paths:
        if not audio_path.is_file():
            parser.error(f"Audio file not found: {audio_path}")
    if args.speaker and len(args.speaker) != len(audio_paths):
        parser.error("Provide exactly one --speaker label for each --audio file.")

    speakers = args.speaker or [path.stem for path in audio_paths]
    rows: list[dict] = []
    whisper_model = None
    whisper_initialization_seconds = 0.0
    whisper_device = "not-run"
    if not args.skip_whisper or (
        args.context is None and not args.no_auto_context
    ):
        whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
        started = time.perf_counter()
        whisper_model = whisper.load_model("small", device=whisper_device)
        whisper_initialization_seconds = time.perf_counter() - started
        whisper_device = str(whisper_model.device)

    indic_model = None
    indic_device = "not-run"
    indic_initialization_seconds = 0.0
    if args.include_indicconformer:
        indic_device = "cuda" if torch.cuda.is_available() else "cpu"
        started = time.perf_counter()
        indic_model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True,
        )
        indic_model.eval().to(indic_device)
        indic_initialization_seconds = time.perf_counter() - started

    srota = SrotaASRProvider(
        model_name=args.srota_model,
        request_timeout=args.timeout,
    )
    try:
        for audio_path, speaker in zip(audio_paths, speakers):
            print(f"\\n[ASR benchmark] Speaker: {speaker}")
            print(f"[ASR benchmark] Audio: {audio_path}")
            print("[ASR benchmark] Loading the same 16 kHz mono waveform for all models...")
            audio = preprocess(str(audio_path))
            if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.size == 0:
                raise RuntimeError("Audio preprocessing did not produce a non-empty mono waveform.")
            duration = audio.size / 16000.0
            print(f"[ASR benchmark] Duration: {duration:.2f}s")

            whisper_results = {}
            if not args.skip_whisper and whisper_model is not None:
                for mode in args.whisper_languages:
                    language = None if mode == "auto" else mode
                    label = "automatic language" if mode == "auto" else f"forced {mode}"
                    print(f"[ASR benchmark] Running Whisper-small ({label})...")
                    started = time.perf_counter()
                    result = whisper_model.transcribe(
                        audio,
                        task="transcribe",
                        language=language,
                        fp16=False,
                        verbose=False,
                        beam_size=5,
                        condition_on_previous_text=False,
                    )
                    elapsed = time.perf_counter() - started
                    transcript = str(result.get("text") or "").strip()
                    whisper_results[mode] = transcript
                    rows.append(make_row(
                        speaker, audio_path, duration,
                        f"Whisper-small / {mode}",
                        "openai/whisper-small", args.reference, transcript,
                        elapsed, whisper_initialization_seconds, elapsed,
                        whisper_device,
                    ))
                    print(transcript or "[empty transcript]")
                    print(error_rates(args.reference, transcript))

            indic_text = ""
            if indic_model is not None:
                print("[ASR benchmark] Running IndicConformer-600M (forced Hindi)...")
                waveform = torch.tensor(
                    audio, dtype=torch.float32
                ).unsqueeze(0).to(indic_device)
                started = time.perf_counter()
                with torch.inference_mode():
                    indic_result = indic_model(waveform, "hi", "ctc")
                indic_elapsed = time.perf_counter() - started
                if isinstance(indic_result, (list, tuple)):
                    indic_result = indic_result[0] if indic_result else ""
                indic_text = str(indic_result).strip()
                rows.append(make_row(
                    speaker, audio_path, duration,
                    "IndicConformer-600M / forced hi",
                    "ai4bharat/indic-conformer-600m-multilingual",
                    args.reference, indic_text, indic_elapsed,
                    indic_initialization_seconds, indic_elapsed, indic_device,
                ))
                print(indic_text or "[empty transcript]")
                print(error_rates(args.reference, indic_text))

            selected_context = args.context
            context_info = {
                "context": selected_context or "",
                "source": "manual_context" if selected_context else "disabled",
                "probe_seconds": 0.0,
                "transcript": "",
            }
            if selected_context is None and not args.no_auto_context:
                if whisper_model is None:
                    raise RuntimeError("Whisper is required for automatic context scouting.")
                context_info = infer_hinglish_context_from_audio_tail(
                    audio,
                    whisper_model,
                    sample_rate=16000,
                    tail_seconds=3.0,
                    language_detector=lambda tail: detect_whisper_language(
                        tail, whisper_model
                    ),
                    min_english_score=0.55,
                )
                selected_context = context_info["context"] or None
            print(
                "[ASR benchmark] Srota context: "
                + (selected_context if selected_context else "[none]")
                + f" ({context_info['source']}, {context_info['probe_seconds']}s)"
            )

            print(f"[ASR benchmark] Running Srota model: {args.srota_model}...")
            started = time.perf_counter()
            try:
                srota_result = srota.transcribe_audio(
                    audio, sample_rate=16000, context=selected_context
                )
                srota_elapsed = time.perf_counter() - started
                srota_text = str(srota_result.get("transcript") or "").strip()
                srota_device = str(
                    srota_result.get("worker_response", {}).get("device", "unknown")
                )
                rows.append(make_row(
                    speaker, audio_path, duration,
                    f"Srota / {args.srota_model}", args.srota_model,
                    args.reference, srota_text,
                    srota_result.get("inference_seconds"),
                    srota_result.get("worker_initialization_seconds"),
                    srota_elapsed, srota_device,
                ))
                print(srota_text or "[empty transcript]")
                print(error_rates(args.reference, srota_text))
                print(
                    f"[ASR benchmark] Srota chunks: {srota_result.get('chunk_count', 1)}; "
                    f"inference: {srota_result.get('inference_seconds', 0)}s; "
                    f"device: {srota_device}"
                )
            except Exception as error:
                srota_elapsed = time.perf_counter() - started
                message = f"{type(error).__name__}: {error}"
                rows.append(make_row(
                    speaker, audio_path, duration,
                    f"Srota / {args.srota_model}", args.srota_model,
                    args.reference, "", None, None, srota_elapsed,
                    "unknown", status="failed", error=message,
                ))
                print(f"[ASR benchmark] Srota failed: {message}")

    finally:
        srota.close()

    output_path = Path(args.csv_output).expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if args.append_csv and output_path.exists() else "w"
    write_header = write_mode == "w"
    with output_path.open(write_mode, newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\\n[ASR benchmark] Saved {len(rows)} model results to: {output_path}")
    print("[ASR benchmark] Lower WER/CER is better. These are diagnostic clip-level scores.")
    print("[ASR benchmark] Production routing was not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
