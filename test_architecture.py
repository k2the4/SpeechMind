"""
SpeechMind - Architecture Pre-Flight Check

Run this BEFORE fine-tuning or testing with real data. It verifies:
    1. Every model actually loads (catches wrong repo names immediately)
    2. Every component's output shape/keys match what downstream code expects
    3. The voice-emotion label space and text-emotion label space are
       compatible with fusion_layer.py's VOICE_TO_UNIFIED mapping (no
       label silently falls through unmapped)

This uses tiny synthetic inputs (silence / dummy text) - it is NOT testing
accuracy, only wiring. Run it once now, fix anything it flags, then don't
change model checkpoints again before fine-tuning.

Usage:
    python test_architecture.py
"""
import sys
import numpy as np

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, PASS, ""))
    except Exception as e:
        results.append((name, FAIL, str(e)))


def warn(name, message):
    results.append((name, WARN, message))


# ============================================================
# 1. ASR
# ============================================================

def test_asr():
    from models.asr import SpeechRecognizer
    asr = SpeechRecognizer()
    assert asr.model_name in ("openai/whisper-small",), (
        f"Unexpected ASR model loaded: {asr.model_name}"
    )
    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    result = asr.transcribe(dummy_audio)
    assert "language" in result and "transcript" in result and "confidence" in result, (
        f"ASR output missing expected keys: {result.keys()}"
    )
    assert isinstance(result["confidence"], float), "ASR confidence must be a float"
    print(f"    -> ASR loaded: {asr.model_name}, dummy output: {result}")


check("ASR (asr.py)", test_asr)


# ============================================================
# 2. Text Emotion
# ============================================================

TEXT_EMOTION_LABELS = None

def test_text_emotion():
    global TEXT_EMOTION_LABELS
    from models.text_emotion import TextEmotionClassifier
    clf = TextEmotionClassifier()
    result = clf.predict("यह एक परीक्षण वाक्य है")
    assert "emotion" in result and "confidence" in result and "distribution" in result, (
        f"TextEmotion output missing expected keys: {result.keys()}"
    )
    TEXT_EMOTION_LABELS = set(result["distribution"].keys())
    print(f"    -> Text emotion labels: {sorted(TEXT_EMOTION_LABELS)}")


check("Text Emotion (text_emotion.py)", test_text_emotion)


# ============================================================
# 3. Voice Emotion
# ============================================================

VOICE_EMOTION_LABELS = None

def test_voice_emotion():
    global VOICE_EMOTION_LABELS
    from models.voice_emotion import VoiceEmotionRecognizer
    rec = VoiceEmotionRecognizer()
    dummy_audio = np.zeros(16000, dtype=np.float32)
    result = rec.predict(dummy_audio)
    assert "emotion" in result and "confidence" in result and "arousal" in result, (
        f"VoiceEmotion output missing expected keys: {result.keys()}"
    )
    if "distribution" in result:
        VOICE_EMOTION_LABELS = set(result["distribution"].keys())
        print(f"    -> Voice emotion labels: {sorted(VOICE_EMOTION_LABELS)}")
    else:
        warn("Voice Emotion distribution", "No 'distribution' key returned - fusion will use one-hot fallback")


check("Voice Emotion (voice_emotion.py)", test_voice_emotion)


# ============================================================
# 4. Fusion label-space alignment (the critical check)
# ============================================================

def test_fusion_label_alignment():
    from fusion_layer import VOICE_TO_UNIFIED

    if VOICE_EMOTION_LABELS is None:
        warn("Fusion label alignment", "Skipped - voice emotion distribution unavailable")
        return

    unmapped = [lbl for lbl in VOICE_EMOTION_LABELS if lbl not in VOICE_TO_UNIFIED]
    assert not unmapped, (
        f"Voice emotion labels NOT in VOICE_TO_UNIFIED (will silently default to 'neutral' "
        f"in fusion instead of being counted correctly): {unmapped}"
    )

    if TEXT_EMOTION_LABELS is not None:
        mapped_voice_targets = set(VOICE_TO_UNIFIED.get(lbl) for lbl in VOICE_EMOTION_LABELS)
        missing_from_text = mapped_voice_targets - TEXT_EMOTION_LABELS
        assert not missing_from_text, (
            f"Voice emotions map to unified labels the text-emotion model doesn't "
            f"produce: {missing_from_text} - fusion scores for these will always be 0 "
            f"from the text branch"
        )

    print(f"    -> All voice labels {sorted(VOICE_EMOTION_LABELS)} map cleanly into fusion")


check("Fusion label alignment (fusion_layer.py)", test_fusion_label_alignment)


# ============================================================
# 5. Full pipeline wiring (dummy audio, no real transcription expected)
# ============================================================

def test_full_pipeline():
    from pipeline import SpeechMindPipeline
    import soundfile as sf
    import tempfile, os

    # Write 1 second of silence to a temp wav file - this is a WIRING test,
    # not an accuracy test. We only care that every stage executes and
    # returns the expected keys.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, np.zeros(16000, dtype=np.float32), 16000)
        tmp_path = tmp.name

    try:
        pipeline = SpeechMindPipeline(load_translator=False)  # translator optional right now
        result = pipeline.run(tmp_path)
        expected_keys = {
            "language", "transcript", "text_emotion", "voice_emotion",
            "valence", "arousal", "overall_interpretation", "confidence",
        }
        missing = expected_keys - set(result.keys())
        assert not missing, f"Pipeline output missing expected keys: {missing}"
        print(f"    -> Full pipeline ran, keys present: {sorted(result.keys())}")
    finally:
        os.unlink(tmp_path)


check("Full pipeline wiring (pipeline.py)", test_full_pipeline)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("SpeechMind Architecture Pre-Flight Report")
print("=" * 60)
any_fail = False
for name, status, detail in results:
    print(f"[{status}] {name}")
    if detail:
        print(f"       {detail}")
    if status == FAIL:
        any_fail = True
print("=" * 60)

if any_fail:
    print("\n❌ Fix the FAIL items above before running fine-tuning or real data tests.")
    sys.exit(1)
else:
    print("\n✅ All checks passed. Architecture is aligned - safe to proceed to fine-tuning.")
