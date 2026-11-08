from pathlib import Path

from utils.audio_utils import preprocess
from models.voice_emotion import VoiceEmotionRecognizer


DATASET = Path(
    r"C:\Users\kvasi\OneDrive\Desktop\Test Data"
    r"\Audio_Speech_Actors_01-24_16k"
    r"\Audio_Speech_Actors_01-24_16k"
)


EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def emotion_from_filename(path):
    parts = path.stem.split("-")
    return EMOTIONS.get(parts[2], "unknown")


files = sorted(DATASET.rglob("*.wav"))[:8]

if not files:
    raise FileNotFoundError(f"No WAV files found in: {DATASET}")

print(f"Found {len(files)} test files.")
print("Loading voice-emotion model...\n")

model = VoiceEmotionRecognizer()

correct = 0

for i, path in enumerate(files, start=1):
    expected = emotion_from_filename(path)

    print("=" * 60)
    print(f"TEST {i}")
    print(f"File:     {path.name}")
    print(f"Expected: {expected}")

    try:
        audio = preprocess(str(path))
        result = model.predict(audio)

        predicted = result["emotion"]
        confidence = result["confidence"]

        print(f"Predicted: {predicted}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Raw result: {result}")

        if predicted == expected:
            correct += 1
            print("MATCH")
        else:
            print("MISMATCH")

    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("VOICE EMOTION TEST COMPLETE")
print(f"Correct: {correct}/{len(files)}")
print(f"Accuracy: {correct / len(files) * 100:.1f}%")