from utils.ffmpeg_setup import *
from datasets import load_dataset
import soundfile as sf
import os

OUTPUT_DIR = r"C:\Users\kvasi\OneDrive\Desktop\Test Data\Hindi"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Hindi validation data...")
dataset = load_dataset(
    "ai4bharat/IndicVoices",
    "hindi",
    split="valid",
    streaming=True
)

print("Downloading 3 Hindi samples...")

for i, sample in enumerate(dataset):
    audio = sample["audio_filepath"]

    # Audio is already decoded by Hugging Face
    sf.write(
        os.path.join(OUTPUT_DIR, f"hindi_test_{i+1}.wav"),
        audio["array"],
        audio["sampling_rate"]
    )

    print(f"\n{i+1}. Saved: hindi_test_{i+1}.wav")
    print(f"   Transcript: {sample['text']}")
    print(f"   Duration: {sample['duration']:.2f}s")

    if i >= 2:
        break

print("\nDONE")
print(f"Files saved to: {OUTPUT_DIR}")