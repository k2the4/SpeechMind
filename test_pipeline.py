from pathlib import Path
from pipeline import SpeechMindPipeline

ROOT = Path(r"C:\Users\kvasi\OneDrive\Desktop\Test Data\Audio_Speech_Actors_01-24_16k")

# Pick 5 different WAV files
files = list(ROOT.rglob("*.wav"))[:5]

print("=" * 60)
print("SPEECHMIND FULL PIPELINE TEST")
print("=" * 60)

pipeline = SpeechMindPipeline()

for i, path in enumerate(files, 1):
    print("\n" + "=" * 60)
    print(f"TEST {i}/5")
    print(f"File: {path.name}")
    print("=" * 60)

    try:
        result = pipeline.run(str(path))

        print(f"Language:       {result.get('language')}")
        print(f"Transcript:     {result.get('transcript')}")
        print(f"Text emotion:   {result.get('text_emotion')}")
        print(f"Voice emotion:  {result.get('voice_emotion')}")
        print(f"Valence:        {result.get('valence')}")
        print(f"Arousal:        {result.get('arousal')}")
        print(f"Overall:        {result.get('overall_emotion')}")
        print(f"Confidence:     {result.get('confidence')}")
        print(f"Processing:     {result.get('processing_time_sec')} sec")

    except Exception as e:
        print("FAILED:")
        print(e)

print("\n" + "=" * 60)
print("PIPELINE TEST COMPLETE")
print("=" * 60)