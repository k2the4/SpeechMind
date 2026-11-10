# Optional Bodhan provider setup

SpeechMind keeps its local ASR and translation providers as the default. The Bodhan providers are optional because they send data to an external service and are metered.

## What the optional provider does

- Bodhan Indic-Transcribe: automatic multilingual/code-mixed speech recognition.
- Bodhan Indic-Translate: converts the original mixed-language transcript into natural English.
- SpeechMind still runs its local WavLM voice-emotion model and existing text-emotion branch.

The hosted ASR API currently accepts up to 30 seconds per request. The adapter rejects longer recordings instead of silently truncating or splitting speech at arbitrary word boundaries.

The API returns transcript text but does not provide a calibrated transcript-correctness probability. SpeechMind therefore reports ASR quality as unavailable for this provider. Whisper is used locally only for a primary-language hint; that hint does not describe all language switches.

## Create an API key

1. Open https://console.bodhan.ai/
2. Create/sign in to an account.
3. Create an API key for Indic-Transcribe and Indic-Translate.
4. Review the current pricing, data handling, and terms before sending any personal or sensitive recordings.

The published pricing page currently lists ₹0.10 per audio minute for transcription and ₹0.20 per 10,000 output tokens for translation. Pricing and API terms can change; check the provider dashboard before use.

## Launch on Windows

1. Pull the SpeechMind branch containing this change.
2. Install updated requirements in the project's Python environment:
   python -m pip install -r requirements.txt
3. Double-click RUN_SPEECHMIND_BODHAN.bat.
4. Enter the API key into the hidden PowerShell prompt.
5. The launcher sets the provider variables for that process only and starts Streamlit.

The key is not written into a file or permanently stored by the launcher. Closing the launched process clears the process environment variables.

## Return to local providers

Start SpeechMind using the existing local launcher, or make sure these variables are not set:
- SPEECHMIND_ASR_PROVIDER
- SPEECHMIND_TRANSLATOR_PROVIDER
- BODHAN_API_KEY

The local provider remains the default when the provider variables are absent.

## Privacy and cost

When Bodhan mode is enabled:
- The audio recording is uploaded to Bodhan for transcription.
- The transcript is sent to Bodhan for English translation.
- API requests incur metered charges.
- Do not use private, confidential, medical, financial, or otherwise sensitive recordings unless the speaker has consented and the provider's current data-handling terms have been reviewed.

This integration is opt-in. It is not activated by default and no API key is included in the repository.
