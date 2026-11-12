# Deploy SpeechMind on Hugging Face Spaces

This deployment uses a Docker Space because SpeechMind needs a persistent Streamlit process, local audio decoding and separate Python environments for the main models and Qwen/IndicTrans2.

## Recommended hardware

Start with CPU Basic (2 vCPU, 16 GB RAM). It avoids hourly GPU charges and has enough memory headroom for the current CPU-first model stack, but inference will be slow. The Qwen Hinglish model may take a minute or more for a short recording.

Hugging Face currently requires an HF Pro subscription to host Docker Spaces on compute. Dedicated GPU hardware is optional and billed separately. Check current pricing before enabling a GPU.

## 1. Create the Space

1. Sign in to Hugging Face and open https://huggingface.co/new-space.
2. Name it `speechmind`.
3. Select **Docker** as the Space SDK.
4. Choose **CPU Basic** initially.
5. Choose Public or Private visibility. Public Spaces are accessible to anyone with the URL.

The root `README.md` in this repository already contains the Docker Space metadata and `app_port: 7860`.

## 2. Push the application

From the SpeechMind project folder, after pulling the latest GitHub branch:

```powershell
git pull --ff-only origin feature/speechmind-product-architecture-v2
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/speechmind
git push hf HEAD:main
```

Replace `YOUR_HF_USERNAME` with your Hugging Face username. If the `hf` remote already exists, do not add it again; use `git remote set-url hf https://huggingface.co/spaces/YOUR_HF_USERNAME/speechmind`.

## 3. Upload the trained checkpoints

Model weights and local checkpoints are intentionally excluded from Git. For the hosted app to use the same trained emotion models as the local app, create a **private Model repository** on Hugging Face, for example:

`YOUR_HF_USERNAME/speechmind-models`

Upload these directories while preserving their paths:

- `checkpoints/text_emotion_muril/`
- `checkpoints/voice_emotion_multicorpus_cpu/`

The voice folder must include `speechmind_metadata.json` and `voice_emotion_head.joblib`. The text folder must include the metadata, tokenizer and model weight files required by Transformers.

From the SpeechMind project folder, authenticate with Hugging Face and upload the checkpoint directory:

```powershell
hf auth login
hf upload YOUR_HF_USERNAME/speechmind-models checkpoints --repo-type model
```

Do not upload datasets, raw recordings, temporary files or unrelated checkpoints.

## 4. Configure Space secrets and variables

In the Space, open **Settings**.

Add this runtime variable:

- `SPEECHMIND_MODEL_REPO` = `YOUR_HF_USERNAME/speechmind-models`

Add this secret:

- `HF_TOKEN` = a Hugging Face token with **read** access to the private model repository.

Never commit the token to Git or put it in the README.

The app downloads the trained checkpoint directories at startup. If the model repository is not configured, the app can still start, but some trained emotion branches will be unavailable or will use their documented fallback.

## 5. Verify the deployment

After the Space finishes building:

1. Open the Space URL.
2. Upload a short, non-sensitive Hinglish recording.
3. Confirm that the original transcript and English meaning are separate.
4. Confirm that text and voice score distributions are visible.
5. Confirm that disagreement does not produce a combined emotion label.
6. Expand performance diagnostics and verify the selected ASR model.

## Current hosted limitations

- CPU Basic is memory-suitable but slower than a GPU deployment.
- IndicTrans2 translation runs through the isolated Linux worker; first use downloads its model.
- A public app should use non-sensitive demo audio only.
- Hinglish ASR and emotion predictions remain experimental. The optional recognition-hints field can supply technical vocabulary such as product names or words like "layout".
- The current emotion models are not validated sufficiently to make claims about a speaker's true emotional state.
