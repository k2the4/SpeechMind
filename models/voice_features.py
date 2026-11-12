"""Shared frozen-WavLM embedding extraction used by training and inference."""
from __future__ import annotations

import numpy as np

POOLING_VERSION = "last4_layer_mean_global_mean_std_q10_q90_v2"
EXPECTED_FEATURE_DIMENSION = 3072


def pool_hidden_states(hidden_states: np.ndarray) -> np.ndarray:
    """Pool [frames, hidden_size] states with robust global statistics."""
    values = np.asarray(hidden_states, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Expected non-empty [frames, hidden_size] hidden states.")
    stats = [
        values.mean(axis=0),
        values.std(axis=0),
        np.quantile(values, 0.10, axis=0),
        np.quantile(values, 0.90, axis=0),
    ]
    return np.concatenate(stats).astype(np.float32, copy=False)


def extract_wavlm_embedding(model, feature_extractor, waveform, device) -> np.ndarray:
    """Extract one deterministic 3072-d embedding from mono 16 kHz audio."""
    import torch
    audio = np.asarray(waveform, dtype=np.float32).squeeze()
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("WavLM expects non-empty mono audio.")
    encoded = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = encoded["input_values"].to(device)
    with torch.inference_mode():
        output = model(input_values=inputs, output_hidden_states=True)
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) < 4:
            raise RuntimeError("WavLM did not return the final four hidden-state layers.")
        top_layers = torch.stack(tuple(hidden_states[-4:]), dim=0).mean(dim=0)
        frame_states = top_layers[0].float().cpu().numpy()
    embedding = pool_hidden_states(frame_states)
    if embedding.shape != (EXPECTED_FEATURE_DIMENSION,):
        raise ValueError(
            f"Unexpected WavLM feature size {embedding.shape}; "
            f"expected ({EXPECTED_FEATURE_DIMENSION},)."
        )
    return embedding


def rank_top_two_scores(distribution: dict) -> tuple[list[dict], float | None]:
    """Return the two highest raw class scores and their uncalibrated margin."""
    if not isinstance(distribution, dict) or not distribution:
        return [], None
    ranked = []
    for label, score in distribution.items():
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            ranked.append((str(label), value))
    ranked.sort(key=lambda item: item[1], reverse=True)
    if not ranked:
        return [], None
    top_two = [
        {"label": label, "score": round(score, 4)}
        for label, score in ranked[:2]
    ]
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0
    return top_two, round(float(margin), 4)
