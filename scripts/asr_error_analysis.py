"""Word-level ASR error attribution for model evaluation reports."""
from __future__ import annotations

from collections import Counter

from scripts.asr_metrics import (
    error_counts_multi_reference,
    normalize_asr_text,
    split_reference_alternatives,
)


def align_word_errors(reference: str, hypothesis: str) -> dict:
    """Return Levenshtein word substitutions, deletions and insertions."""
    ref = normalize_asr_text(reference).split()
    hyp = normalize_asr_text(hypothesis).split()
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            substitution_cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + substitution_cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    substitutions = Counter()
    deletions = Counter()
    insertions = Counter()
    i, j = len(ref), len(hyp)
    while i or j:
        if (
            i and j and ref[i - 1] == hyp[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            i -= 1
            j -= 1
            continue
        # On equal-cost paths, prefer a boundary insertion/deletion over a
        # substitution. This keeps alignments such as "one two three" vs
        # "one too" as a substitution (two -> too) plus trailing deletion.
        if i and dp[i][j] == dp[i - 1][j] + 1:
            deletions[ref[i - 1]] += 1
            i -= 1
            continue
        if j and dp[i][j] == dp[i][j - 1] + 1:
            insertions[hyp[j - 1]] += 1
            j -= 1
            continue
        if i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions[(ref[i - 1], hyp[j - 1])] += 1
            i -= 1
            j -= 1
            continue
        raise RuntimeError("Could not backtrace ASR word alignment.")

    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "word_errors": dp[-1][-1],
    }


def _top(counter: Counter, limit: int) -> list[dict]:
    items = counter.most_common(limit)
    return [
        {"reference_token": key[0], "hypothesis_token": key[1], "count": count}
        if isinstance(key, tuple)
        else {"token": key, "count": count}
        for key, count in items
    ]


def build_word_error_profile(
    rows: list[dict],
    hypothesis_key: str,
    status_key: str,
    limit: int = 20,
) -> dict:
    """Aggregate token errors on successfully transcribed rows only.

    For multi-reference utterances, use the alternative selected by the same
    WER scoring utility that produced the headline metric.
    """
    substitutions = Counter()
    deletions = Counter()
    insertions = Counter()
    scored_samples = 0

    for row in rows:
        if row.get(status_key) != "ok":
            continue
        reference = str(row.get("reference_transcript") or "")
        hypothesis = str(row.get(hypothesis_key) or "")
        metric = error_counts_multi_reference(reference, hypothesis)
        alternatives = split_reference_alternatives(reference)
        ref_index = max(0, int(metric.get("wer_reference_index", 1)) - 1)
        selected_reference = alternatives[min(ref_index, len(alternatives) - 1)]
        profile = align_word_errors(selected_reference, hypothesis)
        substitutions.update(profile["substitutions"])
        deletions.update(profile["deletions"])
        insertions.update(profile["insertions"])
        scored_samples += 1

    return {
        "scored_samples": scored_samples,
        "substitutions": _top(substitutions, limit),
        "deletions": _top(deletions, limit),
        "insertions": _top(insertions, limit),
        "note": (
            "Token attribution uses the same normalization and selected WER reference "
            "alternative as the headline ASR metric. Failed model calls are excluded "
            "from token attribution but included as empty hypotheses in headline WER/CER."
        ),
    }
