"""Scoring helpers shared by the n-gram LM decoding scripts.

Strict WER matches the thesis: words are split on whitespace and nothing else
is normalized. The normalized variant replaces every Unicode punctuation (P*)
or symbol (S*) character with a space. Bangla vowel signs and hasanta are
marks (M*), so they are kept.
"""

from __future__ import annotations

import random
import unicodedata
from typing import Callable, Dict, List, Sequence, Tuple


def edit_distance(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def strict(text: str) -> str:
    return " ".join(text.split())


def no_punct(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def make_unit_splitter(labels: Sequence[str]) -> Callable[[str], List[str]]:
    """Split text into the model's output units (longest match over labels).

    A space is one unit, as the tokenizer's word-boundary token is. Characters
    that are not in `labels` count as one unit each.
    """
    units = sorted({l for l in labels if l and l != " "}, key=len, reverse=True)
    max_len = max((len(u) for u in units), default=1)
    unit_set = set(units)

    def split(text: str) -> List[str]:
        text = strict(text)
        out, i = [], 0
        while i < len(text):
            if text[i] == " ":
                out.append(" ")
                i += 1
                continue
            for n in range(min(max_len, len(text) - i), 0, -1):
                if text[i : i + n] in unit_set or n == 1:
                    out.append(text[i : i + n])
                    i += n
                    break
        return out

    return split


def utterance_errors(
    ref: str, hyp: str, split_units: Callable[[str], List[str]]
) -> Dict[str, int]:
    """Per-utterance error counts and reference lengths for every metric."""
    r, h = strict(ref), strict(hyp)
    rn, hn = no_punct(ref), no_punct(hyp)
    return {
        "wer_e": edit_distance(r.split(), h.split()),
        "wer_n": len(r.split()),
        "nwer_e": edit_distance(rn.split(), hn.split()),
        "nwer_n": len(rn.split()),
        "uer_e": edit_distance(split_units(r), split_units(h)),
        "uer_n": len(split_units(r)),
        "cer_e": edit_distance(list(r), list(h)),
        "cer_n": len(r),
    }


METRICS = (("wer", "WER"), ("nwer", "WER (no punct)"), ("uer", "UER"), ("cer", "CER"))


def summarize(rows: List[Dict[str, int]]) -> Dict[str, float]:
    out = {}
    for key, _ in METRICS:
        e = sum(r[f"{key}_e"] for r in rows)
        n = sum(r[f"{key}_n"] for r in rows)
        out[key] = 100.0 * e / max(n, 1)
    return out


def paired_bootstrap(
    base: List[Dict[str, int]],
    new: List[Dict[str, int]],
    key: str = "wer",
    n_resamples: int = 1000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Mean change in `key` (new minus base, in points) with a 95% interval."""
    rng = random.Random(seed)
    idx = range(len(base))
    deltas = []
    for _ in range(n_resamples):
        sample = [rng.choice(idx) for _ in idx]
        n = sum(base[i][f"{key}_n"] for i in sample) or 1
        eb = sum(base[i][f"{key}_e"] for i in sample)
        en = sum(new[i][f"{key}_e"] for i in sample)
        deltas.append(100.0 * (en - eb) / n)
    deltas.sort()
    return (
        sum(deltas) / len(deltas),
        deltas[int(0.025 * n_resamples)],
        deltas[int(0.975 * n_resamples) - 1],
    )


def format_table(results: Dict[str, Dict[str, float]]) -> str:
    head = f"{'system':34s}" + "".join(f"{name:>16s}" for _, name in METRICS)
    lines = [head, "-" * len(head)]
    for system, m in results.items():
        lines.append(f"{system:34s}" + "".join(f"{m[k]:15.2f}%" for k, _ in METRICS))
    return "\n".join(lines)
