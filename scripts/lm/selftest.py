"""Check that pyctcdecode + KenLM + decode_lm.py work, without the model or GPU.

It builds fake model scores from real Bangla transcripts: every frame points at
the right letter, but some letters are swapped for a wrong one (the right one
is a close second). A small LM trained on other transcripts should repair many
of those words. The size of the gain here means nothing; it only shows the
pipeline runs end to end and the LM is actually used.

    python scripts/lm/selftest.py
"""

from __future__ import annotations

import collections
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_TEXT = HERE.parents[1] / "eval_step10000/ws_1.e4249f83/transcriptions/rank_0.ref.txt"
REGIONS = ["barishal", "sylhet", "chittagong"]


def write_toy_arpa(lines: list[str], path: Path) -> None:
    """Bigram ARPA with add-one bigrams and zero backoff. Good enough to load."""
    uni = collections.Counter()
    bi = collections.Counter()
    for line in lines:
        words = ["<s>"] + line.split() + ["</s>"]
        uni.update(words)
        bi.update(zip(words, words[1:]))
    total = sum(uni.values())
    vocab = sorted(uni)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"\\data\\\nngram 1={len(vocab) + 1}\nngram 2={len(bi)}\n\n\\1-grams:\n")
        f.write(f"{math.log10(0.5 / total):.6f}\t<unk>\t0\n")
        for w in vocab:
            p = -99.0 if w == "<s>" else math.log10(uni[w] / total)
            f.write(f"{p:.6f}\t{w}\t0\n")
        f.write("\n\\2-grams:\n")
        for (a, b), c in bi.items():
            f.write(f"{math.log10((c + 1) / (uni[a] + len(vocab))):.6f}\t{a} {b}\n")
        f.write("\n\\end\\\n")


def fake_split(lines, labels, rng, swap_p=0.12):
    index = {l: i for i, l in enumerate(labels)}
    letters = [i for i, l in enumerate(labels) if l not in ("", " ")]
    chunks, greedy = [], []
    for line in lines:
        frames, hyp = [], []
        for ch in line:
            target = index[ch]
            shown = target
            if ch != " " and rng.random() < swap_p:
                shown = rng.choice(letters)
            for t in (shown, 0):  # letter frame, then a blank frame
                row = np.full(len(labels), -12.0)
                row[t] = -0.1
                if t == shown and shown != target:
                    row[target] = -1.2  # the right letter is a close second
                frames.append(row)
            hyp.append(labels[shown])
        chunks.append(np.array(frames, dtype=np.float16))
        greedy.append(" ".join("".join(hyp).split()))
    offsets = np.cumsum([0] + [c.shape[0] for c in chunks])
    return {
        "logprobs": np.concatenate(chunks),
        "offsets": offsets,
        "ids": np.array([f"x_{REGIONS[i % 3]}_{i:04d}" for i in range(len(lines))]),
        "refs": np.array(lines),
        "greedy": np.array(greedy),
        "regions": np.array([REGIONS[i % 3] for i in range(len(lines))]),
    }


def main() -> None:
    text = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEXT
    lines = [" ".join(l.split()) for l in text.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(0)
    rng.shuffle(lines)
    lm_lines, valid, test = lines[:1500], lines[1500:1560], lines[1560:1620]

    chars = sorted({c for l in lines for c in l} - {" "})
    labels = ["", " "] + chars

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        logits = tmp / "logits"
        logits.mkdir()
        (logits / "labels.json").write_text(json.dumps({"labels": labels}, ensure_ascii=False))
        np.savez(logits / "valid.npz", **fake_split(valid, labels, rng))
        np.savez(logits / "test.npz", **fake_split(test, labels, rng))

        lm_dir = tmp / "toy_lm"
        lm_dir.mkdir()
        write_toy_arpa(lm_lines, lm_dir / "lm.arpa")
        words = collections.Counter(w for l in lm_lines for w in l.split())
        (lm_dir / "unigrams.txt").write_text("\n".join(words) + "\n", encoding="utf-8")

        subprocess.run(
            [sys.executable, str(HERE / "decode_lm.py"), "--logits-dir", str(logits),
             "--lm", str(lm_dir / "lm.arpa"), "--out-dir", str(tmp / "results"),
             "--alphas", "0", "0.5", "1.0", "--betas", "0", "1.0", "--tune-limit", "0",
             "--beam-width", "32", "--workers", "2"],
            check=True,
        )
        metrics = json.loads((tmp / "results/metrics.json").read_text())
        names = list(metrics["test"])
        greedy_wer = metrics["test"][names[0]]["wer"]
        lm_wer = metrics["test"][names[-1]]["wer"]
        ok = lm_wer < greedy_wer
        print(f"\nSELFTEST {'PASSED' if ok else 'FAILED'}: greedy WER {greedy_wer:.2f}% -> LM WER {lm_wer:.2f}%")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
