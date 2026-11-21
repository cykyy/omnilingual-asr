"""Step 3 (CPU is fine): beam-search decoding with an n-gram LM.

Reads the scores saved by dump_logits.py. Tunes the LM weight (alpha) and
word bonus (beta) on the validation split only, then decodes the test split
once with the best setting and reports greedy vs LM side by side.

    python scripts/lm/decode_lm.py \
        --logits-dir lm_work/logits \
        --lm lm_work/lm_raw_4/lm.4gram.bin lm_work/lm_nopunct_4/lm.4gram.bin \
        --out-dir lm_work/results

Each --lm file needs a unigrams.txt next to it (build_lm.py writes one).
Outputs in --out-dir: tuning_grid.csv, test_hyps.tsv, metrics.json, report.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_metrics import (  # noqa: E402
    METRICS,
    format_table,
    make_unit_splitter,
    paired_bootstrap,
    summarize,
    utterance_errors,
)

# Set in the parent right before forking the worker pool, so workers share
# them without pickling (fork start method).
_LOGITS: list[np.ndarray] = []
_DECODER = None
_BEAM_WIDTH = 64


def _decode_one(i: int) -> str:
    return _DECODER.decode(_LOGITS[i].astype(np.float32), beam_width=_BEAM_WIDTH)


def load_split(logits_dir: Path, split: str) -> dict:
    data = np.load(logits_dir / f"{split}.npz")
    offsets = data["offsets"]
    lp = data["logprobs"]
    return {
        "logits": [lp[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)],
        "ids": data["ids"].tolist(),
        "refs": data["refs"].tolist(),
        "greedy": data["greedy"].tolist(),
        "regions": data["regions"].tolist(),
    }


def make_decoder(labels, lm_path: Path | None, alpha: float, beta: float, unk_offset: float):
    from pyctcdecode import build_ctcdecoder

    if lm_path is None:
        return build_ctcdecoder(labels)
    unigram_file = lm_path.parent / "unigrams.txt"
    unigrams = unigram_file.read_text(encoding="utf-8").split() if unigram_file.exists() else None
    return build_ctcdecoder(
        labels,
        kenlm_model_path=str(lm_path),
        unigrams=unigrams,
        alpha=alpha,
        beta=beta,
        unk_score_offset=unk_offset,
    )


def decode_all(decoder, logits, indices, workers: int, beam_width: int) -> list[str]:
    global _LOGITS, _DECODER, _BEAM_WIDTH
    _LOGITS, _DECODER, _BEAM_WIDTH = logits, decoder, beam_width
    if workers <= 1:
        return [_decode_one(i) for i in indices]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        return pool.map(_decode_one, indices, chunksize=max(1, len(indices) // (workers * 4)))


def score(refs, hyps, split_units):
    return [utterance_errors(r, h, split_units) for r, h in zip(refs, hyps)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logits-dir", type=Path, default=Path("lm_work/logits"))
    p.add_argument("--lm", type=Path, nargs="+", required=True, help=".bin or .arpa from build_lm.py")
    p.add_argument("--out-dir", type=Path, default=Path("lm_work/results"))
    p.add_argument("--tune-split", default="valid")
    p.add_argument("--test-split", default="test")
    p.add_argument("--tune-limit", type=int, default=600, help="valid clips used for the grid (0 = all)")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    p.add_argument("--betas", type=float, nargs="+", default=[-1.0, 0.0, 1.0, 2.0, 3.0])
    p.add_argument("--tune-metric", choices=[k for k, _ in METRICS], default="wer")
    p.add_argument("--beam-width", type=int, default=64)
    p.add_argument("--unk-score-offset", type=float, default=-10.0)
    p.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta = json.loads((args.logits_dir / "labels.json").read_text(encoding="utf-8"))
    labels = meta["labels"]
    split_units = make_unit_splitter(labels)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tune = load_split(args.logits_dir, args.tune_split)
    rng = np.random.default_rng(args.seed)
    tune_idx = list(range(len(tune["ids"])))
    if args.tune_limit and args.tune_limit < len(tune_idx):
        tune_idx = sorted(rng.choice(tune_idx, size=args.tune_limit, replace=False).tolist())
    tune_refs = [tune["refs"][i] for i in tune_idx]
    print(f"Tuning on {len(tune_idx)} {args.tune_split} clips; metric = {args.tune_metric}")

    grid_rows = []
    best = None  # (score, lm, alpha, beta)
    for lm in args.lm:
        for alpha in args.alphas:
            for beta in args.betas:
                t0 = time.time()
                dec = make_decoder(labels, lm, alpha, beta, args.unk_score_offset)
                hyps = decode_all(dec, tune["logits"], tune_idx, args.workers, args.beam_width)
                m = summarize(score(tune_refs, hyps, split_units))
                grid_rows.append({"lm": str(lm), "alpha": alpha, "beta": beta, **m})
                print(f"  {lm.parent.name:>16s} a={alpha:<5} b={beta:<5} "
                      + " ".join(f"{k}={m[k]:.2f}" for k, _ in METRICS)
                      + f"  ({time.time() - t0:.0f}s)")
                if best is None or m[args.tune_metric] < best[0]:
                    best = (m[args.tune_metric], lm, alpha, beta)

    with (args.out_dir / "tuning_grid.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0]))
        w.writeheader()
        w.writerows(grid_rows)
    _, best_lm, best_a, best_b = best
    print(f"\nBest on {args.tune_split}: {best_lm} alpha={best_a} beta={best_b}")

    # Test: decoded once, with the setting chosen on validation.
    test = load_split(args.logits_dir, args.test_split)
    idx = list(range(len(test["ids"])))
    beam_nolm = decode_all(make_decoder(labels, None, 0, 0, 0), test["logits"], idx,
                           args.workers, args.beam_width)
    beam_lm = decode_all(make_decoder(labels, best_lm, best_a, best_b, args.unk_score_offset),
                         test["logits"], idx, args.workers, args.beam_width)

    rows = {
        "greedy (thesis setup)": score(test["refs"], test["greedy"], split_units),
        "beam, no LM": score(test["refs"], beam_nolm, split_units),
        f"beam + {best_lm.parent.name}": score(test["refs"], beam_lm, split_units),
    }
    overall = {name: summarize(r) for name, r in rows.items()}

    base, new = rows["greedy (thesis setup)"], rows[f"beam + {best_lm.parent.name}"]
    deltas = {k: paired_bootstrap(base, new, key=k, seed=args.seed) for k, _ in METRICS}

    regions = sorted(set(test["regions"]))
    per_region = {}
    for reg in regions:
        sel = [i for i, r in enumerate(test["regions"]) if r == reg]
        per_region[reg] = {
            "clips": len(sel),
            "greedy": summarize([base[i] for i in sel]),
            "lm": summarize([new[i] for i in sel]),
        }

    with (args.out_dir / "test_hyps.tsv").open("w", encoding="utf-8") as f:
        f.write("id\tregion\tref\tgreedy\tbeam_no_lm\tbeam_lm\n")
        for i in idx:
            f.write("\t".join([test["ids"][i], test["regions"][i], test["refs"][i],
                               test["greedy"][i], beam_nolm[i], beam_lm[i]]) + "\n")

    report = [
        f"Test split: {args.test_split}, {len(idx)} clips",
        f"LM chosen on {args.tune_split} ({len(tune_idx)} clips, by {args.tune_metric}): "
        f"{best_lm}, alpha={best_a}, beta={best_b}, beam={args.beam_width}",
        "",
        format_table(overall),
        "",
        "Change from greedy to beam+LM (points; negative = better), paired bootstrap 95% CI:",
    ]
    for k, name in METRICS:
        mean, lo, hi = deltas[k]
        report.append(f"  {name:16s} {mean:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
    report += ["", f"{'region':14s}{'clips':>6s}{'WER greedy':>12s}{'WER +LM':>10s}"
               f"{'UER greedy':>12s}{'UER +LM':>10s}"]
    for reg, d in per_region.items():
        report.append(f"{reg:14s}{d['clips']:6d}{d['greedy']['wer']:11.2f}%{d['lm']['wer']:9.2f}%"
                      f"{d['greedy']['uer']:11.2f}%{d['lm']['uer']:9.2f}%")
    text = "\n".join(report)
    print("\n" + text)
    (args.out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    (args.out_dir / "metrics.json").write_text(json.dumps({
        "chosen": {"lm": str(best_lm), "alpha": best_a, "beta": best_b,
                   "beam_width": args.beam_width, "tuned_on": args.tune_split,
                   "tune_clips": len(tune_idx), "tune_metric": args.tune_metric},
        "test": overall,
        "bootstrap_delta_vs_greedy": {k: dict(zip(("mean", "lo", "hi"), v)) for k, v in deltas.items()},
        "per_region": per_region,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out_dir}/report.txt, metrics.json, test_hyps.tsv, tuning_grid.csv")


if __name__ == "__main__":
    main()
