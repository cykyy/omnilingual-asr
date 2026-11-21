"""Step 2 (any machine with KenLM): build the n-gram language model.

The LM is trained on training transcripts only. Never add valid or test text:
that leaks the answers and reviewers will reject the result.

    python scripts/lm/build_lm.py \
        --text dataset/regspeech12/data_manifest/train.wrd \
        --order 4 --mode raw --out-dir lm_work/lm_raw_4

    # Extra dialect text (e.g. Vashantor, Kothon), if the licence allows:
    python scripts/lm/build_lm.py --text train.wrd extra_dialect.txt ...

--mode raw keeps punctuation attached to words, matching strict WER.
--mode nopunct strips punctuation first; it usually helps "WER (no punct)".
Build both and let decode_lm.py compare them on the validation split.

Needs the KenLM binaries `lmplz` and `build_binary` on PATH, or pass
--kenlm-bin. See scripts/lm/README.md for installation.
"""

from __future__ import annotations

import argparse
import collections
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_metrics import no_punct, strict  # noqa: E402


def find_tool(name: str, kenlm_bin: Path | None) -> str:
    if kenlm_bin is not None:
        path = kenlm_bin / name
        if path.exists():
            return str(path)
        raise SystemExit(f"{name} not found in {kenlm_bin}")
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found on PATH; build KenLM or pass --kenlm-bin")
    return found


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", type=Path, nargs="+", required=True, help="one sentence per line")
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--mode", choices=["raw", "nopunct"], default="raw")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--kenlm-bin", type=Path, default=None)
    p.add_argument("--memory", default="20%", help="lmplz -S value")
    args = p.parse_args()

    for t in args.text:
        if any(s in t.name for s in ("valid", "test", "dev")):
            raise SystemExit(f"Refusing {t}: LM text must not come from valid or test.")

    clean = no_punct if args.mode == "nopunct" else strict
    lines = []
    for t in args.text:
        for line in t.read_text(encoding="utf-8").splitlines():
            line = clean(line)
            if line:
                lines.append(line)
    if not lines:
        raise SystemExit("No text lines found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    corpus = args.out_dir / "corpus.txt"
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = collections.Counter(w for l in lines for w in l.split())
    (args.out_dir / "unigrams.txt").write_text(
        "\n".join(w for w, _ in counts.most_common()) + "\n", encoding="utf-8"
    )

    arpa = args.out_dir / f"lm.{args.order}gram.arpa"
    binary = args.out_dir / f"lm.{args.order}gram.bin"
    lmplz = find_tool("lmplz", args.kenlm_bin)
    build_binary = find_tool("build_binary", args.kenlm_bin)
    # --discount_fallback: our corpus is small, so some count-of-counts are zero.
    with corpus.open("rb") as src, arpa.open("wb") as dst:
        subprocess.run(
            [lmplz, "-o", str(args.order), "--discount_fallback", "-S", args.memory,
             "-T", str(args.out_dir)],
            stdin=src, stdout=dst, check=True,
        )
    subprocess.run([build_binary, str(arpa), str(binary)], check=True)

    print(f"\n{len(lines)} lines, {sum(counts.values())} words, {len(counts)} distinct words")
    print(f"LM:       {binary}")
    print(f"Unigrams: {args.out_dir / 'unigrams.txt'}")


if __name__ == "__main__":
    main()
