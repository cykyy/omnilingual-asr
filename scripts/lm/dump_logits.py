"""Step 1 (GPU server): run the CTC model once and save its log-probabilities.

The n-gram decoder (decode_lm.py) then works on these saved scores and needs
no GPU and no fairseq2. This script also re-scores plain greedy decoding, so
you can check it reproduces the thesis number (test WER 72.97%) before trusting
any LM result.

Example, from the repo root:

    python scripts/lm/dump_logits.py \
        --model-card bn_regional_CTC_300M_v2 \
        --manifest-dir dataset/regspeech12/data_manifest \
        --train-text dataset/regspeech12/data_manifest/train.wrd \
        --splits valid test \
        --out-dir lm_work/logits

Audio is processed like the eval recipe (regspeech12-test.yaml): mono, zero
mean / unit variance, clips shorter than 1 s or longer than 30 s skipped, and
bfloat16 autocast on GPU. Each clip runs alone, so there is no padding.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model
from fairseq2.nn.batch_layout import BatchLayout

from omnilingual_asr.datasets.utils.audio import (
    apply_audio_normalization,
    convert_to_mono,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_metrics import format_table, make_unit_splitter, summarize, utterance_errors  # noqa: E402

REGION_RE = re.compile(r"^[A-Za-z]+_(.+)_\d+$")
BANGLA_BLOCK = range(0x0980, 0x0A00)


def region_of(audio_name: str) -> str:
    match = REGION_RE.match(Path(audio_name).stem)
    return match.group(1).lower() if match else "unknown"


def read_split(manifest_dir: Path, split: str):
    tsv = manifest_dir / f"{split}.tsv"
    wrd = manifest_dir / f"{split}.wrd"
    lines = tsv.read_text(encoding="utf-8").splitlines()
    audio_dir = Path(lines[0].strip())
    rows = [l.split("\t") for l in lines[1:] if l.strip()]
    texts = wrd.read_text(encoding="utf-8").splitlines()
    if len(rows) != len(texts):
        raise SystemExit(f"{tsv} has {len(rows)} clips but {wrd} has {len(texts)} lines")
    return [(audio_dir / r[0], int(r[1]), t) for r, t in zip(rows, texts)]


def special_ids(vocab_info) -> set[int]:
    names = ("pad_idx", "unk_idx", "bos_idx", "eos_idx")
    return {getattr(vocab_info, n) for n in names if getattr(vocab_info, n, None) is not None}


def token_strings(tokenizer, vocab_size: int, specials: set[int]) -> list[str | None]:
    """String for every token id, found by decoding it between two anchor letters.

    Decoding an id on its own can drop a word-boundary token, so we decode
    [anchor, id, anchor] and strip the anchors. Special tokens return None.
    """
    encoder = tokenizer.create_encoder()
    decoder = tokenizer.create_decoder(skip_special_tokens=True)
    anchor_ids = [int(i) for i in encoder("ক") if int(i) not in specials]
    if len(anchor_ids) != 1:
        raise SystemExit(f"Could not find a single token for the anchor letter: {anchor_ids}")
    a = anchor_ids[0]
    anchor = decoder(torch.tensor([a]))
    out: list[str | None] = []
    for i in range(vocab_size):
        if i in specials:
            out.append(None)
            continue
        s = decoder(torch.tensor([a, i, a]))
        if s.startswith(anchor) and s.endswith(anchor) and len(s) >= 2 * len(anchor):
            out.append(s[len(anchor) : len(s) - len(anchor)])
        else:
            out.append(decoder(torch.tensor([i])))
    return out


def build_columns(strings, train_text: str, blank_id: int):
    """Choose which vocabulary columns to keep and their pyctcdecode labels.

    The full multilingual vocabulary is too large to save per frame. We keep
    the blank, the word boundary, every Bangla-block token, and every token
    whose text occurs in the training transcripts (never valid or test text).
    Tokens with identical text are merged.
    """
    train_chars = set(train_text)
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(strings):
        if i == blank_id or s is None or s == "":
            continue
        label = " " if s.strip() == "" else s
        keep = (
            label == " "
            or any(ord(c) in BANGLA_BLOCK for c in label)
            or all(c in train_chars for c in label)
        )
        if keep:
            groups.setdefault(label, []).append(i)
    if " " not in groups:
        raise SystemExit("No word-boundary token found; cannot build a word-level decoder.")
    labels = [""] + sorted(groups)
    columns = [[blank_id]] + [groups[l] for l in labels[1:]]
    return labels, columns


def greedy(ids: torch.Tensor, blank_id: int) -> torch.Tensor:
    ids = torch.unique_consecutive(ids)
    return ids[ids != blank_id]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-card", default="bn_regional_CTC_300M_v2")
    p.add_argument("--manifest-dir", type=Path, default=Path("dataset/regspeech12/data_manifest"))
    p.add_argument("--train-text", type=Path, default=None, help="default: <manifest-dir>/train.wrd")
    p.add_argument("--splits", nargs="+", default=["valid", "test"])
    p.add_argument("--out-dir", type=Path, default=Path("lm_work/logits"))
    p.add_argument("--blank-id", type=int, default=0, help="CTC blank index (0, as in wer_calculator.py)")
    p.add_argument("--min-audio-len", type=int, default=16_000)
    p.add_argument("--max-audio-len", type=int, default=480_000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="only the first N clips (smoke test)")
    args = p.parse_args()

    train_text_path = args.train_text or args.manifest_dir / "train.wrd"
    train_text = train_text_path.read_text(encoding="utf-8")
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_card} on {device}")
    model = load_model(args.model_card, device=device, dtype=torch.float32)
    model.eval()
    tokenizer = load_tokenizer(args.model_card)
    vocab_info = tokenizer.vocab_info
    specials = special_ids(vocab_info) - {args.blank_id}
    encoder = tokenizer.create_encoder()
    decoder = tokenizer.create_decoder(skip_special_tokens=True)

    strings = token_strings(tokenizer, vocab_info.size, specials)
    labels, columns = build_columns(strings, train_text, args.blank_id)
    print(f"Vocabulary {vocab_info.size} tokens; keeping {len(labels)} labels for the decoder")
    (args.out_dir / "labels.json").write_text(
        json.dumps({"labels": labels, "columns": columns, "blank_id": args.blank_id,
                    "model_card": args.model_card}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    split_units = make_unit_splitter(labels)
    flat_cols = torch.tensor([c for group in columns for c in group], device=device)
    group_index = torch.tensor([g for g, group in enumerate(columns) for _ in group], device=device)

    for split in args.splits:
        items = read_split(args.manifest_dir, split)
        if args.limit:
            items = items[: args.limit]
        chunks, ids, refs, hyps, regions = [], [], [], [], []
        skipped = outside = frames_total = 0
        t0 = time.time()
        for n, (path, _, text) in enumerate(items):
            wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if sr != 16_000:
                raise SystemExit(f"{path}: expected 16 kHz audio, got {sr}")
            wav = torch.from_numpy(wav)
            if not (args.min_audio_len <= wav.shape[0] <= args.max_audio_len):
                skipped += 1
                continue
            wav = apply_audio_normalization(convert_to_mono(wav)).to(device).unsqueeze(0)
            layout = BatchLayout(wav.shape, seq_lens=[wav.shape[1]], device=device)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, out_layout = model(wav, layout)
            logits = logits[0, : out_layout.seq_lens[0]].float()
            logprobs = torch.log_softmax(logits, dim=-1)

            full_ids = logprobs.argmax(-1)
            hyp = decoder(greedy(full_ids, args.blank_id))
            # Reference as the eval recipe sees it: encoded, then decoded.
            ref = decoder(torch.tensor([int(i) for i in encoder(text)]))

            # Sum the probabilities of tokens that share a label (log-sum-exp).
            probs = torch.zeros((logprobs.shape[0], len(columns)), device=device)
            probs.index_add_(1, group_index, logprobs[:, flat_cols].exp())
            kept = probs.log().clamp(min=-1000.0)
            outside += int((~torch.isin(full_ids, flat_cols)).sum())
            frames_total += logprobs.shape[0]

            chunks.append(kept.cpu().numpy().astype(np.float16))
            ids.append(path.stem)
            refs.append(ref)
            hyps.append(hyp)
            regions.append(region_of(path.name))
            if (n + 1) % 200 == 0:
                print(f"  {split}: {n + 1}/{len(items)} clips, {time.time() - t0:.0f}s")

        offsets = np.cumsum([0] + [c.shape[0] for c in chunks])
        np.savez(
            args.out_dir / f"{split}.npz",
            logprobs=np.concatenate(chunks) if chunks else np.zeros((0, len(labels)), np.float16),
            offsets=offsets,
            ids=np.array(ids),
            refs=np.array(refs),
            greedy=np.array(hyps),
            regions=np.array(regions),
        )
        rows = [utterance_errors(r, h, split_units) for r, h in zip(refs, hyps)]
        print(f"\n{split}: saved {len(ids)} clips ({skipped} skipped for length) "
              f"to {args.out_dir / (split + '.npz')}")
        print(f"  frames whose best token was dropped from the saved labels: "
              f"{outside} of {frames_total} ({100 * outside / max(frames_total, 1):.3f}%)")
        print(format_table({f"greedy, full vocab ({split})": summarize(rows)}))
        if split == "test":
            print("  Expect strict WER close to 72.97% and UER close to 35.44% (thesis).")


if __name__ == "__main__":
    main()
