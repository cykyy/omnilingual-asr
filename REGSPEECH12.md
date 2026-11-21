# Bengali Regional ASR on RegSpeech12 (branch `rayth1`)

This branch fine-tunes Meta's `omniASR_CTC_300M_v2` on RegSpeech12, a corpus of spontaneous Bengali
speech from 12 regions of Bangladesh, and compares it with the zero-shot base model.

- Thesis overview, per-dialect results and transcripts: https://github.com/cykyy/thesis-omni-v2
- Fine-tuned model: https://huggingface.co/cykyy/bn-regional-asr-300m
- Dataset: https://www.kaggle.com/datasets/mdrezuwanhassan/regspeech12 ([paper](https://arxiv.org/abs/2510.24096))

## Results (test split, 2,132 clips, strict scoring)

| Model | WER | UER | CTC loss | Logs |
|-------|-----|-----|----------|------|
| omniASR_CTC_300M_v2, zero-shot | 97.93% | 81.74% | 1482.45 | `eval_baseline/ws_1.ef6a65e4/` |
| Fine-tuned, step 10,000 | **72.97%** | **35.44%** | 393.83 | `eval_step10000/ws_1.e4249f83/` |

Each log folder has `metrics/eval.jsonl` and the full hypothesis/reference transcripts.

## What was actually run

| Item | File |
|------|------|
| Data preparation | `prepare_regspeech12.py` (resample to 16 kHz mono WAV, whitespace collapse only) |
| Dataset card | `src/omnilingual_asr/cards/datasets/regspeech12.yaml` |
| **Training config** | `workflows/recipes/wav2vec2/asr/configs/regspeech12-80hr.yaml` (1 GPU, ~6.4 h, 10,000 steps = 13.6 epochs) |
| Resolved training config | `release/bn-regional-asr-300m/config.yaml` (identical to the Hugging Face copy) |
| Eval config | `workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml` |
| WER calculator fix | `workflows/recipes/wav2vec2/asr/wer_calculator.py` (see `BUG_REPORT_WER_EMPTY_HYPOTHESIS.md`) |
| Epoch count | `scripts/calculate_epochs.py` |
| Model card | `bn_regional_CTC_300M_v2` in `src/omnilingual_asr/cards/models/rc_models_v2.yaml` |

Commands: `HOW-TO-RUN.md`. Full guide: `FINETUNING_GUIDE.md`. Inference: `INFERENCE_GUIDE.md`.

## Prepared but not run

| Config | Purpose |
|--------|---------|
| `regspeech12-4gpu.yaml` | Same model on 4 GPUs |
| `regspeech12-300m-50ep-4gpu.yaml` | Longer training (~50 epochs) |
| `regspeech12-1b-llm-4gpu.yaml` | 1B LLM-decoder variant (parked) |
| `regspeech12-300m-ctc-4gpu-16gb.yaml`, `regspeech12-300m-llm-4gpu-16gb.yaml` | 16 GB-GPU variants |

`custom-tune-300m.yaml` and `custom-tune-300m-v2.yaml` were early test runs on the dry-run data in
`dataset/dry_run/`.

## Re-running the training

The original training server is gone. [`scripts/rerun/`](scripts/rerun/README.md) re-runs the
thesis training on a rented GPU and records the hardware, timings, epoch count, checkpoint hash
and test results.
