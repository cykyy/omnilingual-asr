# Re-running the thesis training on a rented GPU

The training server is gone, so a few details can no longer be checked: the GPU model,
the training time, the exact epoch count and the hash of the original checkpoint. One
re-run on a rented GPU records all of them for the new run, and in the same session
produces results the paper needs anyway.

| What the re-run gives | Why it matters |
|---|---|
| GPU model, memory, driver, every package version | Documents the hardware and software exactly |
| Training time, start to end | Replaces the "~6.4 h" we can no longer check |
| Epoch count from the trainer's own logs | Replaces the "~13.6 epochs" from our notes |
| Test WER/UER of the new run | Shows the result reproduces; with a different seed it is the second run reviewers ask for |
| Test WER/UER of the **published** Hugging Face model on all 2,132 clips | Confirms 72.97% for the released weights on the full test set (so far checked on one clip) |
| Zero-shot baseline re-scored | Checks the setup matches the thesis (should give 97.93%) |
| n-gram LM decoding results | The main follow-up experiment for the paper |

## 1. Pick a GPU

- **Needs:** NVIDIA, Ampere generation or newer (for bfloat16), at least 16 GB memory,
  Ubuntu image with CUDA, root access, **100 GB disk** (dataset ~25 GB after preparation,
  checkpoints ~20 GB).
- **Good choices:** RTX A4000 (16 GB, closest to the original), RTX 4000 Ada, L4, A10,
  RTX 3090/4090.
- **Avoid:** T4, V100, P100 (no bfloat16), and free notebook services (Colab, Kaggle) because
  sessions time out before training ends.
- **Do not change the config for a bigger GPU.** Keep the 1,000,000-sample batch and
  accumulation 8 so the run matches the thesis. A faster GPU only changes the time.
- Any provider works (RunPod, Vast.ai, Lambda, a university cluster). Prices change; these GPUs
  are usually well under US$1 per hour.

## 2. Time and cost

| Step | What it does | Approx. time |
|---|---|---|
| `env`, `setup` | Record hardware; install packages; build KenLM | 20–30 min |
| `data` | Download RegSpeech12 from Kaggle and prepare manifests | 30–60 min |
| `eval_baseline` | Zero-shot test eval (expect WER 97.93%) | ~2 min |
| `eval_released` | Download the Hugging Face model, check its hash, test eval (expect 72.97%) | ~5 min |
| `train` | 10,000 steps, same config as the thesis | ~6–7 h (depends on GPU) |
| `eval_new` | Test eval of the new checkpoint, plus its hash | ~2 min |
| `epochs` | Epoch count from the trainer's logs | seconds |
| `lm` | n-gram LM decoding (needs many CPU cores) | 30–90 min |
| `pack` | Bundle logs and results into one small `.tar.gz` | seconds |

Total about 9–10 hours, so roughly **US$3–10** on the GPUs above. A second seed adds ~7 hours.

## 3. Before you start

- A Kaggle account and API token (Kaggle → Settings → API → Create New Token). Put the
  downloaded `kaggle.json` at `~/.kaggle/kaggle.json` on the rented machine, with
  `chmod 600`. Never commit it.
- Decide who runs it and who pays. Set a spending limit in the provider's dashboard.

## 4. Run it

On the rented machine:

```bash
tmux new -s rerun
```

```bash
git clone -b rayth1 https://github.com/cykyy/omnilingual-asr.git && cd omnilingual-asr
```

```bash
bash scripts/rerun/rerun.sh
```

If `pip` refuses with "externally-managed-environment", create a virtual environment first
(`python3 -m venv ~/venv && source ~/venv/bin/activate`) and run the script again. Using the
provider's PyTorch image usually avoids this.

That runs every step in order and stops at the first error. `tmux` keeps it running if your
SSH connection drops: reconnect and type `tmux attach -t rerun`. To run or repeat single
steps, name them, e.g. `bash scripts/rerun/rerun.sh eval_new epochs pack`.

**Check after the first 15 minutes** that `eval_baseline` printed WER ≈ 97.93 and
`eval_released` printed WER ≈ 72.97. If not, stop before paying for training and send the
logs in `rerun_artifacts/`.

Second seed (optional, for variance):

```bash
SEED=3 bash scripts/rerun/rerun.sh train eval_new epochs pack
```

## 5. When it finishes

1. Copy the `rerun_seed2_<date>.tar.gz` file it prints to your computer (scp or the
   provider's file browser). It is small; checkpoints and the model are left out.
2. If you want to keep the new checkpoint, also copy
   `rerun_output/seed2/ws_1.*/checkpoints/step_10000/model/pp_00/tp_00/sdp_00.pt` (1.3 GB).
3. **Stop or delete the instance.** Most providers keep billing until you do.

The tarball holds `env.txt` (GPU and versions), `timeline.txt` (start and end of every step),
the training log, the trainer's logs and TensorBoard data, all eval metrics and transcripts,
the epoch count, the checkpoint hash and the LM results.

## What to expect

- `eval_baseline`: 97.93% WER, 81.74% UER (as in the thesis).
- `eval_released`: 72.97% WER, 35.44% UER. A difference in the second decimal is possible
  from different GPU hardware; a large difference means something is wrong.
- `eval_new`: near 72.97%. GPU training is not bit-for-bit repeatable, so it will not match
  exactly; how close it lands is what this run measures.
