# n-gram LM decoding for the CTC model

The thesis model decodes greedily: it picks the best letter in each frame and never checks
whether the words are real. These scripts add a word-level n-gram language model (KenLM)
through beam search (pyctcdecode), tune it on the validation split, and score the test
split once. Nothing in the model or training changes.

| Step | Script | Where | Needs |
|---|---|---|---|
| 1 | `dump_logits.py` | GPU server | fairseq2 env, model, RegSpeech12 audio |
| 2 | `build_lm.py` | anywhere | KenLM binaries `lmplz`, `build_binary` |
| 3 | `decode_lm.py` | anywhere (CPU, many cores help) | `pyctcdecode`, `kenlm` Python module |
| check | `selftest.py` | anywhere | same as step 3 |

## Install (on the server)

```bash
pip install pyctcdecode https://github.com/kpu/kenlm/archive/master.zip
```

KenLM binaries (Ubuntu):

```bash
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
git clone https://github.com/kpu/kenlm.git ~/kenlm && cd ~/kenlm && mkdir -p build && cd build && cmake .. && make -j"$(nproc)" lmplz build_binary
```

The binaries end up in `~/kenlm/build/bin`. Pass that folder as `--kenlm-bin`.

Check the decoding side works before using the real model:

```bash
python scripts/lm/selftest.py
```

It should end with `SELFTEST PASSED`.

## Run

All commands run from the repo root.

**1. Save the model's scores** for valid and test (a few minutes on one GPU):

```bash
python scripts/lm/dump_logits.py --model-card bn_regional_CTC_300M_v2 --manifest-dir dataset/regspeech12/data_manifest --splits valid test --out-dir lm_work/logits
```

To use the local step-10,000 checkpoint instead of the Hugging Face download, edit the
`checkpoint:` line of `bn_regional_CTC_300M_v2` in
`src/omnilingual_asr/cards/models/rc_models_v2.yaml`.

**Stop here if the numbers are off.** The script prints greedy WER and UER for the test
split. They should be close to the thesis numbers: **WER 72.97%** and **UER about 35.4–35.5%**.
Our UER counts characters and spaces; fairseq2's counts token ids, so expect a small
difference (35.49% vs 35.44% on the committed transcripts). If WER is far from 72.97%, the
audio processing or checkpoint differs from the thesis run; fix that first.

**2. Build two LMs** from training text only (with and without punctuation):

```bash
python scripts/lm/build_lm.py --text dataset/regspeech12/data_manifest/train.wrd --order 4 --mode raw --out-dir lm_work/lm_raw_4 --kenlm-bin ~/kenlm/build/bin
```

```bash
python scripts/lm/build_lm.py --text dataset/regspeech12/data_manifest/train.wrd --order 4 --mode nopunct --out-dir lm_work/lm_nopunct_4 --kenlm-bin ~/kenlm/build/bin
```

The script refuses files named `valid`, `test` or `dev`. Extra dialect text (Vashantor,
Kothon) can be added as more `--text` files if its licence allows; report it as a separate
row in the paper.

**3. Tune and decode:**

```bash
python scripts/lm/decode_lm.py --logits-dir lm_work/logits --lm lm_work/lm_raw_4/lm.4gram.bin lm_work/lm_nopunct_4/lm.4gram.bin --out-dir lm_work/results
```

It tries every LM × alpha × beta on 600 random validation clips (`--tune-limit 0` uses all),
picks the best by strict WER (`--tune-metric nwer` to pick by WER without punctuation),
then decodes the test split once.

## What you get in `lm_work/results`

- `report.txt`: greedy vs beam-no-LM vs beam+LM on test (strict WER, WER without
  punctuation, UER, CER); the change with a 95% paired-bootstrap interval; per-region WER
  and UER.
- `metrics.json`: the same numbers, plus the chosen LM, alpha, beta and beam width.
- `test_hyps.tsv`: every clip's reference, greedy output and LM output, for error analysis.
- `tuning_grid.csv`: every validation setting tried.

## Rules for the paper

- Choose the LM and alpha/beta on validation only. Never re-tune after looking at test.
- Report greedy and beam+LM side by side, with the bootstrap interval.
- Say what text the LM was trained on and its size (`build_lm.py` prints it).
- Regions come from the audio filenames (`test_<region>_NNNN.wav`).
