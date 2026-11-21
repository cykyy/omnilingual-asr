# Thesis Defense Q&A — Fine-tuning OmniLingual ASR on RegSpeech12 (Bengali Regional)

Curated defense questions and fact-based answers. Numbers, file paths and config values come from this repository's code, configs and eval logs. `EXPERIMENT_FINDINGS.md` refers to our internal experiment report, which is not part of this repository. For the published thesis overview and per-dialect results see [cykyy/thesis-omni-v2](https://github.com/cykyy/thesis-omni-v2).

> **Corrected (Nov 2026).** The first version described hardware and batching from `regspeech12-4gpu.yaml`, a 4-GPU config added after training that was never run. The released model was trained with `regspeech12-80hr.yaml` on **one 16 GB GPU** with a 1,000,000-sample batch budget and gradient accumulation 8. Answers Q3, Q7, Q19, Q20, Q22, Q24, Q31, Q41, Q43–Q46, Q48, Q49, Q78, Q81, Q88 and the glossary are fixed accordingly.

---

## A. Objectives & Scope

**Q1. What is the objective of this thesis?**
A. To fine-tune the pre-trained OmniLingual ASR model (`omniASR_CTC_300M_v2`) on the RegSpeech12 Bengali regional speech dataset and measure how much it improves over the zero-shot baseline on WER, UER, and CTC loss. The experiment is documented in `EXPERIMENT_FINDINGS.md`.

**Q2. Why OmniLingual ASR for Bengali regional speech?**
A. OmniLingual ASR is pre-trained on 1,600+ languages (paper: arXiv:2511.09690), making it a strong starting point for low-resource language adaptation. Bengali regional dialects are under-represented in standard ASR corpora, so the zero-shot baseline establishes how much transfer is already present, and fine-tuning measures how much additional adaptation the 80-hour RegSpeech12 training set can provide.

**Q3. What contributions does this work make?**
A. (i) A reproducible fine-tuning pipeline for OmniLingual ASR on RegSpeech12, (ii) quantitative results showing a 24.96 percentage-point WER reduction and 46.30 pp UER reduction over zero-shot, (iii) qualitative transcription analyses showing script-confusion fixes (Bengali vs. Assamese/Devanagari) and word-boundary recovery, (iv) prepared but not yet run configs for longer training (`regspeech12-300m-50ep-4gpu.yaml`) and the 1B LLM variant (`regspeech12-1b-llm-4gpu.yaml`) for follow-on comparison.

**Q4. What is out of scope?**
A. External language model rescoring, data augmentation (speed perturbation, noise injection), domain-specific tokenizer training, and dialect-specific model partitioning. These are listed as future work in `EXPERIMENT_FINDINGS.md` §6.4.

---

## B. Dataset — RegSpeech12

**Q5. What is RegSpeech12?**
A. A Bengali regional-dialect ASR dataset hosted on Kaggle (`mdrezuwanhassan/regspeech12`). It contains conversational speech from 394+ speakers across multiple Bengali regional varieties. Audio is 16 kHz, 16-bit PCM, mono WAV.

**Q6. What are the splits and their sizes?**

| Split | Samples | Duration |
|---|---|---|
| Train | 17,049 | ~80 h |
| Valid | 2,132 | ~10 h |
| Test  | 2,132 | ~10 h |
| **Total** | **21,313** | **~100 h** |

Source: `EXPERIMENT_FINDINGS.md` §1.2.

**Q7. What is the audio length range?**
A. 1–30 seconds per clip, enforced by `min_audio_len: 16_000` and `max_audio_len: 480_000` (at 16 kHz) in every training/eval config (e.g., `workflows/recipes/wav2vec2/asr/configs/regspeech12-80hr.yaml:29-30`).

**Q8. Why is the 80-hour training volume considered limited?**
A. Competitive regional ASR typically needs 500+ hours. With 80 hours spread across multiple dialects, each dialect gets a small share, limiting phonetic and lexical coverage. This is called out in `EXPERIMENT_FINDINGS.md` §6.3.

---

## C. Data Preprocessing

**Q9. What preprocessing did you apply before training?**
A. Two stages:

- **Data preparation** (`prepare_regspeech12.py`):
  - Resample all audio to 16 kHz mono (multi-channel averaged), written as 16-bit PCM WAV (`:54, :138-156`).
  - Extract transcripts from the Kaggle `.xlsx` files (`:213-242`).
  - Whitespace collapse on transcriptions — `" ".join(transcription.split())` at `:406`.
  - **No** lowercasing, punctuation removal, Unicode normalization, or transliteration.
- **Training-time pipeline** (implemented in `src/omnilingual_asr/datasets/utils/audio.py` and `src/omnilingual_asr/datasets/tasks/asr_task.py`, configured by the YAML):
  - Audio z-score normalization (`normalize_audio: true`).
  - Raw waveform input (`use_fbank: false`).
  - Audio length filter 1–30 s.
  - Fast-speech filter (≥160 samples per character).
  - Empty-text and unknown-token-only filtering.
  - Window shuffle (window=1000) before and after batching.
  - Length-based dynamic bucketing (multiple of 8).

**Q10. What is the "name" of the preprocessor?**
A. There is no single named preprocessor class. The data-prep side is the `prepare_regspeech12.py` script (no class). The runtime side is a fairseq2-style **manifest ASR pipeline** (dataset family `manifest_asr_dataset`) combined with the **`omniASR_tokenizer_written_v2`** character tokenizer — these are the closest identifiers. In the thesis I refer to it as the "RegSpeech12 manifest-ASR preprocessing pipeline."

**Q11. Did you use SpecAugment or any data augmentation?**
A. No. `spec_aug_p: null` in every RegSpeech12 config. No speed perturbation, no noise injection. This is documented as future work.

**Q12. Why waveform input instead of log-mel filterbank features?**
A. wav2vec 2.0 encoders consume raw waveforms directly via their learned convolutional feature extractor. Using `use_fbank: false` matches the pre-training signal and avoids a mismatch that would hurt transfer.

**Q13. Do you apply any text normalization at evaluation time?**
A. No. WER is computed by fairseq2's stock `WerMetric` (`workflows/recipes/wav2vec2/asr/wer_calculator.py:18`). Hypotheses are detokenized with `skip_special_tokens=True` and compared verbatim against references. There is no lowercasing, punctuation stripping, or Whisper-style normalization. This is an important caveat — published WER is "strict."

---

## D. Model & Tokenizer

**Q14. Which model did you fine-tune?**
A. `omniASR_CTC_300M_v2`: 325,494,996 parameters, family `wav2vec2_asr`, architecture `300m_v2`. A wav2vec 2.0 encoder with a CTC head.

**Q15. What does the architecture look like end-to-end?**
A. Raw 16 kHz waveform → wav2vec 2.0 CNN feature extractor → Transformer encoder (contextualized representations) → linear CTC projection over the tokenizer vocabulary → CTC loss during training / greedy argmax + duplicate collapse at inference. For the 1B variant, the CTC head is replaced with an autoregressive LLaMA decoder (`wav2vec2_llama`).

**Q16. What tokenizer did you use?**
A. `omniASR_tokenizer_written_v2`, a character-level tokenizer included with OmniLingual ASR that supports Bengali script and many other writing systems. Referenced in every config and in the dataset card `src/omnilingual_asr/cards/datasets/regspeech12.yaml:7`.

**Q17. Why a character tokenizer and not BPE/SentencePiece subwords?**
A. The pre-trained model ships with this tokenizer; fine-tuning must reuse it to keep the decoder projection aligned. Character tokenization is also more robust to regional spelling variation where subword units may not generalize.

**Q18. What is the difference between the CTC 300M and the 1B LLM variant?**

| Aspect | 300M CTC | 1B LLM (LLaMA decoder) |
|---|---|---|
| Output | Conditionally independent | Autoregressive |
| Implicit LM | None | Yes (decoder) |
| Inference | Parallel (fast) | Sequential (slower) |
| Memory | ~10 GB | ~30 GB, requires FSDP |
| Best for | Real-time / streaming | Offline / accuracy |

Source: `FINETUNING_GUIDE.md` §"CTC vs LLM Comparison".

---

## E. Training Setup

**Q19. What training command and hardware?**
A. Single GPU: `python -m workflows.recipes.wav2vec2.asr --config-file workflows/recipes/wav2vec2/asr/configs/regspeech12-80hr.yaml output/` (world size 1, run folder `ws_1.ecbbc39f`). The GPU had 16 GB of memory (the config is sized for a 16 GB card). Training took about 6.4 hours for 10,000 steps (13.6 epochs). The ~2.7 GB peak memory in the logs is from the **evaluation** run, not training.

**Q20. What are the key hyperparameters?**

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β=[0.9, 0.98], ε=1e-08, wd=0.01) |
| Learning rate | 5e-05 |
| LR schedule | tri-stage: 10% warmup / 40% hold / 50% decay |
| Precision | bfloat16 |
| Training steps | 10,000 (13.6 epochs) |
| Encoder freeze | 0 steps (full fine-tuning) |
| Grad accumulation | 8 batches per update |
| Batching | Dynamic, length-based, up to 1,000,000 audio samples (~62 s) per batch |

Source: `workflows/recipes/wav2vec2/asr/configs/regspeech12-80hr.yaml`; full resolved config in `release/bn-regional-asr-300m/config.yaml`. The prepared but unrun configs differ: the 50-epoch config uses LR 3e-05, 9,500 steps and a 5/45/50 schedule; the 1B LLM config uses LR 3e-05 and gradient accumulation 2.

**Q21. Why didn't you freeze the encoder?**
A. `freeze_encoder_for_n_steps: 0` — full fine-tuning was chosen because the entire acoustic model needs to adapt to Bengali regional phonology. Freezing is more appropriate when attaching a brand-new decoder head to a fully-trained encoder, not when adapting both to a new language domain.

**Q22. Why bfloat16 and not float16?**
A. bfloat16 has the same exponent range as fp32, eliminating the loss-scaling required by fp16 and avoiding NaN issues on the large dynamic range of CTC loss values. It needs a recent NVIDIA GPU (Ampere generation or newer).

**Q23. How did you compute the number of epochs?**
A. fairseq2 uses length-based dynamic batching, so `steps × batch_size / N` over-counts. I used the checkpoint metadata (`_data_epoch_nr - 1`) and training logs (`End of epoch N reached after step M`). The `scripts/calculate_epochs.py` helper consolidates this. Reported: 10,000 steps = **13.6 actual epochs** (naive calculation would give 37.5).

**Q24. What batch size did you actually run?**
A. There is no fixed batch size. `max_num_elements: 1_000_000` caps each batch at one million audio samples (~62 s of 16 kHz audio), so a batch holds a few clips depending on their length. Gradients from 8 batches are accumulated per optimizer step. From the checkpoint, 231,881 clips were seen in 10,000 steps, so about **23 clips per update** (not 64, which the thesis book states). The unrun 1B LLM config uses `800_000`.

**Q25. What was the learning-rate schedule?**
A. Tri-stage:
- Warmup: 10% of steps (≈1,000), linear from 0.01× → 1× peak LR.
- Hold: 40% (≈4,000 steps) at peak LR.
- Decay: 50% (≈5,000 steps), linear down to 0.05× peak LR.

The 50-epoch variant uses 5/45/50 stage ratios with `final_lr_scale: 0.01` for smoother late-stage decay (`workflows/recipes/wav2vec2/asr/configs/regspeech12-300m-50ep-4gpu.yaml:58-63`).

---

## F. Evaluation

**Q26. How did you evaluate?**
A. `python -m workflows.recipes.wav2vec2.asr.eval --config model.path=... --config model.family=wav2vec2_asr --config model.arch=300m_v2 --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml eval_step10000/`. For the zero-shot baseline, I used `model.name=omniASR_CTC_300M_v2` with `model.path=null`.

**Q27. What metrics did you report?**
A.
- **WER** (Word Error Rate): (S + D + I) / N at word level.
- **UER** (Unit / Character Error Rate): same formula at character level.
- **CTC Loss**: total loss on the test set.

**Q28. How is the decoded hypothesis produced?**
A.
- 300M CTC: greedy argmax over frame logits, followed by CTC blank collapse via `unique_consecutive()` (`wer_calculator.py:188`).
- 1B LLM: beam search with `Wav2Vec2LlamaBeamSearchSeq2SeqGenerator`.
- Detokenization: `tokenizer.create_decoder(skip_special_tokens=True)` (`wer_calculator.py:60`).

**Q29. What are the headline results?**

| Metric | Zero-shot | Fine-tuned (10 k steps) | Δ (pp) | Relative |
|---|---|---|---|---|
| WER | 97.93% | **72.97%** | −24.96 | −25.5% |
| UER | 81.74% | **35.44%** | −46.30 | −56.6% |
| CTC Loss | 1482.45 | **393.83** | −1088.62 | −73.4% |

Source: `eval_baseline/ws_1.ef6a65e4/metrics/eval.jsonl` and `eval_step10000/ws_1.e4249f83/metrics/eval.jsonl`.

**Q30. Why is UER so much better than WER?**
A. Character-level errors are a more granular signal. The model has clearly learned Bengali **character-level** patterns (script, glyph sequences, common digraphs) far better than **word-level** patterns (morphology, dialectal word forms, word boundaries). With ~80 hours, character co-occurrence is well-sampled; word-level coverage is not. This gap is exactly what an external language model or more data would close.

**Q31. Why is the fine-tuned WER still 72.97%?**
A. The limitations are data volume (80 h across many dialects), no LM rescoring, no augmentation, and a general (not Bengali-specialised) tokenizer. The WER metric itself is also strict — no text normalization is applied before scoring, so every punctuation mismatch counts. Re-scored after the defense with punctuation removed, WER is 68.01%.

---

## G. Techniques & Design Choices

**Q32. What is CTC and why does ASR use it?**
A. Connectionist Temporal Classification defines a loss over all valid alignments between input frames and an output sequence by introducing a blank token and collapsing consecutive duplicates. It lets you train a frame-level classifier without frame-aligned labels, which is essential because we only have utterance-level transcripts.

**Q33. What is wav2vec 2.0?**
A. A self-supervised speech representation model (Baevski et al., NeurIPS 2020). It learns by solving a contrastive task over masked latent speech representations, producing a pre-trained encoder that can be fine-tuned for downstream tasks like ASR. OmniLingual ASR extends this to 1,600+ languages (arXiv:2511.09690).

**Q34. What is FSDP and why is it required for the 1B model?**
A. Fully Sharded Data Parallel. Parameters, gradients, and optimizer states are sharded across GPUs and gathered just-in-time for forward/backward. Without sharding, a 1B model in bf16 plus AdamW state would not fit on a 24 GB GPU. The 1B config enables FSDP at layer granularity with `fp32_reduce`, `reshard_after_forward`, and selective activation checkpointing every 2 layers (`regspeech12-1b-llm-4gpu.yaml:53-67`).

**Q35. What is activation checkpointing and why use it?**
A. A memory-saving technique: during the forward pass, intermediate activations for selected layers are discarded and recomputed during backward. Trades compute for memory. Enabled in the 1B config with `mode: selective, every_nth_layer: 2`.

**Q36. Why dynamic length-based batching?**
A. ASR audio lengths range 1–30 s. Fixed-size batches waste memory on padding for short clips or OOM on long ones. fairseq2's dynamic bucketing packs sequences by time-step budget (`max_num_elements`) so GPU memory stays constant regardless of clip length.

**Q37. Why the window shuffle (1000) instead of full-dataset shuffle?**
A. Streaming from manifest storage — we don't want to materialize and shuffle the entire dataset in memory. A 1000-example window is large enough to decorrelate neighboring samples while preserving streaming efficiency.

---

## H. Result Analysis

**Q38. Which qualitative improvements did you observe?**
A. From the five transcription examples in `EXPERIMENT_FINDINGS.md` §5:

- **Script disambiguation**: zero-shot output mixed Bengali, Assamese, and Devanagari glyphs; fine-tuned consistently produces Bengali script.
- **Word-boundary recovery**: zero-shot frequently merges or splits words; fine-tuned preserves sentence structure.
- **Regional vocabulary**: zero-shot fails on dialectal words; fine-tuned often captures them with minor spelling variation.
- **Long-form narrative**: zero-shot can collapse to near-empty output on complex utterances; fine-tuned recovers full content.

**Q39. Where does the model still fail?**
A. Spelling variants of the same dialect word, code-switched English/Bengali tokens (pop-culture names like "Freddy Mercury"), and rare morphological forms. These are vocabulary-coverage issues that 80 hours cannot fully solve.

**Q40. How does a 25 pp WER drop translate in practice?**
A. At 97.93% WER the system is essentially unusable — nearly every word is wrong. At 72.97% the system is still not production-grade (typical production target is <20%), but it has crossed from "random-looking output" to "recognisable transcription with errors" — useful as a pre-fill for human correction or as a retrieval signal.

---

## I. Limitations & Future Work

**Q41. What are the main limitations of this work?**
A.
- Only 80 h training; dialect-per-dialect hours are much smaller.
- No data augmentation (SpecAugment, speed perturbation, noise).
- No external language model at decode time; pure CTC argmax.
- Tokenizer is multilingual, not Bengali-tuned.
- Single random seed — no variance estimate.
- No fine-tuned comparison between 300M CTC, 50-epoch 300M and 1B LLM (configs exist; the 13.6-epoch 300M run is the only fine-tuning reported). The 1B LLM was only tested zero-shot in an early experiment (88.17% WER).

**Q42. What would you do next?**
A. Following `EXPERIMENT_FINDINGS.md` §6.4:
1. Data augmentation (speed perturbation 0.9×/1.1×, additive noise, SpecAugment).
2. Extended training (50+ epochs) — config `regspeech12-300m-50ep-4gpu.yaml` is ready.
3. External LM rescoring (n-gram or neural) on CTC lattices.
4. Scale to 1B LLM variant for implicit LM benefits — config `regspeech12-1b-llm-4gpu.yaml` is ready.
5. Multi-task joint training with Standard Bengali for positive transfer.
6. Collect 500+ hours of regional data.

**Q43. Why didn't you run the 50-epoch or 1B LLM experiments as reported results?**
A. Compute: our runs used a single 16 GB GPU, while the 50-epoch and 1B LLM configs were written for a 4-GPU setup (the 1B model needs FSDP sharding). The thesis reports the completed 13.6-epoch 300M run and leaves those configs as future work.

---

## J. Reproducibility

**Q44. Can someone reproduce these results?**
A. Yes. All artifacts are checked in:

| Artifact | Path |
|---|---|
| Data prep script | `prepare_regspeech12.py` |
| Training config (headline result) | `workflows/recipes/wav2vec2/asr/configs/regspeech12-80hr.yaml` |
| Resolved training config (every setting) | `release/bn-regional-asr-300m/config.yaml` |
| 50-epoch config (not run) | `workflows/recipes/wav2vec2/asr/configs/regspeech12-300m-50ep-4gpu.yaml` |
| 1B LLM config (not run) | `workflows/recipes/wav2vec2/asr/configs/regspeech12-1b-llm-4gpu.yaml` |
| Eval config (CTC) | `workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml` |
| Eval config (1B LLM) | `workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-1b-llm-test.yaml` |
| Dataset card | `src/omnilingual_asr/cards/datasets/regspeech12.yaml` |
| Release model card | `release/bn-regional-asr-300m/README.md` |
| Baseline metrics | `eval_baseline/ws_1.ef6a65e4/metrics/eval.jsonl` |
| Fine-tuned metrics | `eval_step10000/ws_1.e4249f83/metrics/eval.jsonl` |
| Finetuning guide | `FINETUNING_GUIDE.md` |
| Thesis overview and per-dialect results | [cykyy/thesis-omni-v2](https://github.com/cykyy/thesis-omni-v2) |

**Q45. What random-seed / non-determinism caveats should the board know about?**
A. fairseq2 uses a fixed seed for example shuffling (seed=2 in `asr_task.py`), but dynamic batching, distributed gradient reduction, and cudnn non-determinism mean bit-exact reproduction across hardware is not guaranteed. We ran a single seed, so run-to-run variance was not measured.

---

## K. Likely "Gotcha" Questions

**Q46. Your WER is strict — have you checked whether normalising case/punctuation changes the story?**
A. The reported 72.97% is computed with no text normalisation on hypotheses or references. A Whisper-style normalisation pass (lowercase, strip punctuation, collapse whitespace) would likely shave several points off the absolute WER, and re-scoring after the defense confirmed it: removing punctuation only gives 68.01% WER (−4.96 pp). The thesis reports strict scoring because (a) the baseline and fine-tuned model are compared under identical scoring, so the 24.96 pp **improvement** is unaffected, and (b) strict scoring is the honest number for a deployed system.

**Q47. Why is this fine-tuning and not training from scratch?**
A. 80 hours is vastly insufficient to train a 300M-parameter wav2vec 2.0 ASR model from scratch (comparable models use tens of thousands of hours). Fine-tuning leverages the pre-trained acoustic representations — including any transfer from the 1,600+ pre-training languages — which is the only viable path for low-resource regional speech.

**Q48. How do we know the improvement isn't just memorisation?**
A. WER is reported on the held-out **test** split (2,132 samples, ~10 h), which the model never sees during training or validation. The validation split (another 2,132 samples) was used to monitor WER during training; the reported model is the final checkpoint (step 10,000). Train/valid/test come pre-partitioned by the RegSpeech12 authors; we did not check speaker overlap between splits ourselves.

**Q49. Did you compare against any other Bengali ASR baseline?**
A. The thesis reports the zero-shot `omniASR_CTC_300M_v2` as the baseline. An earlier experiment also tested the larger `omniASR_LLM_1B_v2` zero-shot (88.17% WER on the same test split); our fine-tuned model beats it in every dialect. Comparing against other Bengali systems (e.g., Whisper, MMS) would require running them on identical test splits with identical scoring — that is a valid reviewer request and an honest "future work" answer.

**Q50. One sentence: what does this thesis claim?**
A. Fine-tuning `omniASR_CTC_300M_v2` on 80 hours of RegSpeech12 reduces word error rate on Bengali regional speech from 97.93% (zero-shot) to 72.97% and unit error rate from 81.74% to 35.44% under strict scoring, demonstrating meaningful but still data-limited adaptation of a large multilingual ASR model to a low-resource regional language variety.

---

## L. Basic Concepts — "Trivial" Questions the Board May Ask

**Q51. What is "zero-shot" in this context?**
A. Evaluating the pre-trained model on the RegSpeech12 test set **without any fine-tuning on RegSpeech12** — i.e., only the knowledge acquired during OmniLingual ASR's multilingual pre-training is used. It measures out-of-the-box transfer to Bengali regional speech before any target-domain adaptation.

**Q52. What is "fine-tuning"?**
A. Continuing training of an already-trained model on a new (usually smaller) dataset so it specialises to that domain. Here, all model parameters are updated (no layers frozen) using the 80-hour RegSpeech12 training set for 10,000 optimizer steps.

**Q53. What is an "epoch"?**
A. One full pass over the entire training dataset. For RegSpeech12 with 17,049 training samples, one epoch means the model has seen every training sample exactly once (subject to the length filter).

**Q54. What is a "training step"?**
A. One optimizer update: a forward pass on a batch, a backward pass to compute gradients, and a parameter update. 10,000 steps with fairseq2's dynamic batching corresponds to ~13.6 actual epochs on this dataset.

**Q55. Why do 10,000 steps equal 13.6 epochs and not 37.5?**
A. Because batches are sized by **time-step budget** (`max_num_elements`), not by fixed sequence count. Long audio clips fit fewer sequences per batch, short clips fit more. The naive formula `steps × batch_size / N` assumes fixed batch size, which is wrong here. The real count comes from the checkpoint's `_data_epoch_nr` field.

**Q56. What is a "batch"?**
A. A group of training examples processed together in one forward/backward pass so the GPU is used efficiently. Here batches are assembled by audio length so all clips in a batch have similar duration, minimising padding.

**Q57. What is "loss"?**
A. A scalar measuring how wrong the model's output is on a batch. Training minimises this quantity by adjusting model weights via gradient descent.

**Q58. What is "CTC loss"?**
A. **C**onnectionist **T**emporal **C**lassification loss. It measures the negative log-probability of the correct transcription summed over every valid alignment between the encoder's per-frame outputs and the target character sequence, using a special blank symbol to allow repeats and silences. It lets us train an ASR model without frame-level alignments between audio and text.

**Q59. What is a "blank token" in CTC?**
A. An extra symbol ("∅") the model can output at any frame to mean "no new character." Consecutive duplicate characters separated by a blank are kept as duplicates; duplicates not separated by a blank are collapsed into one. This is how CTC handles variable audio-to-text length.

**Q60. What is "greedy decoding"?**
A. At inference, picking the highest-probability token at every frame independently, then collapsing duplicates and removing blanks. Simple and fast, but ignores global sequence probability.

**Q61. What is "beam search"?**
A. A decoding strategy that tracks the top-K most probable partial hypotheses at each step instead of just one. Used by the 1B LLM variant via `Wav2Vec2LlamaBeamSearchSeq2SeqGenerator`. Produces better-quality outputs than greedy at the cost of extra compute.

**Q62. What is "WER"?**
A. **W**ord **E**rror **R**ate. Computed as `(S + D + I) / N`, where S = substitutions, D = deletions, I = insertions needed to turn the hypothesis into the reference, and N = number of reference words. Lower is better; 0% is perfect.

**Q63. What is "UER" (and how is it different from CER)?**
A. **U**nit **E**rror **R**ate. Same formula as WER but at the character/unit level instead of the word level. In practice it is equivalent to **CER** (**C**haracter **E**rror **R**ate); fairseq2's metric is named UER because it counts "tokenizer units," which for `omniASR_tokenizer_written_v2` are characters.

**Q64. What is "pp" in "24.96 pp reduction"?**
A. **P**ercentage **p**oints. The arithmetic difference between two percentages. 97.93% − 72.97% = 24.96 **pp**. This is not the same as percent change (−25.5% relative). Using "pp" avoids the common ambiguity between absolute and relative change.

**Q65. What is an "encoder" vs a "decoder" in this model?**
A.
- **Encoder**: takes the raw waveform and produces contextualised per-frame representations. Here it is a wav2vec 2.0 Transformer encoder.
- **Decoder**: takes those representations and produces the output text. For the 300M model this is a single linear CTC projection; for the 1B model it is an autoregressive LLaMA language model.

**Q66. What is "wav2vec 2.0"?**
A. A self-supervised speech representation model from Meta (Baevski et al., 2020). During pre-training it learns to predict masked spans of its own latent speech features using a contrastive objective. The pre-trained encoder can then be fine-tuned for ASR with far less labelled data than training from scratch.

**Q67. What is "self-supervised pre-training"?**
A. Learning from unlabelled data by constructing a surrogate task from the data itself (e.g., predicting masked portions of a signal). It lets the model learn general speech structure from thousands of hours of raw audio before seeing any transcripts.

**Q68. What is "self-attention"?**
A. The core operation inside Transformer layers: each position in a sequence computes a weighted combination of all other positions, with weights learned from the content itself. It lets the encoder capture long-range context in the audio.

**Q69. What is a "Transformer"?**
A. The neural network architecture behind modern speech and language models (Vaswani et al., 2017). Stacks of self-attention + feed-forward layers with residual connections and normalisation. Both the wav2vec 2.0 encoder and the LLaMA decoder are Transformers.

**Q70. What is "LLaMA" (in "wav2vec2-LLaMA")?**
A. A family of decoder-only Transformer language models from Meta. In the 1B variant of OmniLingual ASR, a LLaMA-style decoder is attached to the wav2vec 2.0 encoder to autoregressively generate transcriptions with an implicit internal language model.

**Q71. What is "autoregressive" generation?**
A. Producing each output token conditioned on all previously generated tokens. The LLaMA decoder works this way, which is why inference is sequential (slower) but quality is typically higher — it has an implicit language model.

**Q72. What is a "tokenizer"?**
A. A deterministic mapping between text and a fixed vocabulary of integer IDs. Our tokenizer is character-level: each Bengali (or other script) character becomes one token. Tokenisation is how the model reads and writes text.

**Q73. What is a "manifest" (TSV / WRD)?**
A. A simple two-file format listing the training examples:
- `.tsv` — audio root directory + one line per clip with `filename<TAB>num_samples`.
- `.wrd` — one transcription per line, in the same order.

It lets the loader stream audio and labels without touching a database.

**Q74. What is an "optimizer" and what is AdamW?**
A. The **optimizer** is the algorithm that updates model weights from gradients. **AdamW** (Adam with decoupled **W**eight decay) is a widely-used variant that applies weight decay outside the gradient computation, which is mathematically cleaner and more effective than Adam's L2 regularisation. Configured here as β=(0.9, 0.98), ε=1e-08, weight decay 0.01.

**Q75. What is the "learning rate"?**
A. A scalar controlling how large each weight update is. Too high → unstable training; too low → slow or stuck. I used 5e-05 for the main run (moderate, appropriate for fine-tuning a pre-trained model) and 3e-05 for longer / larger variants.

**Q76. What is a "learning-rate schedule"?**
A. A rule for changing the learning rate during training. I used **tri-stage**: 10% warmup (ramp up), 40% hold (constant peak), 50% decay (ramp down). Warmup avoids early instability, hold lets the model learn at full rate, decay lets it settle into a good minimum.

**Q77. What is "weight decay"?**
A. A regularisation term that pulls weights toward zero every step, preventing them from growing arbitrarily large and thus reducing overfitting. Value here: 0.01.

**Q78. What is "gradient accumulation"?**
A. Running several small forward/backward passes and summing their gradients before one optimizer step, to simulate a larger effective batch when GPU memory is limited. Set to 8 for our single-GPU 300M run (and 2 in the unrun 1B LLM config).

**Q79. What is "mixed precision" and what is "bfloat16"?**
A. **Mixed precision** means running parts of the computation in a lower-precision floating-point format (16-bit) to save memory and speed up math, while keeping critical bits (master weights, some reductions) in 32-bit. **bfloat16** is a 16-bit format with the same exponent range as float32, so it avoids the overflow/underflow issues of regular float16 and needs no loss scaling.

**Q80. What is "FSDP"?**
A. **F**ully **S**harded **D**ata **P**arallel. A distributed-training strategy that shards model parameters, gradients, and optimizer state across GPUs and gathers them only when needed. Needed for the (unrun) 1B LLM config, written for 4 GPUs, because a full replica plus optimizer state would not fit on one GPU.

**Q81. What is "DDP"?**
A. **D**istributed **D**ata **P**arallel. Each GPU holds a full copy of the model and works on a different slice of the batch; gradients are averaged across GPUs each step. The 300M config uses DDP; our run had one GPU (world size 1), so it was effectively single-GPU training.

**Q82. What is "activation checkpointing" (or "gradient checkpointing")?**
A. A memory-saving technique: during forward pass, discard intermediate activations for some layers instead of storing them; during backward pass, recompute them. Trades extra compute for reduced memory. Enabled selectively in the 1B LLM config.

**Q83. What is "TF32"?**
A. **T**ensor**F**loat-**32**. An NVIDIA Ampere+ compute mode that runs matrix multiplies in a reduced-precision 19-bit format internally while keeping 32-bit inputs/outputs, giving a large speed-up for training with negligible accuracy loss. Enabled in the extended-training configs.

**Q84. What is "SpecAugment"?**
A. A data-augmentation technique for speech (Park et al., 2019) that randomly masks out bands of frequencies and time-windows in the spectrogram (or feature) input during training, to force the model to be robust to missing information. **Not used** in this project (`spec_aug_p: null`).

**Q85. What is "dynamic / length-based batching"?**
A. A batching strategy that groups training examples by length and caps each batch by a total-elements budget rather than a fixed sequence count, so GPU memory usage stays constant regardless of audio duration.

**Q86. What is "bucketing / window shuffle"?**
A. Instead of shuffling the whole dataset (which requires loading it all into memory), maintain a window of N recent examples, shuffle within it, and advance the window. With window=1000, two adjacent examples in the stream end up random within each 1000-sample chunk.

**Q87. What is a "checkpoint"?**
A. A saved snapshot of model weights (and optimizer state) at a particular step. Lets you resume training, evaluate a specific point during training, or roll back. Checkpoints are written every 2,000 steps in the main config and kept for best/last selection.

**Q88. What is "validation" vs "test"?**
A.
- **Validation set**: used during training to monitor WER (we report the final checkpoint, step 10,000).
- **Test set**: held out entirely, used once at the end to report the headline WER/UER.

Keeping them disjoint prevents optimising for the numbers you report.

**Q89. What is "overfitting"?**
A. When a model learns patterns specific to the training set that don't generalise, causing training loss to keep decreasing while validation loss rises. Early stopping, weight decay, and dropout are common defences.

**Q90. What is a "language model" (LM) and why is it missing here?**
A. A model of the probability distribution over token sequences; in ASR it re-ranks or rescoring decoder outputs using linguistic likelihood ("the cat sat" beats "the cat st"). This project uses pure CTC greedy decoding with **no external LM**, which is one reason the remaining WER is high and is the most obvious future improvement.

**Q91. What is "fairseq2"?**
A. Meta's research library for sequence modelling (the successor to fairseq). OmniLingual ASR is built on it, and this project uses its training loop, dataset abstractions, WerMetric, and beam-search generator.

**Q92. What is a "parameter" and what does "300M" / "1B" mean?**
A. A **parameter** is one learnable weight inside the neural network. "300M" means ~300 million parameters (exactly 325,494,996 for our model); "1B" means ~1 billion. More parameters = more capacity but also more memory, compute, and data required to train well.

---

## M. Acronym / Abbreviation Glossary

| Acronym | Full form | Meaning in this thesis |
|---|---|---|
| **ASR** | Automatic Speech Recognition | The task of converting speech audio to text. |
| **WER** | Word Error Rate | Primary metric: `(S + D + I) / N` at the word level. |
| **UER** | Unit Error Rate | Same formula as WER but at the tokenizer-unit (here, character) level. Equivalent to CER. |
| **CER** | Character Error Rate | Character-level error rate. Used interchangeably with UER in this work. |
| **S / D / I / N** | Substitutions / Deletions / Insertions / reference word count | Building blocks of the WER formula. |
| **pp** | Percentage points | Absolute difference between two percentages (e.g., 97.93% − 72.97% = 24.96 pp). |
| **CTC** | Connectionist Temporal Classification | Loss/decoding scheme that aligns audio frames to text without frame-level labels. |
| **LM** | Language Model | A model of sequence probability over text. Not used here (pure CTC decoding). |
| **LR** | Learning Rate | Optimiser step size. |
| **AdamW** | Adam with decoupled Weight decay | The optimiser used for training. |
| **MLP** | Multi-Layer Perceptron | Feed-forward block inside each Transformer layer. |
| **RNN** | Recurrent Neural Network | Older sequence architecture; not used here (Transformer-based throughout). |
| **CNN** | Convolutional Neural Network | Used in the wav2vec 2.0 feature extractor on raw waveform. |
| **FFN** | Feed-Forward Network | Same as the MLP block inside Transformers. |
| **GPU** | Graphics Processing Unit | Hardware accelerator. One 16 GB GPU used here. |
| **VRAM** | Video RAM | GPU memory. |
| **OOM** | Out Of Memory | Training failure when a batch exceeds VRAM. |
| **TF32** | TensorFloat-32 | NVIDIA Ampere-era reduced-precision matmul mode. |
| **bf16 / bfloat16** | Brain Floating Point 16 | 16-bit float with fp32 exponent range; used for mixed-precision training. |
| **fp16 / fp32** | Floating-Point 16 / 32 | Standard 16-bit / 32-bit floating-point formats. |
| **DDP** | Distributed Data Parallel | Replicates the whole model per GPU, shards batches. |
| **FSDP** | Fully Sharded Data Parallel | Shards parameters/gradients/optimizer state across GPUs. |
| **FSDP v2** | FSDP version 2 | The PyTorch FSDP2 API used in the 1B config. |
| **DP / TP / PP / SDP** | Data / Tensor / Pipeline / Shard Data Parallelism | Forms of distributed parallelism. Checkpoint file names (`pp_00/tp_00/sdp_00.pt`) reflect these axes. |
| **AMP** | Automatic Mixed Precision | Framework feature that selectively uses lower-precision math. |
| **SGD** | Stochastic Gradient Descent | Family of optimisation algorithms; AdamW is a variant. |
| **TSV** | Tab-Separated Values | Audio manifest file format (`train.tsv`). |
| **WRD** | "Word" transcript file | One-transcript-per-line file accompanying each TSV. |
| **WAV** | Waveform Audio File Format | Uncompressed 16-bit PCM audio container used here. |
| **PCM** | Pulse-Code Modulation | Digital audio encoding; "16-bit PCM" = 16-bit signed integer samples. |
| **Hz / kHz** | Hertz / kilohertz | Sample-rate unit. 16 kHz = 16,000 samples per second. |
| **flac / ogg** | Audio codecs | Compressed audio formats supported by the loader (not used for RegSpeech12). |
| **fbank** | Log-mel filterbank features | Hand-crafted spectral features; **not used** (waveform input instead). |
| **SpecAugment** | Spectrogram Augmentation | Frequency/time masking augmentation; disabled in this project. |
| **CLI** | Command-Line Interface | How training and eval are invoked (`python -m ...`). |
| **YAML** | YAML Ain't Markup Language | Configuration file format used for all train/eval configs. |
| **FLEURS** | Few-shot Learning Evaluation of Universal Representations of Speech | Reference multilingual ASR benchmark (example dataset, not used in our results). |
| **MLS** | Multilingual LibriSpeech | Reference multilingual ASR dataset (example only, not used in our results). |
| **HF** | Hugging Face | The datasets / model hub (referenced in example scripts only). |
| **NLP** | Natural Language Processing | Broader field containing ASR. |
| **SSL** | Self-Supervised Learning | The pre-training paradigm behind wav2vec 2.0. |
| **LLM** | Large Language Model | The LLaMA-based decoder variant of OmniLingual ASR. |
| **LLaMA** | Large Language Model Meta AI | Meta's decoder-only Transformer LM, used in the 1B variant. |
| **API** | Application Programming Interface | — |
| **IDE** | Integrated Development Environment | — |
| **RegSpeech12** | Regional Speech (12 dialects) | The Kaggle Bengali regional-dialect dataset used for fine-tuning. |
| **omniASR** | Omnilingual ASR | Meta's multilingual ASR model family (our base model). |
| **OOV** | Out-Of-Vocabulary | Tokens/words the tokenizer can't represent directly. |
| **PEFT / LoRA / Adapter** | Parameter-Efficient Fine-Tuning / Low-Rank Adaptation / Adapter modules | Alternative fine-tuning strategies; **not used** — I do full fine-tuning. |
| **EPS / ε** | Epsilon | Small constant in AdamW's denominator (1e-08 here) for numerical stability. |
| **β₁, β₂** | Beta coefficients | AdamW momentum coefficients; (0.9, 0.98) here. |
| **wd** | Weight decay | AdamW regularisation strength (0.01 here). |
| **step** | Optimizer update | One forward + backward + optimizer.step(). |
| **epoch** | One pass over training set | 17,049 training samples = one epoch. |
| **ckpt** | Checkpoint | Saved model snapshot. |
| **RT / RTF** | Real-Time / Real-Time Factor | Throughput metric (not reported here). |
| **AIUB** | American International University–Bangladesh | Host institution. |
