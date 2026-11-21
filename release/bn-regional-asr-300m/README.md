---
language:
  - bn
tags:
  - asr
  - speech-recognition
  - wav2vec2
  - ctc
  - bengali
  - dialect
license: mit
pipeline_tag: automatic-speech-recognition
model-index:
  - name: bn-regional-asr-300m
    results:
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: regspeech12
          name: RegSpeech12 (test split, 2,132 clips)
          split: test
        metrics:
          - type: wer
            value: 72.97
            name: Test WER (strict, no text normalization)
          - type: cer
            value: 35.44
            name: Test UER / CER (strict)
---

# Bengali Regional ASR (300M)

`omniASR_CTC_300M_v2` from Meta's [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr),
fully fine-tuned on [RegSpeech12](https://arxiv.org/abs/2510.24096), a corpus of spontaneous Bengali
speech from 12 regions of Bangladesh.

Developed for the BSc thesis *Fine-Tuning Multilingual ASR for Bengali Regional Dialects: A Zero-Shot
and Adaptation Study* (American International University-Bangladesh, 2026) by Rayhan Mia, Sudipta Shikha
Ausmita, Prajnadip Das Samya and Farduna Tajkia, supervised by Aminun Nahar.

- Thesis overview, per-dialect results and test transcripts: https://github.com/cykyy/thesis-omni-v2
- Training and evaluation code, configs and logs: https://github.com/cykyy/omnilingual-asr/tree/rayth1

## Results (RegSpeech12 test split)

| Model | WER | UER (≈ CER) | CTC loss |
|-------|-----|-------------|----------|
| omniASR_CTC_300M_v2, zero-shot | 97.93% | 81.74% | 1482.45 |
| **This model** (fine-tuned, step 10,000) | **72.97%** | **35.44%** | 393.83 |

Scoring is strict: hypotheses and references are compared as-is, with no punctuation removal or
Unicode normalization. With punctuation removed before scoring, the test WER is 68.01%.

### Per dialect (test split)

| Dialect | Clips | WER | CER |
|---------|------:|----:|----:|
| Tangail | 127 | 46.37% | 15.34% |
| Comilla | 32 | 48.19% | 15.87% |
| Habiganj | 117 | 61.91% | 24.00% |
| Narail | 186 | 65.80% | 29.19% |
| Narsingdi | 137 | 71.42% | 30.57% |
| Barishal | 101 | 72.89% | 36.09% |
| Sylhet | 762 | 74.88% | 38.66% |
| Rangpur | 130 | 76.24% | 39.03% |
| Sandwip | 131 | 79.15% | 40.26% |
| Chittagong | 176 | 82.20% | 38.57% |
| Noakhali | 28 | 82.29% | 44.62% |
| Kishoreganj | 205 | 87.05% | 47.29% |

Re-scored from the released test transcripts (CER counts characters including spaces). Noakhali and
Comilla have too few clips for firm conclusions.

## Training

| Setting | Value |
|---------|-------|
| Base model | omniASR_CTC_300M_v2 (~326M parameters, wav2vec2 encoder + CTC head) |
| Tokenizer | omniASR_tokenizer_written_v2 (character level) |
| Training data | RegSpeech12 train split: 17,049 clips, ~80 h (validation: 2,132 clips, ~10 h) |
| Steps | 10,000 (~13.6 epochs) |
| Batching | Length-based, up to 1,000,000 audio samples per batch; gradient accumulation 8 (~23 clips per update) |
| Optimizer | AdamW, lr 5e-05, β = (0.9, 0.98), weight decay 0.01 |
| LR schedule | Tri-stage: 10% warmup, 40% hold, 50% decay |
| Precision | bfloat16 mixed precision |
| Hardware | 1 GPU, ~6.4 hours |
| Augmentation | None |
| Decoding | Greedy CTC (no language model) |

The full resolved training config is in `config.yaml`
(source: `workflows/recipes/wav2vec2/asr/configs/regspeech12-80hr.yaml`).

## Usage

Install the code from the repository branch, which registers this model as `bn_regional_CTC_300M_v2`
and downloads `model.pt` from this page:

```bash
git clone -b rayth1 https://github.com/cykyy/omnilingual-asr.git
cd omnilingual-asr
pip install -e .
python run_pipeline.py path/to/audio.wav
```

Or in Python:

```python
import torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(
    model_card="bn_regional_CTC_300M_v2",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
print(pipeline.transcribe(["audio.wav"], batch_size=1)[0])
```

To load a downloaded copy of `model.pt`, create `~/.config/fairseq2/assets/model/bn_regional.yaml`:

```yaml
name: bn_regional_CTC_300M_v2
model_family: wav2vec2_asr
model_arch: 300m_v2
checkpoint: /path/to/model.pt
tokenizer_ref: omniASR_tokenizer_written_v2
```

Audio must be 16 kHz mono, up to 40 seconds.

## Limitations

- A WER of ~73% (strict) is far from production quality. The model is a research baseline and is
  useful mainly as a draft transcript for human correction.
- Trained on ~80 hours spread across 12 dialects; accuracy varies widely by dialect (46–87% WER, see above).
- No language model is used during decoding.
- Results come from a single training run (one seed).
- Not evaluated outside RegSpeech12.

## License

Model weights: MIT. The base model, omniASR_CTC_300M_v2, is released by Meta under Apache-2.0.
Check the RegSpeech12 dataset terms before commercial use.
