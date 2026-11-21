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

Code, configs and evaluation logs: https://github.com/cykyy/omnilingual-asr/tree/rayth1

## Results (RegSpeech12 test split)

| Model | WER | UER (≈ CER) | CTC loss |
|-------|-----|-------------|----------|
| omniASR_CTC_300M_v2, zero-shot | 97.93% | 81.74% | 1482.45 |
| **This model** (fine-tuned, step 10,000) | **72.97%** | **35.44%** | 393.83 |

Scoring is strict: hypotheses and references are compared as-is, with no punctuation removal or
Unicode normalization. WER with punctuation removed is much lower.

## Training

| Setting | Value |
|---------|-------|
| Base model | omniASR_CTC_300M_v2 (325,494,996 parameters, wav2vec2 encoder + CTC head) |
| Tokenizer | omniASR_tokenizer_written_v2 (character level) |
| Training data | RegSpeech12 train split: 17,049 clips, ~80 h (validation: 2,132 clips, ~10 h) |
| Steps | 10,000 (13.6 epochs, measured from the checkpoint) |
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
- Trained on ~80 hours spread across many dialects; accuracy per dialect has not been measured yet.
- No language model is used during decoding.
- Results come from a single training run (one seed).
- Not evaluated outside RegSpeech12.

## License

Model weights: MIT. The base model, omniASR_CTC_300M_v2, is released by Meta under Apache-2.0.
Check the RegSpeech12 dataset terms before commercial use.
