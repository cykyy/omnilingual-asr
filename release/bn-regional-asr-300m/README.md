---
language:
  - bn
tags:
  - asr
  - speech-recognition
  - wav2vec2
  - ctc
  - bengali
license: mit
base_model: facebook/omniASR_CTC_300M_v2
---

# Bengali Regional ASR Model (300M)

Fine-tuned version of `omniASR_CTC_300M_v2` on RegSpeech12 Bengali regional speech dataset.

## Model Details
- **Base Model**: omniASR_CTC_300M_v2 (300M parameters)
- **Tokenizer**: omniASR_tokenizer_written_v2
- **Training Data**: RegSpeech12 (80hr train, 10hr valid)
- **Training Steps**: 10,000
- **Learning Rate**: 5e-05

## Usage

### With OmniLingual ASR

1. Create model card at `~/.config/fairseq2/assets/model/bn_regional.yaml`:
```yaml
name: bn_regional_CTC_300M_v2
model_family: wav2vec2_asr
model_arch: 300m_v2
checkpoint: /path/to/model.pt
tokenizer_ref: omniASR_tokenizer_written_v2
```

2. Use with run_pipeline.py or ASRInferencePipeline

## License
MIT
