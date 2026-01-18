# Running Fine-Tuned CTC Model Inference

## Prerequisites
- Fine-tuned checkpoint at: `output/test_run_300m/ws_1.1e6d484a/checkpoints/step_100/`
- Audio file(s) in WAV format (16kHz, max 40 seconds)

## Setup

### 1. Create Asset Card
```bash
mkdir -p ~/.config/fairseq2/assets/model
```

Create `~/.config/fairseq2/assets/model/my_finetuned_model.yaml`:
```yaml
name: my_finetuned_model
model_family: wav2vec2_asr
model_arch: 300m
checkpoint: /root/omnilingual-asr/output/test_run_300m/ws_1.1e6d484a/checkpoints/step_100/model/pp_00/tp_00/sdp_00.pt
tokenizer_ref: omniASR_tokenizer_v1
```

## Run Inference

### Option A: Use run_pipeline.py
Edit `run_pipeline.py` to point to your audio file, then:
```bash
python run_pipeline.py
```

### Option B: Python Script
```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import torch

pipeline = ASRInferencePipeline(
    model_card="my_finetuned_model",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

transcriptions = pipeline.transcribe(["/path/to/audio.wav"], batch_size=1)
print(transcriptions[0])
```

## Notes
- CTC models do NOT support language conditioning (no `lang` parameter)
- Max audio length: 40 seconds
- Audio must be 16kHz WAV format
