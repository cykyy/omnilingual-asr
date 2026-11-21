# Running Inference with the Fine-Tuned Bengali Regional Model

The fine-tuned model `bn_regional_CTC_300M_v2` (omniASR_CTC_300M_v2 fine-tuned on RegSpeech12,
test WER 72.97%) is registered in `src/omnilingual_asr/cards/models/rc_models_v2.yaml`.
Its checkpoint is downloaded from Hugging Face: https://huggingface.co/cykyy/bn-regional-asr-300m

## Setup

```bash
pip install -e .   # editable install so the model card in src/ is picked up
```

## Run Inference

### Option A: Command line

```bash
python run_pipeline.py path/to/audio.wav
```

Compare with the zero-shot base model:

```bash
python run_pipeline.py --model omniASR_CTC_300M_v2 path/to/audio.wav
```

### Option B: Python

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import torch

pipeline = ASRInferencePipeline(
    model_card="bn_regional_CTC_300M_v2",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

transcriptions = pipeline.transcribe(["/path/to/audio.wav"], batch_size=1)
print(transcriptions[0])
```

### Using a local checkpoint

If you trained your own model, create `~/.config/fairseq2/assets/model/my_model.yaml`:

```yaml
name: my_model
model_family: wav2vec2_asr
model_arch: 300m_v2
checkpoint: /path/to/output/ws_1.XXXXXXXX/checkpoints/step_10000/model/pp_00/tp_00/sdp_00.pt
tokenizer_ref: omniASR_tokenizer_written_v2
```

Then run `python run_pipeline.py --model my_model audio.wav`.

## Notes
- CTC models do NOT support language conditioning (no `lang` parameter)
- Max audio length: 40 seconds
- Audio must be 16 kHz mono WAV
