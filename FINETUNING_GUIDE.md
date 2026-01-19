# Fine-Tuning Guide for OmniLingual ASR

This guide covers fine-tuning the OmniLingual ASR models on custom datasets, with proven commands from actual training sessions.

## Table of Contents

1. [Quick Start Overview](#quick-start-overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Configuration](#training-configuration)
4. [Running Training](#running-training)
5. [Evaluation](#evaluation)
6. [Running Inference with Fine-tuned Model](#running-inference-with-fine-tuned-model)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start Overview

Fine-tuning adapts a pre-trained ASR model to your specific domain, language, or accent. The process:

1. Prepare dataset in manifest format (TSV + WRD files)
2. Configure training parameters (learning rate, steps, batch size)
3. Run training with `python -m workflows.recipes.wav2vec2.asr`
4. Evaluate checkpoints to find best model
5. Register model for inference

**Prerequisites:**
- GPU with 16GB+ VRAM (24GB recommended)
- Dataset with audio and transcriptions
- Audio files: 16kHz, mono, WAV format

---

## Dataset Preparation

### Manifest Format

Two files are required for each split:

**`train.tsv`** - Audio paths and sample counts:
```
/path/to/audio/directory
file1.wav	48000
file2.wav	96000
```
- First line: root directory containing audio files
- Subsequent lines: `filename<TAB>num_samples` (samples = duration × 16000)

**`train.wrd`** - Transcriptions (one per line, matching TSV order):
```
this is the first transcription
second audio transcription here
```

### Directory Structure

```
dataset/my_dataset/
├── data_manifest/
│   ├── train.tsv
│   ├── train.wrd
│   ├── valid.tsv
│   └── valid.wrd
└── wavs/
    ├── file1.wav
    └── file2.wav
```

### Audio Requirements

- **Format**: WAV (16-bit PCM)
- **Sample rate**: 16kHz
- **Channels**: Mono
- **Duration**: 1-30 seconds per file recommended

---

## Training Configuration

See `workflows/recipes/wav2vec2/asr/configs/regspeech12-4gpu.yaml` for a complete working example.

### Key Parameters

```yaml
model:
  name: "omniASR_CTC_300M_v2"    # Pre-trained model
  family: "wav2vec2_asr"         # Model architecture

dataset:
  asr_task_config:
    max_num_elements: 2_500_000  # Controls batch size (reduce if OOM)

optimizer:
  config:
    lr: 5e-05                    # Learning rate (1e-05 for small datasets)

trainer:
  freeze_encoder_for_n_steps: 0  # 0 for fine-tuning, >0 for new decoder
  grad_accumulation:
    num_batches: 1               # Increase if reducing max_num_elements

regime:
  num_steps: 10_000              # Total training steps
  checkpoint_every_n_steps: 2000 # Checkpoint frequency
```

---

## Running Training

### Single GPU

```bash
python -m workflows.recipes.wav2vec2.asr \
    --config-file workflows/recipes/wav2vec2/asr/configs/your-config.yaml \
    output/
```

### Multi-GPU (Recommended)

```bash
torchrun --nproc_per_node=4 \
    -m workflows.recipes.wav2vec2.asr \
    --config-file workflows/recipes/wav2vec2/asr/configs/regspeech12-4gpu.yaml \
    output/
```

Training creates checkpoints at `output/ws_1.XXXXXXXX/checkpoints/step_N/`.

---

## Evaluation

**This section documents the correct evaluation process, which was the main pain point during development.**

### Checkpoint Path Format

Checkpoints are saved in a distributed format. The correct path is either:
- The `model/` directory: `output/ws_1.XXXXXXXX/checkpoints/step_10000/model`
- Or the specific file: `output/ws_1.XXXXXXXX/checkpoints/step_10000/model/pp_00/tp_00/sdp_00.pt`

### Required Parameters for Evaluation

When evaluating a checkpoint, you **must** specify:
- `model.path` - path to the checkpoint directory
- `model.family` - architecture family (`wav2vec2_asr`)
- `model.arch` - model architecture (`300m_v2`)

### Evaluating a Fine-tuned Checkpoint

**Working command (proven):**

```bash
python -m workflows.recipes.wav2vec2.asr.eval \
    --config model.path=output/ws_1.ecbbc39f/checkpoints/step_10000/model \
    --config model.family=wav2vec2_asr \
    --config model.arch=300m_v2 \
    --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml \
    eval_step10000/
```

### Zero-Shot Baseline Evaluation

To compare against the pre-trained model without fine-tuning:

```bash
python -m workflows.recipes.wav2vec2.asr.eval \
    --config model.name=omniASR_CTC_300M_v2 \
    --config model.path=null \
    --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml \
    eval_baseline/
```

### Eval Config Example

See `workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml`:

```yaml
model:
  name: "omniASR_CTC_300M_v2"
  path: "output/ws_1.ecbbc39f/checkpoints/step_10000"
  family: "wav2vec2_asr"

tokenizer:
  name: "omniASR_tokenizer_written_v2"

dataset:
  name: "regspeech12"
  valid_split: "test"
  storage_mode: "MANIFEST"
  task_mode: "ASR"
  # ... (see file for full config)
```

### Evaluation Output

The evaluation reports:
- **WER (Word Error Rate)**: Primary metric
- **UER (Unit Error Rate)**: Character-level error rate
- **CTC Loss**: Model loss on test data

---

## Running Inference with Fine-tuned Model

After training, register your model to use with `run_pipeline.py`.

### Method 1: User Config Directory (No Code Changes)

Create a model card in the fairseq2 user config directory:

```bash
mkdir -p ~/.config/fairseq2/assets/model
```

Create `~/.config/fairseq2/assets/model/my_finetuned_model.yaml`:

```yaml
name: bn_regional_CTC_300M_v2
model_family: wav2vec2_asr
model_arch: 300m_v2
checkpoint: /root/omnilingual-asr/output/ws_1.ecbbc39f/checkpoints/step_10000/model/pp_00/tp_00/sdp_00.pt
tokenizer_ref: omniASR_tokenizer_written_v2
```

Then update `run_pipeline.py`:

```python
pipeline = ASRInferencePipeline(
    model_card="bn_regional_CTC_300M_v2",  # Your model card name
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
```

### Method 2: Source Code Cards (Requires Editable Install)

Add your model to `src/omnilingual_asr/cards/models/rc_models_v2.yaml`:

```yaml
---

name: bn_regional_CTC_300M_v2
model_family: wav2vec2_asr
model_arch: 300m_v2
checkpoint: /root/omnilingual-asr/output/ws_1.ecbbc39f/checkpoints/step_10000/model/pp_00/tp_00/sdp_00.pt
tokenizer_ref: omniASR_tokenizer_written_v2
```

Reinstall in editable mode to pick up changes:

```bash
pip install -e .
```

### Model Card Fields

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Unique identifier for your model | `bn_regional_CTC_300M_v2` |
| `model_family` | Architecture family | `wav2vec2_asr` |
| `model_arch` | Model architecture | `300m_v2` |
| `checkpoint` | Full path to `.pt` file | `/path/to/model/pp_00/tp_00/sdp_00.pt` |
| `tokenizer_ref` | Reference to tokenizer card | `omniASR_tokenizer_written_v2` |

### Using run_pipeline.py

```bash
python run_pipeline.py path/to/audio.wav
```

---

## Troubleshooting

### Evaluation Fails with "Model not found"

**Problem**: Using `model.name` instead of `model.path` for checkpoints.

**Solution**: For local checkpoints, use `--config model.path=...` with `model.family` and `model.arch`. Reserve `model.name` for registered model cards only.

### Evaluation Fails with "Cannot load checkpoint"

**Problem**: Wrong checkpoint path format.

**Solution**: Use the `model/` directory or full path to `sdp_00.pt`:
```bash
--config model.path=output/ws_1.XXXXX/checkpoints/step_10000/model
```

### Out of Memory (OOM)

**Solution**:
1. Reduce `max_num_elements` in config
2. Increase `grad_accumulation.num_batches` proportionally
3. Reduce `max_audio_len` for shorter sequences

### Loss Not Decreasing

**Check**:
- TSV and WRD files have same number of lines
- Audio is 16kHz mono WAV
- Learning rate appropriate for dataset size (smaller datasets = lower LR)

### High WER After Training

**Possible causes**:
- Dataset too small - collect more data
- Not enough training steps
- Learning rate too high/low - try adjusting

### NaN Loss

**Check**:
- Learning rate not too high (try 1e-05)
- No corrupted/empty audio files
- No empty transcriptions in WRD file

---

## Calculating Training Epochs

**Important for thesis/papers**: fairseq2 uses dynamic batching, so simple calculations like `steps × batch_size / samples` give incorrect results.

### Official Methods to Get Epoch Count

**Method 1: From Checkpoint (Most Reliable)**

```python
import torch
ckpt = torch.load('output/ws_1.XXXXX/checkpoints/step_10000/trainer/rank_00.pt',
                  map_location='cpu', weights_only=False)
completed_epochs = ckpt['_data_epoch_nr'] - 1  # _data_epoch_nr is current epoch (1-indexed)
print(f"Completed epochs: {completed_epochs}")
```

**Method 2: From Training Logs**

Look for lines like:
```
End of epoch 13 reached after step 9555.
```

Or check `Data Epoch: X` in Train Metrics.

**Method 3: Use the Calculate Epochs Script**

```bash
python scripts/calculate_epochs.py --output-dir output/ws_1.XXXXX --manifest dataset/YOUR_DATASET/data_manifest/train.tsv
```

This script reads from checkpoint, logs, and config to give you accurate epoch counts.

### Why Config-Based Estimates Are Wrong

The naive calculation:
```
epochs = (steps × batch_size × grad_accum) / training_samples
```

Gives **incorrect results** because:
- fairseq2 uses LENGTH batching (batches vary by audio duration)
- Actual batch sizes are ~22-23, not the configured 64
- Dynamic batching optimizes GPU memory, not sample count

**Example**: For 10,000 steps with config batch_size=8, grad_accum=8:
- Naive estimate: 37.5 epochs
- Actual (from checkpoint): 13.6 epochs

### Training Summary Example

| Parameter | Value |
|-----------|-------|
| Training samples | 17,049 |
| Training steps | 10,000 |
| Total examples processed | 231,881 |
| **Epochs** | **13.6** |
| Training time | ~6.4 hours |

---

## Extended Training Experiments

### Available Training Configs

| Config | Model | Epochs | GPUs | Est. Time |
|--------|-------|--------|------|-----------|
| `regspeech12-4gpu.yaml` | 300M CTC | ~54 | 4 | ~10 hrs |
| `regspeech12-300m-50ep-4gpu.yaml` | 300M CTC | ~50 | 4 | ~10 hrs |
| `regspeech12-1b-llm-4gpu.yaml` | 1B LLM | ~13.6 | 4 | ~15 hrs |

### 50-Epoch Training (300M CTC on 4 GPUs)

For extended training to improve WER:

```bash
torchrun --nproc_per_node=4 \
    -m workflows.recipes.wav2vec2.asr \
    --config-file workflows/recipes/wav2vec2/asr/configs/regspeech12-300m-50ep-4gpu.yaml \
    output/
```

**Config highlights** (`regspeech12-300m-50ep-4gpu.yaml`):
- Steps: 9,500 (≈50 epochs on 4 GPUs)
- Learning rate: 3e-05 (lower for longer training)
- Validation: Every 500 steps
- Checkpoints: Every 1000 steps, keeps best 3 by WER

**Evaluate after training:**
```bash
python -m workflows.recipes.wav2vec2.asr.eval \
    --config model.path=output/ws_4.XXXXXXXX/checkpoints/step_9500/model \
    --config model.family=wav2vec2_asr \
    --config model.arch=300m_v2 \
    --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml \
    eval_300m_50ep/
```

### 1B LLM Training (4 GPUs with FSDP)

For training larger LLM-based models:

```bash
torchrun --nproc_per_node=4 \
    -m workflows.recipes.wav2vec2.asr \
    --config-file workflows/recipes/wav2vec2/asr/configs/regspeech12-1b-llm-4gpu.yaml \
    output/
```

**Requirements:**
- 4x GPUs with 24GB+ VRAM (e.g., RTX 3090)
- FSDP enabled (automatic in config)
- ~15-20 hours for ~13.6 epochs

**Evaluate:**
```bash
python -m workflows.recipes.wav2vec2.asr.eval \
    --config model.path=output/ws_4.XXXXXXXX/checkpoints/step_10000/model \
    --config model.family=wav2vec2_llama \
    --config model.arch=1b_v2 \
    --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-1b-llm-test.yaml \
    eval_1b_llm/
```

### CTC vs LLM Comparison

| Aspect | CTC | LLM (LLaMA decoder) |
|--------|-----|---------------------|
| Output modeling | Conditionally independent | Autoregressive |
| Language model | None built-in | Implicit in decoder |
| Inference speed | Fast (parallel) | Slower (sequential) |
| Memory (1B model) | ~10 GB | ~30 GB (needs FSDP) |
| Typical WER | Higher | 5-15% relative improvement |
| Best for | Streaming, real-time | Offline, accuracy-critical |

### Experimental Plan for Thesis

| Exp | Model | Epochs | Config | Purpose |
|-----|-------|--------|--------|---------|
| 1 | 300M CTC | 13.6 | `regspeech12-4gpu.yaml` | Baseline |
| 2 | 300M CTC | 50 | `regspeech12-300m-50ep-4gpu.yaml` | Extended training |
| 3 | 1B LLM | 13.6 | `regspeech12-1b-llm-4gpu.yaml` | Architecture comparison |

**Expected results** (RegSpeech12 Bengali regional):

| Model | Zero-shot | Fine-tuned |
|-------|-----------|------------|
| 300M CTC (13 ep) | 97.9% WER | 73.5% WER |
| 300M CTC (50 ep) | - | ~60-65% WER (est.) |
| 1B LLM (13 ep) | TBD | ~60-68% WER (est.) |

**Note**: Achieving <20% WER on 80hr regional data typically requires additional techniques:
- More training data (500+ hours)
- Data augmentation (speed perturbation, noise)
- External language model integration
- Domain-specific tokenizer

---

## Publishing Fine-tuned Models

### HuggingFace

```bash
pip install huggingface_hub
python -c "from huggingface_hub import login; login()"

# Create repo and upload
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('YOUR_USERNAME/model-name', repo_type='model')
api.upload_folder(
    folder_path='release/your-model/',
    repo_id='YOUR_USERNAME/model-name',
    repo_type='model'
)
"
```

### GitHub Release

```bash
git tag -a v1.0.0 -m "Model release"
git push origin v1.0.0
gh release create v1.0.0 --title "Model v1.0.0" --notes "Description here"
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Train 300M (4 GPU) | `torchrun --nproc_per_node=4 -m workflows.recipes.wav2vec2.asr --config-file workflows/recipes/wav2vec2/asr/configs/regspeech12-4gpu.yaml output/` |
| Train 300M 50ep (4 GPU) | `torchrun --nproc_per_node=4 -m workflows.recipes.wav2vec2.asr --config-file workflows/recipes/wav2vec2/asr/configs/regspeech12-300m-50ep-4gpu.yaml output/` |
| Train 1B LLM (4 GPU) | `torchrun --nproc_per_node=4 -m workflows.recipes.wav2vec2.asr --config-file workflows/recipes/wav2vec2/asr/configs/regspeech12-1b-llm-4gpu.yaml output/` |
| Eval CTC checkpoint | `python -m workflows.recipes.wav2vec2.asr.eval --config model.path=PATH --config model.family=wav2vec2_asr --config model.arch=300m_v2 --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml output/` |
| Eval LLM checkpoint | `python -m workflows.recipes.wav2vec2.asr.eval --config model.path=PATH --config model.family=wav2vec2_llama --config model.arch=1b_v2 --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-1b-llm-test.yaml output/` |
| Eval baseline | `python -m workflows.recipes.wav2vec2.asr.eval --config model.name=omniASR_CTC_300M_v2 --config model.path=null --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml output/` |
| Inference | `python run_pipeline.py audio.wav` |
| Calculate epochs | `python scripts/calculate_epochs.py --output-dir output/ws_1.XXXXX --manifest path/to/train.tsv` |
