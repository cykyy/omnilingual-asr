# Fine-Tuning Guide for OmniLingual ASR

This guide covers best practices for fine-tuning the OmniLingual ASR models on custom datasets.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Config Parameter Reference](#config-parameter-reference)
4. [Best Practices](#best-practices)
5. [Example Configs by Scale](#example-configs-by-scale)
6. [Running Training](#running-training)
7. [Post-Training Inference](#post-training-inference)
8. [Publishing Your Fine-Tuned Model](#publishing-your-fine-tuned-model)

---

## Overview

The OmniLingual ASR system supports fine-tuning pre-trained models on custom audio datasets. The framework uses a YAML-based configuration system with the following key components:

- **Model**: Pre-trained ASR model (CTC or LLM-based)
- **Dataset**: Custom audio with transcriptions in manifest format
- **Tokenizer**: Text tokenizer matching the model version
- **Optimizer**: AdamW with configurable learning rate
- **Trainer**: Controls freezing, gradient accumulation, precision
- **Regime**: Training schedule (steps, checkpoints, validation)

### Available Models

| Model | Parameters | Tokenizer |
|-------|------------|-----------|
| `omniASR_CTC_300M` | 300M | `omniASR_tokenizer_v1` |
| `omniASR_CTC_300M_v2` | 300M | `omniASR_tokenizer_written_v2` |
| `omniASR_CTC_1B` | 1B | `omniASR_tokenizer_v1` |

---

## Dataset Preparation

### Manifest Format (TSV + WRD)

The dataset requires two files:

**`train.tsv`** - Audio file paths with sample counts:
```
/path/to/audio/directory
file1.wav	48000
file2.wav	96000
file3.wav	72000
```

- First line: Root directory for audio files
- Subsequent lines: `filename<TAB>num_samples`
- `num_samples` = duration_seconds × sample_rate (typically 16kHz)

**`train.wrd`** - Transcriptions (one per line):
```
this is the first transcription
second audio transcription here
third example text
```

- Each line corresponds to the same line number in the TSV file
- Text should be normalized (lowercase, no punctuation for CTC models)

### Directory Structure

```
dataset/my_dataset/
├── data_manifest/
│   ├── train.tsv
│   ├── train.wrd
│   ├── valid.tsv    # Optional: separate validation set
│   └── valid.wrd
└── wavs/
    ├── file1.wav
    ├── file2.wav
    └── file3.wav
```

The TSV first line should be the path to the wavs directory relative to project root:
```
dataset/my_dataset/wavs
file1.wav	48000
file2.wav	96000
```

### Audio Requirements

- **Format**: WAV (16-bit PCM recommended)
- **Sample rate**: 16kHz (model native rate)
- **Channels**: Mono
- **Duration**: 1-30 seconds per file recommended

---

## Config Parameter Reference

### Model Section

```yaml
model:
  name: "omniASR_CTC_300M_v2"  # Pre-trained model name
  family: "wav2vec2_asr"       # Model architecture family
```

### Dataset Section

```yaml
dataset:
  name: "my_custom_dataset"      # Arbitrary name for logging
  storage_mode: "MANIFEST"       # Use manifest format
  train_split: "train"           # Name of train split files
  valid_split: "valid"           # Name of validation split files

  manifest_storage_config:
    read_text: true              # Load transcriptions from .wrd file

  asr_task_config:
    min_audio_len: 16_000        # Min samples (1 second at 16kHz)
    max_audio_len: 480_000       # Max samples (30 seconds at 16kHz)
    max_num_elements: 3_000_000  # Batch size control (see below)
    normalize_audio: true        # Normalize audio amplitude
    example_shuffle_window: 20   # Shuffle buffer size
```

**Batch Size Calculation:**
```
effective_batch_size ≈ max_num_elements / avg_audio_len
```
Example: `3,000,000 / 480,000 ≈ 6 samples` (worst case with max length audio)

### Optimizer Section

```yaml
optimizer:
  name: "adamw"
  config:
    lr: 1e-05        # Peak learning rate
    betas: [0.9, 0.98]
    eps: 1e-08
    weight_decay: 0.0  # Usually 0 for ASR fine-tuning
```

### Trainer Section

```yaml
trainer:
  freeze_encoder_for_n_steps: 0  # Steps to keep encoder frozen
  grad_accumulation:
    num_batches: 1               # Gradient accumulation steps
  mixed_precision:
    dtype: "bfloat16"            # Use bfloat16 for efficiency
```

### Regime Section

```yaml
regime:
  num_steps: 10000                     # Total training steps
  score_metric: "wer"                  # Metric for best checkpoint
  publish_metrics_every_n_steps: 50    # Logging frequency
  validate_after_n_steps: 0            # Start validation from step 0
  validate_every_n_steps: 500          # Validation frequency
  checkpoint_every_n_steps: 1000       # Checkpoint frequency
```

---

## Best Practices

### Learning Rate by Dataset Size

| Dataset Size | Recommended LR | Notes |
|--------------|----------------|-------|
| < 1 hour | `5e-06` to `1e-05` | Very low to prevent overfitting |
| 1-10 hours | `1e-05` | Conservative fine-tuning |
| 10-100 hours | `1e-05` to `5e-05` | Standard fine-tuning |
| 100+ hours | `5e-05` to `1e-04` | Can use higher LR |

### Encoder Freezing Strategy

- **Fine-tuning pre-trained model**: `freeze_encoder_for_n_steps: 0`
  - The encoder is already trained; update all weights

- **Training from encoder checkpoint**: `freeze_encoder_for_n_steps: 10_000`
  - Freeze encoder initially to stabilize decoder training
  - Unfreeze after decoder converges

### Learning Rate Schedule

The default tri-stage schedule:
1. **Warmup (10%)**: Linear increase from 1% to 100% of base LR
2. **Training (40%)**: Constant at base LR
3. **Decay (50%)**: Linear decrease to 5% of base LR

### Steps Calculation Formula

```
total_samples = num_audio_files
samples_per_step = batch_size × grad_accumulation
steps_per_epoch = total_samples / samples_per_step
num_steps = steps_per_epoch × desired_epochs
```

### Memory Optimization

For limited GPU memory:
1. Reduce `max_num_elements` (smaller batches)
2. Increase `grad_accumulation.num_batches`
3. Reduce `max_audio_len` (shorter sequences)

---

## Example Configs by Scale

### Dry Run (3 samples)

For testing the pipeline with minimal data:

```yaml
# configs/custom-tune-300m-v2.yaml
model:
  name: "omniASR_CTC_300M_v2"
  family: "wav2vec2_asr"

dataset:
  name: "custom_dry_run"
  storage_mode: "MANIFEST"
  train_split: "train"
  valid_split: "train"  # Use train as validation for dry run
  manifest_storage_config:
    read_text: true
  asr_task_config:
    min_audio_len: 16_000
    max_audio_len: 480_000
    max_num_elements: 3_000_000
    normalize_audio: true
    example_shuffle_window: 20

tokenizer:
  name: "omniASR_tokenizer_written_v2"

optimizer:
  name: "adamw"
  config:
    lr: 5e-06  # Very low LR
    betas: [0.9, 0.98]
    eps: 1e-08
    weight_decay: 0.0

trainer:
  freeze_encoder_for_n_steps: 0
  grad_accumulation:
    num_batches: 1
  mixed_precision:
    dtype: "bfloat16"

regime:
  num_steps: 500
  score_metric: "wer"
  publish_metrics_every_n_steps: 50
  validate_after_n_steps: 0
  validate_every_n_steps: 100
  checkpoint_every_n_steps: 100

gang: {}
common: {}
```

### Small Dataset (1-10 hours)

Approximately 360-3,600 samples (assuming 10s average):

```yaml
model:
  name: "omniASR_CTC_300M_v2"
  family: "wav2vec2_asr"

dataset:
  name: "small_custom_dataset"
  storage_mode: "MANIFEST"
  train_split: "train"
  valid_split: "valid"
  manifest_storage_config:
    read_text: true
  asr_task_config:
    min_audio_len: 16_000
    max_audio_len: 480_000
    max_num_elements: 3_000_000
    normalize_audio: true
    example_shuffle_window: 100

tokenizer:
  name: "omniASR_tokenizer_written_v2"

optimizer:
  name: "adamw"
  config:
    lr: 1e-05
    betas: [0.9, 0.98]
    eps: 1e-08
    weight_decay: 0.0

trainer:
  freeze_encoder_for_n_steps: 0
  grad_accumulation:
    num_batches: 2  # Effective batch ~12
  mixed_precision:
    dtype: "bfloat16"

regime:
  num_steps: 3000
  score_metric: "wer"
  publish_metrics_every_n_steps: 100
  validate_after_n_steps: 0
  validate_every_n_steps: 500
  checkpoint_every_n_steps: 500

gang: {}
common: {}
```

### Medium Dataset (10-50 hours)

Approximately 3,600-18,000 samples:

```yaml
model:
  name: "omniASR_CTC_300M_v2"
  family: "wav2vec2_asr"

dataset:
  name: "medium_custom_dataset"
  storage_mode: "MANIFEST"
  train_split: "train"
  valid_split: "valid"
  manifest_storage_config:
    read_text: true
  asr_task_config:
    min_audio_len: 16_000
    max_audio_len: 480_000
    max_num_elements: 4_000_000
    normalize_audio: true
    example_shuffle_window: 500

tokenizer:
  name: "omniASR_tokenizer_written_v2"

optimizer:
  name: "adamw"
  config:
    lr: 3e-05
    betas: [0.9, 0.98]
    eps: 1e-08
    weight_decay: 0.0

trainer:
  freeze_encoder_for_n_steps: 0
  grad_accumulation:
    num_batches: 4  # Effective batch ~32
  mixed_precision:
    dtype: "bfloat16"

regime:
  num_steps: 8000
  score_metric: "wer"
  publish_metrics_every_n_steps: 100
  validate_after_n_steps: 0
  validate_every_n_steps: 1000
  checkpoint_every_n_steps: 1000

gang: {}
common: {}
```

### Large Dataset (100+ hours with 80/10/10 split)

For 100 hours total (80 hours train, 10 hours valid, 10 hours test):
- ~28,800 training samples (assuming 10s average)
- Batch size 8, grad_accum 4 = effective batch 32
- 10 epochs ≈ 9,000 steps

```yaml
model:
  name: "omniASR_CTC_300M_v2"
  family: "wav2vec2_asr"

dataset:
  name: "large_custom_dataset"
  storage_mode: "MANIFEST"
  train_split: "train"
  valid_split: "valid"
  manifest_storage_config:
    read_text: true
  asr_task_config:
    min_audio_len: 16_000
    max_audio_len: 480_000
    max_num_elements: 5_000_000
    normalize_audio: true
    example_shuffle_window: 1000

tokenizer:
  name: "omniASR_tokenizer_written_v2"

optimizer:
  name: "adamw"
  config:
    lr: 5e-05
    betas: [0.9, 0.98]
    eps: 1e-08
    weight_decay: 0.01  # Light regularization for large datasets

trainer:
  freeze_encoder_for_n_steps: 0
  grad_accumulation:
    num_batches: 4
  mixed_precision:
    dtype: "bfloat16"

regime:
  num_steps: 10000
  score_metric: "wer"
  publish_metrics_every_n_steps: 100
  validate_after_n_steps: 0
  validate_every_n_steps: 1000
  checkpoint_every_n_steps: 2000

gang: {}
common: {}
```

---

## Running Training

### Basic Training Command

```bash
# Run training (output_dir is a positional argument)
python -m workflows.recipes.wav2vec2.asr \
    --config-file workflows/recipes/wav2vec2/asr/configs/custom-tune-300m-v2.yaml \
    output/ 2>&1 | tee asr.log
```

### Using the Pipeline Script

If using `run_pipeline.py`:

```bash
python run_pipeline.py \
    --config workflows/recipes/wav2vec2/asr/configs/custom-tune-300m-v2.yaml \
    --dataset-root /path/to/dataset \
    --output-dir ./output
```

### Multi-GPU Training

For distributed training across multiple GPUs, use `torchrun`:

```bash
torchrun --nproc_per_node=4 \
    -m workflows.recipes.wav2vec2.asr \
    --config-file workflows/recipes/wav2vec2/asr/configs/regspeech12-4gpu.yaml \
    output/
```

#### Multi-GPU Config Adjustments

When using multiple GPUs, adjust these parameters:

| Parameter | 1 GPU (16GB) | 4 GPUs (24GB each) | Reason |
|-----------|--------------|-------------------|--------|
| `max_num_elements` | 1,000,000 | 2,500,000 | More VRAM per GPU |
| `grad_accumulation` | 8 | 1 | 4 GPUs = 4x batch, no accum needed |
| `num_steps` | 10,000 | 10,000 | Same steps = more epochs |

#### GPU Configurations Comparison

| Setup | VRAM | Training Time (10K steps) | Cost |
|-------|------|---------------------------|------|
| 1x 16GB | 16GB | ~8 hours | ~$400 |
| 4x RTX 3060 (12GB) | 48GB | ~3.5 hours | ~$1,200 |
| 1x RTX 6000 Ada (48GB) | 48GB | ~2.5 hours | ~$6,500 |
| **4x RTX 3090 (24GB)** | **96GB** | **~1.5-2 hours** | ~$3,500 |

#### Example: 4x RTX 3090 Config

```yaml
# Key settings for 4x RTX 3090 (24GB each)
asr_task_config:
  max_num_elements: 2_500_000  # 2.5M per GPU

trainer:
  grad_accumulation:
    num_batches: 1  # No accumulation with 4 GPUs
```

Pre-made config: `configs/regspeech12-4gpu.yaml`

### Monitoring Training

- Check `output/` directory for checkpoints
- Monitor loss curves in training logs
- Validation WER should decrease over time

---

## Evaluating the Trained Model

After training completes, evaluate on the test set to get final WER metrics.

### Official Evaluation Command

```bash
python -m workflows.recipes.wav2vec2.asr.eval \
    --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml \
    eval_output/
```

### Creating an Eval Config

Create a config file (e.g., `eval/configs/my-eval.yaml`):

```yaml
# Model - point to your checkpoint
model:
  name: "omniASR_CTC_300M_v2"  # Base architecture
  path: "output/ws_1.xxxxx/checkpoints/step_10000"  # Your checkpoint

tokenizer:
  name: "omniASR_tokenizer_written_v2"

# Dataset - test split
dataset:
  name: "regspeech12"
  valid_split: "test"  # Evaluate on test set
  storage_mode: "MANIFEST"
  task_mode: "ASR"
  manifest_storage_config:
    read_text: true
  asr_task_config:
    min_audio_len: 16_000
    max_audio_len: 480_000
    max_num_elements: 1_000_000
    normalize_audio: true
    example_shuffle_window: 1  # No shuffle for eval

evaluator:
  amp: true
  amp_dtype: "bfloat16"

gang: {}
common: {}
```

### Evaluation Output

The evaluation will output:
- **WER (Word Error Rate)**: Primary metric for ASR quality
- **UER (Unit Error Rate)**: Character-level error rate
- **CTC Loss**: Model's loss on test data
- Transcriptions saved to `eval_output/transcriptions/`

### Comparing Checkpoints

Evaluate multiple checkpoints to find the best:

```bash
# Evaluate step 6000
python -m workflows.recipes.wav2vec2.asr.eval \
    --config model.path=output/ws_1.xxxxx/checkpoints/step_6000 \
    --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml \
    eval_step6000/

# Evaluate step 10000
python -m workflows.recipes.wav2vec2.asr.eval \
    --config model.path=output/ws_1.xxxxx/checkpoints/step_10000 \
    --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml \
    eval_step10000/
```

---

## Post-Training Inference

### Creating an Asset Card

After training, create an asset card to register your fine-tuned model:

```python
# save_model_card.py
from pathlib import Path

checkpoint_path = Path("output/checkpoints/step_500")
model_card = {
    "name": "my_finetuned_asr_v2",
    "base_model": "omniASR_CTC_300M_v2",
    "checkpoint": str(checkpoint_path),
    "tokenizer": "omniASR_tokenizer_written_v2"
}
```

### Running Inference

```python
from workflows.recipes.wav2vec2.asr.inference import load_model, transcribe

# Load fine-tuned model
model = load_model("path/to/checkpoint")

# Transcribe audio
result = transcribe(model, "audio.wav")
print(result)
```

### Batch Inference

```bash
python -m workflows.recipes.wav2vec2.asr.inference \
    --checkpoint output/checkpoints/step_500 \
    --input-dir /path/to/audio/files \
    --output results.json
```

---

## Publishing Your Fine-Tuned Model

After training, you can share your fine-tuned model on Hugging Face Hub or GitHub.

### Preparing the Model for Publishing

First, organize your checkpoint and metadata:

```bash
# Create a release directory
mkdir -p release/my-finetuned-asr-v2

# Copy the best checkpoint
cp -r output/checkpoints/step_500/* release/my-finetuned-asr-v2/

# Copy the config used for training
cp workflows/recipes/wav2vec2/asr/configs/custom-tune-300m-v2.yaml \
   release/my-finetuned-asr-v2/config.yaml
```

Create a model card (`release/my-finetuned-asr-v2/README.md`):

```markdown
# My Fine-Tuned ASR Model

## Model Description
Fine-tuned version of `omniASR_CTC_300M_v2` for [your language/domain].

## Training Details
- **Base Model**: omniASR_CTC_300M_v2
- **Tokenizer**: omniASR_tokenizer_written_v2
- **Training Data**: [Description of your dataset]
- **Training Steps**: 500
- **Learning Rate**: 5e-06

## Usage
See the [OmniLingual ASR repository](https://github.com/facebookresearch/omnilingual-asr) for inference instructions.

## License
[Specify license]
```

### Publishing to Hugging Face Hub

#### 1. Install Hugging Face CLI

```bash
pip install huggingface_hub
huggingface-cli login
```

#### 2. Create and Upload Repository

```python
from huggingface_hub import HfApi, create_repo

# Create repository
repo_id = "your-username/my-finetuned-asr-v2"
create_repo(repo_id, repo_type="model", exist_ok=True)

# Upload all files
api = HfApi()
api.upload_folder(
    folder_path="release/my-finetuned-asr-v2",
    repo_id=repo_id,
    repo_type="model"
)

print(f"Model uploaded to: https://huggingface.co/{repo_id}")
```

Or use the CLI:

```bash
# Create repo and upload
huggingface-cli repo create my-finetuned-asr-v2 --type model
cd release/my-finetuned-asr-v2
huggingface-cli upload your-username/my-finetuned-asr-v2 . .
```

#### 3. Add Model Card on Hugging Face

Edit the model card on Hugging Face to include proper tags:

```yaml
---
language:
  - bn  # Example: Bengali
tags:
  - asr
  - speech-recognition
  - wav2vec2
  - ctc
license: mit
base_model: facebook/omniASR_CTC_300M_v2
---
```

### Publishing to GitHub

#### 1. Using Git LFS for Large Files

Checkpoint files are large, so use Git LFS:

```bash
# Install Git LFS
sudo apt-get install git-lfs
git lfs install

# In your repository
cd your-repo
git lfs track "*.pt"
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
```

#### 2. Create a GitHub Release

```bash
# Tag your release
git tag -a v1.0.0 -m "Fine-tuned ASR model release"
git push origin v1.0.0
```

Then on GitHub:
1. Go to **Releases** → **Create a new release**
2. Select your tag
3. Upload checkpoint files as release assets
4. Add release notes with model details

#### 3. Using GitHub CLI

```bash
# Install GitHub CLI if needed
# https://cli.github.com/

# Create release with assets
gh release create v1.0.0 \
    release/my-finetuned-asr-v2/model.pt \
    release/my-finetuned-asr-v2/config.yaml \
    --title "Fine-Tuned ASR Model v1.0.0" \
    --notes "Fine-tuned omniASR_CTC_300M_v2 on custom dataset"
```

### Downloading and Using Published Models

#### From Hugging Face

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Download entire model
local_path = snapshot_download(repo_id="your-username/my-finetuned-asr-v2")

# Or download specific file
checkpoint = hf_hub_download(
    repo_id="your-username/my-finetuned-asr-v2",
    filename="model.pt"
)
```

#### From GitHub Release

```bash
# Download release assets
gh release download v1.0.0 --repo your-username/your-repo --dir ./downloaded_model

# Or with curl
curl -L -o model.pt \
    https://github.com/your-username/your-repo/releases/download/v1.0.0/model.pt
```

### Best Practices for Model Publishing

1. **Include all necessary files**:
   - Model checkpoint (`.pt` or `.safetensors`)
   - Training config (`.yaml`)
   - README with usage instructions

2. **Document training details**:
   - Base model and tokenizer used
   - Dataset description (without sharing private data)
   - Hyperparameters (LR, steps, batch size)
   - Evaluation metrics (WER, CER)

3. **Specify license clearly**:
   - Check base model license requirements
   - Add LICENSE file to your release

4. **Version your releases**:
   - Use semantic versioning (v1.0.0, v1.1.0, etc.)
   - Tag commits corresponding to checkpoints

---

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce `max_num_elements`
- Increase `grad_accumulation.num_batches`
- Use shorter `max_audio_len`

**Loss Not Decreasing**
- Check dataset format (TSV/WRD alignment)
- Verify audio is 16kHz mono
- Try lower learning rate

**High WER After Training**
- More training steps needed
- Dataset too small or noisy
- Try higher learning rate if underfitting

**NaN Loss**
- Learning rate too high
- Corrupted audio files
- Check for empty transcriptions

### Validation Tips

Before full training, always:
1. Run with 3-5 samples first (dry run)
2. Verify checkpoint saves correctly
3. Test inference on checkpoint
4. Then scale to full dataset

---

## References

- [Wav2Vec 2.0 Paper](https://arxiv.org/abs/2006.11477)
- [fairseq2 Documentation](https://github.com/facebookresearch/fairseq2)
- [OmniLingual ASR Repository](https://github.com/facebookresearch/omnilingual-asr)
