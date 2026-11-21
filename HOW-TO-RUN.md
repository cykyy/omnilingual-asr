# How to Run (RegSpeech12 thesis experiments)

See `REGSPEECH12.md` for an overview and `FINETUNING_GUIDE.md` for details.

## Train (released model, 1 GPU, ~6.4 h)

```bash
python -m workflows.recipes.wav2vec2.asr \
    --config-file workflows/recipes/wav2vec2/asr/configs/regspeech12-80hr.yaml \
    output/ 2>&1 | tee log.log
```

## Evaluate on the test split

Fine-tuned checkpoint (reported: WER 72.97%, UER 35.44%):

```bash
python -m workflows.recipes.wav2vec2.asr.eval \
    --config model.path=output/ws_1.ecbbc39f/checkpoints/step_10000/model \
    --config model.family=wav2vec2_asr \
    --config model.arch=300m_v2 \
    --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml \
    eval_step10000/
```

Zero-shot base model (reported: WER 97.93%, UER 81.74%):

```bash
python -m workflows.recipes.wav2vec2.asr.eval \
    --config model.name=omniASR_CTC_300M_v2 \
    --config model.path=null \
    --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml \
    eval_baseline/
```

TensorBoard logs need: `pip install tensorboard`

## Tokenizer location

If the tokenizer file is needed directly, fairseq2 caches it under `~/.cache/fairseq2/assets/`:

```bash
find ~ -name "*omniASR_tokenizer*.model" 2>/dev/null
```

## Package the model for release

```bash
mkdir -p release_model
# Weights from the final step of the real run (NOT a test run)
cp output/ws_1.ecbbc39f/checkpoints/step_10000/model/pp_00/tp_00/sdp_00.pt release_model/model.pt
cp release/bn-regional-asr-300m/README.md release/bn-regional-asr-300m/config.yaml release_model/
ls -lh release_model/
```
