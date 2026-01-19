To run WER with test split:
```
python -m workflows.recipes.wav2vec2.asr.eval --config model.path=output/ws_1.ecbbc39f/checkpoints/step_10000/model --config model.family=wav2vec2_asr --config model.arch=300m_v2 --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml eval_zero_shot_omniASR_CTC_300M_v2/
```

pip install tensorboard

# Tokenizer issue
(venv) root@ubuntu:~/omnilingual-asr# find /root -name "*omniASR_tokenizer*.model" 2>/dev/null
/root/.cache/fairseq2/assets/b86047ffa9089216c2972a21/omniASR_tokenizer.model

```
python -m workflows.recipes.wav2vec2.asr output/test_run_300m --config-file workflows/recipes/wav2vec2/asr/configs/custom-tune-300m.yaml
```
```
python -m workflows.recipes.wav2vec2.asr output/test_run_300m --config-file workflows/recipes/wav2vec2/asr/configs/custom-tune-300m-v2.yaml
```
# 1. Create the folder
mkdir -p release_model

# 2. Copy the Tokenizer you just found
cp /root/.cache/fairseq2/assets/b86047ffa9089216c2972a21/omniASR_tokenizer.model release_model/tokenizer.model

# 3. Copy the Model Weights (from the successful step 100)
cp output/test_run_300m/ws_1.1e6d484a/checkpoints/step_100/model/pp_00/tp_00/sdp_00.pt release_model/model.pt

# 4. Verify
ls -lh release_model/