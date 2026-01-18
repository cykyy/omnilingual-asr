Test file. not working. crashes. Ignore this file.
```
cat <<EOF > ~/.config/fairseq2/assets/dataset/omn_ctc_300m_dry_run_v2.yaml
name: "omn_ctc_300m_dry_run_v2"
dataset_family: "manifest_asr_dataset"
# Providing all common field aliases to ensure one hits:
data: "/root/omnilingual-asr/dataset/dry_run/data_manifest"
path: "/root/omnilingual-asr/dataset/dry_run/data_manifest"
location: "/root/omnilingual-asr/dataset/dry_run/data_manifest"
manifest_dir: "/root/omnilingual-asr/dataset/dry_run/data_manifest"
EOF
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