#!/usr/bin/env bash
# Re-run of the thesis training on a rented GPU, recording everything needed to
# document it: hardware, software versions, timings, logs, epoch count,
# checkpoint hash and test results. See scripts/rerun/README.md.
#
# Run from the repo root, inside tmux so a dropped SSH session does not kill it:
#
#   bash scripts/rerun/rerun.sh                 # all steps, seed 2 (same as the thesis)
#   SEED=3 bash scripts/rerun/rerun.sh          # a second seed
#   bash scripts/rerun/rerun.sh train eval_new  # only some steps
#
# Steps, in order: env setup data eval_baseline eval_released train eval_new
# epochs lm pack. Each writes its own log to rerun_artifacts/<step>.log.

set -euo pipefail

SEED="${SEED:-2}"
ART="${ART:-rerun_artifacts}"
OUT="${OUT:-rerun_output/seed${SEED}}"
KENLM_BIN="${KENLM_BIN:-$HOME/kenlm/build/bin}"
TRAIN_CONFIG="workflows/recipes/wav2vec2/asr/configs/regspeech12-80hr.yaml"
EVAL_CONFIG="workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml"
RELEASED_URL="https://huggingface.co/cykyy/bn-regional-asr-300m/resolve/main/model.pt"
RELEASED_SHA256="6fa100c22acc2b31b230b03ce8d0c0bcb9a4d5d2ed8f32903dd5d1dd819a582a"
ASSETS="$PWD/$ART/fairseq2_assets"

mkdir -p "$ART" "$ASSETS"
export FAIRSEQ2_USER_ASSET_DIR="$ASSETS"

if [[ ! -f "$TRAIN_CONFIG" ]]; then
    echo "Run this from the root of the omnilingual-asr repo (branch rayth1)." >&2
    exit 1
fi

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
note() { echo "$(stamp) $*" | tee -a "$ART/timeline.txt"; }

seed_overrides() {
    if [[ "$SEED" != "2" ]]; then
        echo "--config common.seed=$SEED --config dataset.asr_task_config.seed=$SEED"
    fi
}

run_dir() {
    # fairseq2 names the run folder ws_<number of GPUs>.<hash>
    local d
    d=$(ls -d "$OUT"/ws_*.* 2>/dev/null | head -1)
    if [[ -z "$d" ]]; then
        echo "No training run found under $OUT" >&2
        exit 1
    fi
    echo "$d"
}

step_env() {
    {
        echo "== date"; stamp
        echo "== git"; git rev-parse HEAD; git status --short
        echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
        nvidia-smi
        echo "== CPU / RAM / disk"; nproc; free -g || true; df -h .
        echo "== Python"; python --version
        echo "== packages"; pip freeze
    } > "$ART/env.txt" 2>&1
    grep -A1 "^name" "$ART/env.txt" | head -2
}

step_setup() {
    if command -v apt-get >/dev/null && [[ "$(id -u)" == "0" ]]; then
        apt-get update -qq
        apt-get install -y -qq libsndfile1 tmux build-essential cmake \
            libboost-system-dev libboost-thread-dev libboost-program-options-dev \
            libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
    else
        echo "Not root or no apt-get: install libsndfile1 (and cmake + boost for KenLM) yourself."
    fi
    pip install -e .
    pip install kagglehub pandas openpyxl soundfile tqdm tensorboard pyctcdecode \
        https://github.com/kpu/kenlm/archive/master.zip
    # fairseq2 0.6 pins torch 2.8 and NumPy 1.x. Without these pins pip can pull a newer
    # torchaudio (crashes on import) and a pyarrow that needs NumPy 2 (seen in testing).
    pip install "torchaudio==2.8.0" "pyarrow>=20,<21"
    if [[ ! -x "$KENLM_BIN/lmplz" ]]; then
        rm -rf "$HOME/kenlm"
        git clone -q https://github.com/kpu/kenlm.git "$HOME/kenlm"
        mkdir -p "$HOME/kenlm/build"
        (cd "$HOME/kenlm/build" && cmake .. >/dev/null && make -j"$(nproc)" lmplz build_binary)
    fi
    python -c "import fairseq2, torch, torchaudio, pyarrow, pyctcdecode, kenlm; from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline; print('fairseq2', fairseq2.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    "$KENLM_BIN/lmplz" --help >/dev/null 2>&1 && echo "KenLM binaries OK"
}

step_data() {
    # Needs Kaggle credentials: ~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY.
    python prepare_regspeech12.py
    wc -l dataset/regspeech12/data_manifest/*.tsv
}

eval_into() {
    # eval_into <output folder> <extra --config arguments...>
    local dest="$1"; shift
    python -m workflows.recipes.wav2vec2.asr.eval "$@" --config-file "$EVAL_CONFIG" "$dest"
    cat "$dest"/ws_*/metrics/eval.jsonl
}

step_eval_baseline() {
    # Zero-shot base model. Thesis: WER 97.93%, UER 81.74%.
    eval_into "$ART/eval_baseline" --config model.name=omniASR_CTC_300M_v2 --config model.path=null
}

step_eval_released() {
    # The model published on Hugging Face, on the full test split. Thesis: WER 72.97%, UER 35.44%.
    local pt="$PWD/$ART/released/model.pt"
    mkdir -p "$(dirname "$pt")"
    [[ -f "$pt" ]] || curl -sSL -o "$pt" "$RELEASED_URL"
    echo "$RELEASED_SHA256  $pt" | sha256sum -c -
    cat > "$ASSETS/released_300m.yaml" <<EOF
name: released_300m
model_family: wav2vec2_asr
model_arch: 300m_v2
checkpoint: $pt
tokenizer_ref: omniASR_tokenizer_written_v2
EOF
    eval_into "$ART/eval_released" --config model.name=released_300m --config model.path=null
}

step_train() {
    note "train start (seed $SEED)"
    # shellcheck disable=SC2046
    python -m workflows.recipes.wav2vec2.asr --config-file "$TRAIN_CONFIG" $(seed_overrides) "$OUT" \
        2>&1 | tee "$ART/train_seed${SEED}.log"
    note "train end (seed $SEED)"
}

step_eval_new() {
    local run ckpt
    run=$(run_dir)
    ckpt="$run/checkpoints/step_10000/model"
    eval_into "$ART/eval_seed${SEED}_step10000" --config model.path="$ckpt" \
        --config model.family=wav2vec2_asr --config model.arch=300m_v2
    sha256sum "$ckpt"/pp_00/tp_00/sdp_00.pt | tee "$ART/checkpoint_seed${SEED}.sha256"
}

step_epochs() {
    python scripts/calculate_epochs.py --output-dir "$(run_dir)" | tee "$ART/epochs_seed${SEED}.txt"
}

step_lm() {
    # n-gram LM decoding of the released model (scripts/lm/README.md).
    python scripts/lm/dump_logits.py --model-card released_300m --splits valid test \
        --out-dir "$ART/lm/logits"
    for mode in raw nopunct; do
        python scripts/lm/build_lm.py --text dataset/regspeech12/data_manifest/train.wrd \
            --order 4 --mode "$mode" --out-dir "$ART/lm/lm_${mode}_4" --kenlm-bin "$KENLM_BIN"
    done
    python scripts/lm/decode_lm.py --logits-dir "$ART/lm/logits" \
        --lm "$ART/lm/lm_raw_4/lm.4gram.bin" "$ART/lm/lm_nopunct_4/lm.4gram.bin" \
        --out-dir "$ART/lm/results"
}

step_pack() {
    local run
    run=$(run_dir)
    mkdir -p "$ART/run_seed${SEED}"
    cp -r "$run/config.yaml" "$run/logs" "$run/tb" "$ART/run_seed${SEED}/" 2>/dev/null || true
    local tarball
    tarball="rerun_seed${SEED}_$(date -u +%Y%m%d).tar.gz"
    # Large files stay out: checkpoints, the downloaded model, LM binaries and saved scores.
    tar --exclude='*.pt' --exclude='*.bin' --exclude='*.arpa' --exclude='*.npz' \
        --exclude='corpus.txt' -czf "$tarball" "$ART"
    ls -lh "$tarball"
    echo "Copy it to your computer, e.g.: scp <user>@<host>:$PWD/$tarball ."
}

ALL=(env setup data eval_baseline eval_released train eval_new epochs lm pack)
STEPS=("$@")
[[ ${#STEPS[@]} -eq 0 ]] && STEPS=("${ALL[@]}")

for s in "${STEPS[@]}"; do
    if ! declare -F "step_$s" >/dev/null; then
        echo "Unknown step '$s'. Steps: ${ALL[*]}" >&2
        exit 1
    fi
    note "step $s: start"
    "step_$s" 2>&1 | tee "$ART/$s.log"
    note "step $s: done"
done
