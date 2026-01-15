import os
import soundfile as sf
import csv

# --- CONFIGURATION ---
AUDIO_ROOT = "dataset/dry_run/wavs"  # Absolute path to your wavs
OUTPUT_DIR = "dataset/dry_run/data_manifest"  # Where to save the tsv/wrd files
SPLIT_NAME = "train"  # Will create train.tsv and train.wrd
# ---------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
tsv_path = os.path.join(OUTPUT_DIR, f"{SPLIT_NAME}.tsv")
wrd_path = os.path.join(OUTPUT_DIR, f"{SPLIT_NAME}.wrd")

print(f"Scanning {AUDIO_ROOT}...")

tsv_rows = []
wrd_rows = []

# 1. Collect Data
for root, dirs, files in os.walk(AUDIO_ROOT):
    for filename in sorted(files):
        if filename.endswith(".wav"):
            abs_path = os.path.join(root, filename)

            # Get Relative Path (for TSV)
            rel_path = os.path.relpath(abs_path, AUDIO_ROOT)

            # Get Frames
            try:
                frames = sf.info(abs_path).frames
            except:
                print(f"Error reading {filename}, skipping.")
                continue

            # Get Text (Assuming .txt exists, or use placeholder)
            txt_path = abs_path.replace(".wav", ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    text = f.read().strip()
            else:
                text = "placeholder text"

            tsv_rows.append((rel_path, frames))
            wrd_rows.append(text)

# 2. Write TSV (With Header)
with open(tsv_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    # Rule: First line is the Root Directory
    f.write(f"{AUDIO_ROOT}\n")
    # Subsequent lines: relative_path <tab> length
    writer.writerows(tsv_rows)

# 3. Write WRD (Text Only)
with open(wrd_path, "w") as f:
    for text in wrd_rows:
        f.write(f"{text}\n")

print(f"Done!")
print(f"TSV: {tsv_path} (Header: {AUDIO_ROOT})")
print(f"WRD: {wrd_path} ({len(wrd_rows)} lines)")