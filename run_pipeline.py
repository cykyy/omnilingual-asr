"""Transcribe Bengali regional speech with the fine-tuned bn-regional-asr-300m model.

Usage:
    python run_pipeline.py audio1.wav [audio2.wav ...]
    python run_pipeline.py --model omniASR_CTC_300M_v2 audio.wav   # zero-shot base model

Audio: 16 kHz mono WAV, up to 40 seconds. CTC models take no language argument.
"""
import argparse

import torch

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("audio", nargs="+", help="Path(s) to 16 kHz mono WAV files")
parser.add_argument("--model", default="bn_regional_CTC_300M_v2", help="Model card name")
parser.add_argument("--batch-size", type=int, default=1)
args = parser.parse_args()

print(f"Loading {args.model}...")
pipeline = ASRInferencePipeline(
    model_card=args.model,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

transcriptions = pipeline.transcribe(args.audio, batch_size=args.batch_size)
for path, text in zip(args.audio, transcriptions):
    print(f"{path}\t{text}")
