from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import torch

# 1. Initialize Pipeline
print("Loading Pipeline...")
pipeline = ASRInferencePipeline(
    model_card="finetuned2",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# 2. Define Audio
audio_files = ["/root/omnilingual-asr/dataset/dry_run/wavs/train_barishal_0001.wav"]

# 3. Transcribe (no lang for CTC models)
print(f"Transcribing {audio_files[0]}...")
transcriptions = pipeline.transcribe(audio_files, batch_size=1)

print("\nRESULT:", transcriptions[0])
