from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
from TTS.api import TTS
import torch
import numpy as np
from pydub import AudioSegment
import os

app = FastAPI()

# ==================== CONFIGURABLE SETTINGS ====================
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
REFERENCE_WAV = "/app/reference_voice.wav"  # Path inside container
LANGUAGE = "en"
USE_HALF_PRECISION = True
OUTPUT_FORMAT = "mp3"
MP3_BITRATE = "128K"
# ==============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading XTTS-v2 on {device}")
tts = TTS(MODEL_NAME, progress_bar=True)  # Remove gpu= parameter

# Manually move model to device (new recommended way)
tts.to(device)

class TextInput(BaseModel):
    text: str

@app.post("/generate")
async def generate_audio(input: TextInput):
    try:
        temp_wav = "/tmp/output.wav"
        tts.tts_to_file(
            text=input.text,
            speaker_wav=REFERENCE_WAV,
            language=LANGUAGE,
            file_path=temp_wav,
            speed=0.98,              # 0.9-1.1; adjust if too fast/slow
            temperature=0.65,       # 0.6-0.75: higher = more expressive, lower = more stable
            repetition_penalty=12.0, # Higher (8-12) reduces glitches/repeats
            top_k=50,               # 30-60: reduces randomness
            top_p=0.90              # 0.8-0.95: nucleus sampling
        )
        # Load raw
        audio_segment = AudioSegment.from_wav(temp_wav)

        # === ANTI-CRUSH + DE-ESSING (SAFE VERSION) ===
        # Gentle compression to tame peaks
        audio_segment = audio_segment.compress_dynamic_range(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0
        )

        # Peak limiting for headroom
        audio_segment = audio_segment.apply_gain(-audio_segment.max_dBFS)  # Normalize to 0 dB peak first
        audio_segment = audio_segment - 0.5  # Add 0.5 dB headroom (simple subtract)

        # De-essing: Gentle low-pass to reduce harsh sibilance (S/F/Sh hiss)
        audio_segment = audio_segment.low_pass_filter(8000)  # 8-9 kHz cutoff is sweet spot for speech

        # Optional: Very light high-pass to remove rumble
        audio_segment = audio_segment.high_pass_filter(100) # Remove sub-100Hz rumble

        # Final normalize with safe headroom
        audio_segment = audio_segment.normalize(headroom=1.0)
        output_buffer = BytesIO()
        if OUTPUT_FORMAT.lower() == "mp3":
            audio_segment.export(output_buffer, format="mp3", bitrate=MP3_BITRATE)
            media_type = "audio/mpeg"
        else:
            audio_segment.export(output_buffer, format="wav")
            media_type = "audio/wav"

        output_buffer.seek(0)
        os.remove(temp_wav)  # Clean up temp file

        return StreamingResponse(output_buffer, media_type=media_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))