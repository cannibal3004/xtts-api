# XTTS API

A FastAPI-based text-to-speech (TTS) service powered by [Coqui TTS](https://github.com/coqui-ai/TTS) using the XTTS-v2 multilingual model. This project provides a REST API endpoint for generating high-quality, natural-sounding audio from text with optional audio processing (compression, de-essing, normalization).

## Features

- **Multilingual Support**: XTTS-v2 supports multiple languages
- **Voice Cloning**: Uses a reference WAV file to maintain consistent voice characteristics
- **Audio Processing**: Includes dynamic range compression, de-essing, and normalization for polished output
- **Docker Support**: Containerized with NVIDIA GPU acceleration
- **Multiple Output Formats**: Supports both MP3 and WAV output
- **FastAPI**: Modern, fast web framework with automatic API documentation

## Requirements

### Hardware
- NVIDIA GPU (CUDA compute capability 6.0+) - recommended for real-time inference
- CPU fallback supported but significantly slower

### Software
- Docker & Docker Compose with NVIDIA GPU support
- Python 3.10+ (if running locally without Docker)

## Installation & Setup

### Option 1: Docker Compose (Recommended)

1. **Prepare your reference voice file**:
   - Place a `default.wav` file under `voices/` (i.e., `voices/default.wav`)
   - This WAV file should be a clear sample of the voice you want to clone (typically 5-10 seconds)

2. **Build and run**:
   ```bash
   docker-compose up --build
   ```

   The service will:
   - Build the Docker image (first run may take 10-15 minutes due to model compilation)
   - Download the XTTS-v2 model (~2GB)
   - Start the API server on `http://localhost:8000`

### Option 2: Local Python Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare reference voice**:
   - Place `default.wav` under `voices/` (i.e., `voices/default.wav`)

3. **Run the server**:
   ```bash
   python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```

## Configuration

Edit the following variables in `app.py` to customize behavior:

```python
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"  # TTS model
REFERENCE_WAV = "/app/voices/default.wav"  # Path to reference voice inside container
LANGUAGE = "en"  # Output language code (e.g., "en", "es", "fr", "de")
USE_HALF_PRECISION = True  # Use FP16 for faster inference
OUTPUT_FORMAT = "mp3"  # "mp3" or "wav"
MP3_BITRATE = "128K"  # MP3 bitrate (e.g., "192K", "256K")
```

### Audio Processing Parameters

Adjust these values in the `generate_audio()` function for fine-tuning:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| `speed` | 0.9-1.1 | Playback speed adjustment |
| `temperature` | 0.6-0.75 | Expression level (higher = more expressive) |
| `repetition_penalty` | 8-12 | Reduce glitches and repeating words |
| `top_k` | 30-60 | Reduce output randomness |
| `top_p` | 0.8-0.95 | Nucleus sampling diversity |

## API Usage

### Endpoint: POST `/generate`

Generates audio from input text.

**Request**:
```json
{
   "text": "Hello, this is a test of the text to speech system.",
   "voice": "/app/voices/default.wav"
}
```

`voice` is optional; if omitted, the default voice at `REFERENCE_WAV` is used. Paths must be valid inside the container (for Docker) or local process (for bare-metal runs).

**Response**:
- Content-Type: `audio/mpeg` (if MP3) or `audio/wav` (if WAV)
- Body: Binary audio stream

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world"}' \
  --output output.mp3
```

**Example using Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={"text": "Hello, how are you?"}
)

with open("output.mp3", "wb") as f:
    f.write(response.content)
```

### Interactive API Documentation

Once running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Project Structure

```
.
├── app.py                    # FastAPI application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker image configuration
├── docker-compose.yml       # Docker Compose orchestration
├── voices/                  # Directory for reference voices
│   └── default.wav          # Default voice (provide your own)
└── README.md               # This file
```

## Performance Considerations

- **First inference**: ~10-30 seconds (model loading and compilation)
- **Subsequent requests**: 2-5 seconds depending on text length and GPU
- **GPU Memory**: ~6-8GB VRAM recommended for smooth operation
- **CPU-only**: 30-60+ seconds per request (not recommended for production)

## Troubleshooting

### Build takes too long / runs out of memory
- The `MAX_JOBS=4` setting in the Dockerfile limits parallel compilation
- Reduce to `MAX_JOBS=2` for systems with <16GB RAM
- Increase to `MAX_JOBS=8` for systems with >32GB RAM

### CUDA out of memory errors
- Decrease batch processing if applicable
- Ensure other GPU processes are not running

### No reference voice detected
- Verify `reference_voice.wav` exists in the project root
- Inside Docker, it should be at `/app/reference_voice.wav`
- Ensure it's a valid WAV file

### API not responding
- Check Docker logs: `docker logs xtts-tts-api`
- Verify port 8000 is not in use: `lsof -i :8000`
- Ensure GPU is available: `nvidia-smi`

## Dependencies

Key packages:
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **Coqui TTS**: Text-to-speech engine
- **PyTorch**: Deep learning framework with CUDA support
- **Pydub**: Audio processing and format conversion

See [requirements.txt](requirements.txt) for complete list with versions.

## License

This project uses Coqui TTS which is licensed under the MPL-2.0. Ensure compliance with all dependencies' licenses before commercial use.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Support

For issues related to:
- **XTTS/Coqui TTS**: See [Coqui TTS GitHub](https://github.com/coqui-ai/TTS)
- **FastAPI**: See [FastAPI Documentation](https://fastapi.tiangolo.com/)
- **This project**: Create an issue in the repository
