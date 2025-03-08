# AI Music Generator with Song Extension

This application uses AI to generate short music clips and intelligently extend them into structured songs with proper musical form.

## Features

- Generate short music clips (20 seconds by default) using MusicGen
- Extend short clips into full-length songs (3 minutes by default)
- Upload your own audio clips for extension
- Generate new content based on the original style for more varied songs
- Apply genre-specific song structures (pop, rock, electronic, jazz, etc.)
- Create smooth transitions between song sections
- Add subtle variations to avoid repetitiveness
- Support for different musical genres and structures
- Dual-mode operation for different hardware configurations

## Hardware Modes

The application can run in two different modes to accommodate various hardware configurations:

### Local Mode (Low Resource)
- Optimized for systems with limited GPU VRAM (6GB or less)
- Uses smaller models and memory-efficient processing
- Processes audio in smaller chunks to avoid out-of-memory errors
- Applies simpler variations and shorter crossfades
- Uses lower bitrate for output files

### High Performance Mode
- Designed for systems with high GPU VRAM (10GB+)
- Uses larger models for better quality generation
- Processes audio in full for optimal quality
- Applies more complex variations and longer crossfades
- Uses higher bitrate for output files

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- FFmpeg installed on your system

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install FFmpeg if not already installed:
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

4. Install Ollama for LLM support:
   - Follow instructions at [ollama.ai](https://ollama.ai)
   - Pull the required model: `ollama pull llama3.1:8b` (for local mode)
   - Pull the required model: `ollama pull llama3.1:70b` (for high-performance mode, if available)

## Usage

### Local Mode (Low Resource)

1. Start the FastAPI server:
   ```
   uvicorn app:app --reload
   ```

2. Access the API at `http://localhost:8000`

### High Performance Mode

1. Navigate to the docker directory:
   ```
   cd docker
   ```

2. Start the FastAPI server:
   ```
   uvicorn app:app --reload
   ```

3. Access the API at `http://localhost:8000`

### Web Interface

The application provides a web interface with two main features:

1. **Generate Music**: Create new music from a text description and optionally extend it
2. **Upload & Extend**: Upload your own audio clips and extend them into full songs

Both options include the ability to generate new content based on the original style, creating more varied and interesting extended songs.

### API Endpoints

- `POST /generate_music/`: Generate music from a text prompt
  - Parameters:
    - `prompt`: Text description of the music to generate
    - `duration`: Duration of the initial clip in seconds (default: 20)
    - `extend_to_full_song`: Whether to extend the clip into a full song (default: false)
    - `target_duration`: Target duration for the extended song in seconds (default: 180)
    - `low_resource_mode`: Whether to use low resource mode (default: true for local mode, false for high-performance mode)
    - `regenerate_content`: Whether to generate new content based on the original style (default: false)

- `POST /extend_uploaded_audio/`: Extend an uploaded audio file
  - Parameters (multipart/form-data):
    - `file`: The audio file to extend (MP3, WAV, OGG, FLAC, or M4A)
    - `target_duration`: Target duration for the extended song in seconds (default: 180)
    - `low_resource_mode`: Whether to use low resource mode (default: true for local mode, false for high-performance mode)
    - `regenerate_content`: Whether to generate new content based on the original style (default: false)
    - `style_description`: Description of the music style (required when regenerate_content is true)

- `GET /download_music/`: Download the most recently generated music clip

- `GET /download_extended_music/`: Download the most recently extended song

## Example

```python
import requests

# Generate a short clip
response = requests.post("http://localhost:8000/generate_music/", json={
    "prompt": "A cheerful pop song with piano and drums"
})
print(response.json())

# Generate and extend to a full song (low resource mode)
response = requests.post("http://localhost:8000/generate_music/", json={
    "prompt": "A cheerful pop song with piano and drums",
    "extend_to_full_song": True,
    "target_duration": 180,
    "low_resource_mode": True
})
print(response.json())

# Generate and extend with new content generation
response = requests.post("http://localhost:8000/generate_music/", json={
    "prompt": "A cheerful pop song with piano and drums",
    "extend_to_full_song": True,
    "target_duration": 180,
    "regenerate_content": True
})
print(response.json())

# Upload and extend an existing audio file
with open("my_audio.mp3", "rb") as f:
    files = {"file": ("my_audio.mp3", f, "audio/mpeg")}
    data = {
        "target_duration": 180,
        "low_resource_mode": True
    }
    response = requests.post("http://localhost:8000/extend_uploaded_audio/", files=files, data=data)
print(response.json())

# Upload and extend with new content generation
with open("my_audio.mp3", "rb") as f:
    files = {"file": ("my_audio.mp3", f, "audio/mpeg")}
    data = {
        "target_duration": 180,
        "regenerate_content": True,
        "style_description": "A jazz song with upbeat tempo featuring piano and saxophone"
    }
    response = requests.post("http://localhost:8000/extend_uploaded_audio/", files=files, data=data)
print(response.json())
```

## How It Works

1. **Generating Music**:
   - The application uses MusicGen to generate a short music clip based on the text prompt
   - If song extension is requested, the clip is processed by the song extender

2. **Extending Audio**:
   - The clip is analyzed to find optimal loop points using autocorrelation
   - The clip is divided into sections (intro, verse, chorus, etc.)
   - Sections are arranged according to the genre-specific song structure
   - Variations are applied to repeated sections to avoid monotony
   - Smooth crossfades are created between sections
   - The song is extended until it reaches the target duration
   - An outro section is added to complete the song

3. **Generating New Content**:
   - When regeneration is enabled, the system creates new musical content based on the original style
   - For generated music, the style is extracted from the original prompt
   - For uploaded files, a style description must be provided
   - New verse, chorus, and bridge sections are generated and integrated into the song structure
   - This creates a more varied and interesting extended song with fresh musical ideas

4. **Uploading Audio**:
   - Users can upload their own audio clips in various formats
   - The uploaded audio is processed by the song extender in the same way as generated clips
   - The extended version is made available for download

### Low Resource Mode vs. High Performance Mode

In low resource mode:
- Audio is processed in smaller chunks to reduce memory usage
- Simpler algorithms are used for finding loop points
- Only one variation effect is applied at a time
- Shorter crossfades are used between sections
- Lower bitrate is used for output files
- Smaller models are used for content generation

In high performance mode:
- Audio is processed in full for optimal quality
- Advanced autocorrelation is used for finding optimal loop points
- Multiple variation effects can be applied simultaneously
- Longer crossfades are used for smoother transitions
- Higher bitrate is used for output files
- Larger models are used for content generation

## License

MIT 