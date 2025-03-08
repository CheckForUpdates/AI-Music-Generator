import json
import re
import requests
import torchaudio
from fastapi import FastAPI
from pydantic import BaseModel
from audiocraft.models import MusicGen
from pathlib import Path

app = FastAPI()
OLLAMA_MODEL = "llama3.2"

# Load AudioCraft model once at startup (small model to save memory)
model = MusicGen.get_pretrained('facebook/musicgen-small')

class MusicRequest(BaseModel):
    prompt: str

def get_music_params(prompt: str):
    """Use LLaMA to generate structured music parameters."""
    system_prompt = f"""
    You are an AI music composer. Convert user descriptions into structured JSON music parameters.

    **Rules:**
    - Your response MUST be in **valid JSON format**.
    - Example:

    <JSON>
    {{
      "Mood": "Energetic, uplifting",
      "Genre": "Jazz",
      "Tempo": 120,
      "Time Signature": ["4/4"],
      "Instruments": ["Piano", "Saxophone", "Bass"],
      "Energy Level": 7,
      "Effects & Processing": ["Reverb", "Swing Feel"]
    }}
    </JSON>

    User request:
    "{prompt}"

    Respond only in JSON format inside `<JSON>...</JSON>`.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": system_prompt,
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)
    data = response.json()
    raw_output = data.get('response', '').strip()

    # Extract JSON
    JSON_PATTERN = re.compile(r"<JSON>(.*?)</JSON>", re.DOTALL)
    match = JSON_PATTERN.search(raw_output)
    if not match:
        return {"error": "Invalid JSON output from AI"}

    return json.loads(match.group(1))

def generate_audiocraft_music(prompt):
    """Use AudioCraft to generate AI music and return a file path."""
    music_params = get_music_params(prompt)
    
    if "error" in music_params:
        return music_params  # Return error message if JSON is invalid

    # Generate AI music from description
    output_waveform = model.generate([prompt], progress=True)

    # Save to WAV file
    output_file = "generated_music.wav"
    torchaudio.save(output_file, output_waveform[0].cpu(), 32000)
    
    return {"message": "Music generated successfully", "file": output_file}

@app.post("/generate_music/")
async def generate_music(request: MusicRequest):
    return generate_audiocraft_music(request.prompt)
