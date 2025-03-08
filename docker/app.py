import json
import re
import requests
import torch
import torchaudio
import subprocess
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from audiocraft.models import MusicGen
from pathlib import Path
from typing import Optional
from pydub import AudioSegment

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# Import the song extender
from song_extender import extend_audio_file

# Load genre knowledge database
with open("genres.json", "r") as file:
    GENRES_DB = json.load(file)["Genres"]

app = FastAPI(title="AI Music Generator - High Performance Mode")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

OLLAMA_MODEL = "llama3.1:70b"

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Optimize CUDA performance
torch.backends.cudnn.benchmark = True  
torch.backends.cuda.matmul.allow_tf32 = True  

# Load MusicGen model (optimized for GPU use)
model = MusicGen.get_pretrained("facebook/musicgen-large", device=device)

# Enable mixed-precision for lower VRAM usage
if device == "cuda":
    torch.set_default_dtype(torch.float16)
    torch.cuda.empty_cache()  

# Default MusicGen parameters
model.set_generation_params(duration=20)

class MusicRequest(BaseModel):
    prompt: str
    duration: int = 20  # Default 20s if not specified
    extend_to_full_song: bool = False  # New parameter to control song extension
    target_duration: Optional[int] = 180  # Target duration for extended song (3 minutes default)
    low_resource_mode: bool = False  # Default to False for high-performance environments
    regenerate_content: bool = False  # New parameter to control content regeneration

def get_music_params(prompt: str):
    """Use LLaMA to generate structured music parameters."""
    system_prompt = f"""
    You are an advanced AI music composer that converts user descriptions into structured music parameters. 
    Reference the provided genre knowledge database to extract **accurate and creative** music characteristics.

    - If a genre is **explicitly mentioned**, extract attributes from the database.
    - If the genre is **unknown**, infer details based on the mood and style keywords.
    - If multiple genres are combined, **blend their characteristics** intelligently.    

    For each request, output:
    - **Mood**
    - **Genre**
    - **Tempo**
    - **Time Signature**
    - **Instrumentation**
    - **Energy Level**
    - **Effects & Processing**

    **User Request:**
    "{prompt}"

    If a user **invents a genre**, create a **fusion of two closest genres**.

    ### Genre Knowledge Database ###
{json.dumps(GENRES_DB, indent=2)}

    **STRICT OUTPUT FORMAT:**
    - Your response MUST be **valid JSON inside <JSON>...</JSON> tags**.
    - DO NOT add explanations or extra text.
    """

    payload = {"model": OLLAMA_MODEL, "prompt": system_prompt, "stream": False}
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    data = response.json()

    # Extract JSON between <JSON>...</JSON>
    json_pattern = re.compile(r"<JSON>(.*?)</JSON>", re.DOTALL)
    match = json_pattern.search(data.get('response', '').strip())

    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return {"error": "Malformed JSON from AI"}

    return {"error": "Invalid JSON output from AI"}

def get_unique_filename(base_name, extension):
    """Generate a unique filename if a file with the same name exists."""
    counter = 1
    new_name = f"{base_name}{extension}"
    
    while Path(new_name).exists():
        new_name = f"{base_name}_{counter}{extension}"
        counter += 1
    
    return new_name

def convert_wav_to_mp3(wav_file):
    """Convert WAV to MP3 using ffmpeg."""
    mp3_file = get_unique_filename("generated_music", ".mp3")
    command = ["ffmpeg", "-y", "-i", wav_file, "-b:a", "192K", mp3_file]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return mp3_file

def generate_multitrack_music(prompt, duration, extend_to_full_song=False, target_duration=180, regenerate_content=False):
    """Generate separate instrument layers and mix them."""
    torch.cuda.empty_cache()  

    music_params = get_music_params(prompt)
    if "error" in music_params:
        return music_params  # Return error if JSON is invalid

    instrument_files = []

    for instrument in music_params["Instruments"]:
        structured_prompt = f"A standalone {instrument} track in the style of {music_params['Genre']}, featuring {', '.join(music_params['Effects & Processing'])}."

        with torch.no_grad(), torch.cuda.amp.autocast():
            output_waveform = model.generate([structured_prompt])

        torch.cuda.empty_cache()  

        # Move to CPU
        output_waveform = output_waveform[0].cpu()

        # Save as a unique WAV file
        wav_file = get_unique_filename(f"generated_{instrument.lower()}", ".wav")
        torchaudio.save(wav_file, output_waveform.to(torch.float32), 32000)
        instrument_files.append(wav_file)

    # Merge all instrument layers into a final mix
    final_mix = "generated_music.wav"
    sox_mix_command = ["sox"] + instrument_files + [final_mix]
    subprocess.run(sox_mix_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Convert to MP3
    mp3_file = convert_wav_to_mp3(final_mix)
    
    # Extend the song if requested
    if extend_to_full_song:
        try:
            # Check the duration of the generated clip
            audio_info = AudioSegment.from_file(mp3_file)
            clip_duration_sec = len(audio_info) / 1000
            
            # If clip is very short, warn the user
            if clip_duration_sec < 5:
                return {
                    "message": f"Generated clip is very short ({clip_duration_sec:.1f}s). Extension may not work well. Try generating again.",
                    "file": mp3_file,
                    "parameters": music_params
                }
            
            # Extract genre from music parameters
            extracted_genre = None
            # Use the genre from music parameters
            if "Genre" in music_params:
                extracted_genre = music_params["Genre"].lower()
                # If it's a compound genre, take the first part
                if " " in extracted_genre:
                    extracted_genre = extracted_genre.split()[0]
            
            # For regeneration, we need the style description
            style_description = None
            if regenerate_content:
                # Create a style description from the music parameters
                style_description = f"A {music_params['Genre']} song with {music_params['Mood']} mood"
                if "Tempo" in music_params:
                    style_description += f" at {music_params['Tempo']} BPM"
                if "Instrumentation" in music_params:
                    instruments = ", ".join(music_params["Instrumentation"][:3])  # Limit to 3 instruments
                    style_description += f" featuring {instruments}"
                
            extended_mp3_file = extend_audio_file(
                input_file=mp3_file,
                target_duration=target_duration,
                genre=extracted_genre,
                variation_amount=0.3,
                transition_smoothness=0.5,
                low_resource_mode=False,  # Use full capabilities on high-performance hardware
                regenerate_content=regenerate_content,
                style_description=style_description
            )
            
            # Add information about regeneration to the message
            regeneration_info = " with new generated content" if regenerate_content else ""
            
            return {
                "message": f"Multi-track extended music generated successfully{regeneration_info}",
                "file": extended_mp3_file,
                "original_file": mp3_file,
                "parameters": music_params
            }
        except Exception as e:
            error_message = str(e)
            # Provide more helpful error messages for common issues
            if "Crossfade is longer than" in error_message:
                error_message = "Generated clip has sections that are too short for smooth transitions. Try generating a longer initial clip."
            elif "too short" in error_message.lower():
                error_message = "Generated clip is too short to extend properly. Try generating again or using a different prompt."
                
            return {
                "message": f"Multi-track music generated but extension failed: {error_message}",
                "file": mp3_file,
                "parameters": music_params
            }
    
    return {
        "message": "Multi-track music generated successfully",
        "file": mp3_file,
        "parameters": music_params
    }

@app.post("/generate_music/")
async def generate_music(request: MusicRequest):
    return generate_multitrack_music(
        request.prompt, 
        request.duration,
        extend_to_full_song=request.extend_to_full_song,
        target_duration=request.target_duration,
        regenerate_content=request.regenerate_content
    )

@app.get("/download_music/")
async def download_music():
    """Serve the most recent MP3 file."""
    files = sorted(Path(".").glob("generated_music*.mp3"), key=lambda f: f.stat().st_mtime, reverse=True)
    if files:
        return FileResponse(files[0], media_type="audio/mpeg", filename=files[0].name)
    return {"error": "No generated music file found."}

@app.get("/download_extended_music/")
async def download_extended_music():
    """Serve the most recent extended MP3 file."""
    files = sorted(Path(".").glob("extended_song*.mp3"), key=lambda f: f.stat().st_mtime, reverse=True)
    if files:
        return FileResponse(files[0], media_type="audio/mpeg", filename=files[0].name)
    return {"error": "No extended music file found."}

@app.get("/")
async def serve_index():
    """Serve the HTML interface."""
    return FileResponse("static/index.html")

@app.post("/extend_uploaded_audio/")
async def extend_uploaded_audio(
    file: UploadFile = File(...),
    target_duration: int = Form(180),
    low_resource_mode: bool = Form(False),
    regenerate_content: bool = Form(False),
    style_description: Optional[str] = Form(None)
):
    """Extend an uploaded audio file into a full song."""
    # Validate file type
    if not file.filename.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload MP3, WAV, OGG, FLAC, or M4A files.")
    
    # Validate style description for regeneration
    if regenerate_content and not style_description:
        raise HTTPException(status_code=400, detail="Style description is required when regeneration is enabled.")
    
    # Save the uploaded file
    temp_file_path = get_unique_filename("uploaded_audio", Path(file.filename).suffix)
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Check the duration of the uploaded clip
        audio_info = AudioSegment.from_file(temp_file_path)
        clip_duration_sec = len(audio_info) / 1000
        
        # If clip is very short, warn the user
        if clip_duration_sec < 5:
            return {
                "message": f"Uploaded clip is very short ({clip_duration_sec:.1f}s). Extension may not work well.",
                "file": temp_file_path
            }
        
        # Convert to MP3 if not already
        if not temp_file_path.lower().endswith('.mp3'):
            mp3_file = convert_wav_to_mp3(temp_file_path)
        else:
            mp3_file = temp_file_path
        
        # For uploaded files, we don't have a prompt to extract from
        # We'll use a default genre structure
        genre = "pop"  # Default to a versatile genre structure
        
        # Extend the audio file
        extended_mp3_file = extend_audio_file(
            input_file=mp3_file,
            target_duration=target_duration,
            genre=genre,
            variation_amount=0.3,
            transition_smoothness=0.5,
            low_resource_mode=low_resource_mode,
            regenerate_content=regenerate_content,
            style_description=style_description
        )
        
        # Add information about regeneration to the message
        regeneration_info = " with new generated content" if regenerate_content else ""
        
        return {
            "message": f"Uploaded audio extended successfully{regeneration_info}",
            "file": extended_mp3_file,
            "original_file": mp3_file
        }
    except Exception as e:
        error_message = str(e)
        # Provide more helpful error messages for common issues
        if "Crossfade is longer than" in error_message:
            error_message = "Uploaded clip has sections that are too short for smooth transitions."
        elif "too short" in error_message.lower():
            error_message = "Uploaded clip is too short to extend properly."
        
        return {
            "message": f"Failed to extend uploaded audio: {error_message}",
            "file": temp_file_path
        }
