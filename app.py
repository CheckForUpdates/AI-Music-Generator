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

app = FastAPI(title="AI Music Generator - Local Mode")

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

OLLAMA_MODEL = "llama3.1:8b"

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Reduce memory fragmentation risk
torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for better performance

# Load MusicGen model (optimized for GPU use)
model = MusicGen.get_pretrained("facebook/musicgen-small")

# Set device manually for tensor operations
if device == "cuda":
    torch.set_default_dtype(torch.float16)  # Lower memory usage with fp16
    torch.cuda.empty_cache()  # Free any cached memory

# Model generation parameters
model.set_generation_params(duration=30)

class MusicRequest(BaseModel):
    prompt: str
    duration: int = 20  # Default 20s if not specified
    extend_to_full_song: bool = False  # New parameter to control song extension
    target_duration: Optional[int] = 180  # Target duration for extended song (3 minutes default)
    low_resource_mode: bool = True  # Default to True for local machine
    regenerate_content: bool = False  # New parameter to control content regeneration

def get_music_params(prompt: str):
    """Use LLaMA to generate structured music parameters with the genre database."""
    system_prompt = f"""
    You are an advanced AI music composer that converts user descriptions into structured music parameters. 
    Reference the provided genre knowledge database to extract **accurate and creative** music characteristics.

    - If a genre is **explicitly mentioned**, extract attributes from the database.
    - If the genre is **unknown**, infer details based on the mood and style keywords.
    - If multiple genres are combined, **blend their characteristics** intelligently.    

    For each request, output:
    - **Mood:** (A short descriptor summarizing the emotion.)
    - **Genre:** (Primary genre and potential blends.)
    - **Tempo:** (BPM, between 40-200, based on the genre database.)
    - **Time Signature:** (4/4, 6/8, etc., based on the style.)
    - **Instrumentation:** (List of instruments commonly used in the selected genre.)
    - **Energy Level:** (Scale from 1-10, where 1 is calm, and 10 is chaotic.)
    - **Effects & Processing:** (Special effects such as reverb, distortion, synth arpeggiation.)

    ONLY return JSON. Do NOT include explanations or conversational responses.
    - The JSON MUST be wrapped inside <JSON>...</JSON> tags.
    - If the user asks a question, IGNORE IT and return JSON.

    **User Request:**
    "{prompt}"    

    If a user **invents a genre**, create a **fusion of two closest genres**.

    ### Genre Knowledge Database ###
{json.dumps(GENRES_DB, indent=2)}
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": system_prompt,
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)
    data = response.json()

    # Log the full raw AI response to a file
    with open("ai_raw_output.log", "w") as log_file:
        log_file.write(f"Raw AI Response:\n{data}\n")

    raw_output = data.get('response', '').strip()

    # Extract JSON between <JSON>...</JSON>
    json_pattern = re.compile(r"<JSON>(.*?)</JSON>", re.DOTALL)
    match = json_pattern.search(raw_output)

    if match:
        json_data = match.group(1).strip()

        # Attempt to parse JSON
        try:
            return json.loads(json_data)
        except json.JSONDecodeError:
            print("Warning: AI response contained malformed JSON. Attempting fix...")
            json_data = json_data.replace("\n", "").replace("\t", "").strip()
            json_data = re.sub(r",\s*}", "}", json_data)
            json_data = re.sub(r",\s*]", "]", json_data)

            try:
                return json.loads(json_data)
            except json.JSONDecodeError:
                return {"error": "AI response contained malformed JSON after auto-fix"}

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
    """Convert WAV to MP3 using ffmpeg, ensuring unique filenames."""
    mp3_file = get_unique_filename("generated_music", ".mp3")
    command = ["ffmpeg", "-y", "-i", wav_file, "-b:a", "192K", mp3_file]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return mp3_file

def generate_audiocraft_music(prompt, extend_to_full_song=False, target_duration=180, low_resource_mode=True, regenerate_content=False):
    """Use MusicGen to generate AI music and return an MP3 file."""
    torch.cuda.empty_cache()  # Free memory before generation

    structured_prompt = f"Generated music: {prompt}"

    with torch.no_grad(), torch.cuda.amp.autocast():  # Use mixed precision for efficiency
        output_waveform = model.generate([structured_prompt])

    torch.cuda.empty_cache()  # Free memory after generation

    # Move output to CPU immediately to avoid OOM crashes
    output_waveform = output_waveform[0].cpu()

    # Ensure unique file names
    wav_file = get_unique_filename("generated_music", ".wav")
    
    torchaudio.save(wav_file, output_waveform.to(torch.float32), 32000)  # Save in CPU memory

    # Convert to MP3 with unique naming
    mp3_file = convert_wav_to_mp3(wav_file)
    
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
                    "file": mp3_file
                }
            
            # Extract genre from prompt if not explicitly provided
            # This is now the primary way to determine genre since we removed the UI selection
            extracted_genre = None
            # Try to extract genre from music parameters
            try:
                music_params = get_music_params(prompt)
                if "Genre" in music_params:
                    extracted_genre = music_params["Genre"].lower()
                    # If it's a compound genre, take the first part
                    if " " in extracted_genre:
                        extracted_genre = extracted_genre.split()[0]
            except Exception:
                # If extraction fails, leave as None
                pass
                
            # For regeneration, we need the style description
            style_description = prompt if regenerate_content else None
                
            extended_mp3_file = extend_audio_file(
                input_file=mp3_file,
                target_duration=target_duration,
                genre=extracted_genre,
                variation_amount=0.3,
                transition_smoothness=0.5,
                low_resource_mode=low_resource_mode,
                regenerate_content=regenerate_content,
                style_description=style_description
            )
            
            # Add information about regeneration to the message
            regeneration_info = " with new generated content" if regenerate_content else ""
            
            return {
                "message": f"Extended music generated successfully{regeneration_info}",
                "file": extended_mp3_file,
                "original_file": mp3_file
            }
        except Exception as e:
            error_message = str(e)
            # Provide more helpful error messages for common issues
            if "Crossfade is longer than" in error_message:
                error_message = "Generated clip has sections that are too short for smooth transitions. Try generating a longer initial clip."
            elif "too short" in error_message.lower():
                error_message = "Generated clip is too short to extend properly. Try generating again or using a different prompt."
            
            return {
                "message": f"Music generated but extension failed: {error_message}",
                "file": mp3_file
            }
    
    return {
        "message": "Music generated successfully",
        "file": mp3_file,
    }

@app.post("/generate_music/")
async def generate_music(request: MusicRequest):
    return generate_audiocraft_music(
        request.prompt, 
        extend_to_full_song=request.extend_to_full_song,
        target_duration=request.target_duration,
        low_resource_mode=request.low_resource_mode,
        regenerate_content=request.regenerate_content
    )

@app.get("/download_music/")
async def download_music():
    """Serve the most recent MP3 file for download."""
    files = sorted(Path(".").glob("generated_music*.mp3"), key=lambda f: f.stat().st_mtime, reverse=True)
    
    if files:
        return FileResponse(files[0], media_type="audio/mpeg", filename=files[0].name)
    
    return {"error": "No generated music file found."}

@app.get("/download_extended_music/")
async def download_extended_music():
    """Serve the most recent extended MP3 file for download."""
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
    low_resource_mode: bool = Form(True),
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
