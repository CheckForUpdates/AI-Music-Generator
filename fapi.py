from fastapi import FastAPI
from ollama import generate

app = FastAPI()

@app.post("/generate_music_params")
async def generate_music_params(prompt: str):
    response = generate(model="llama3.1:8b", prompt=f"Extract mood, genre, tempo, and instruments from: {prompt}")
    return response
