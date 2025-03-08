import requests
import json

OLLAMA_MODEL = "llama3.1:8b"

def get_music_params(prompt):
    system_prompt = """You are an advanced AI music composer that converts user descriptions into structured music parameters. 
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

If a user **invents a genre**, create a **fusion of two closest genres**.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system_prompt}\nUser: {prompt}\nResponse:",
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)
    data = response.json()

    # Extract AI's response
    text_output = data['response'].strip()

    try:
        return json.loads(text_output)  # Convert text to JSON
    except json.JSONDecodeError:
        print("Error: AI response is not valid JSON.")
        return None

# Example usage
user_prompt = "A dark horror soundtrack with eerie piano and deep bass."
music_params = get_music_params(user_prompt)

print("Generated Music Parameters:", music_params)
