import os
import json
import torch
import torchaudio
import numpy as np
import subprocess
from pathlib import Path
from pydub import AudioSegment
from scipy.signal import correlate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from audiocraft.models import MusicGen

# Constants for song structure
SONG_STRUCTURES = {
    "pop": ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"],
    "electronic": ["intro", "buildup", "drop", "breakdown", "buildup", "drop", "breakdown", "outro"],
    "rock": ["intro", "verse", "chorus", "verse", "chorus", "solo", "chorus", "outro"],
    "ambient": ["intro", "theme", "variation", "theme", "variation", "climax", "resolution"],
    "orchestral": ["exposition", "development", "recapitulation", "coda"],
    "jazz": ["head", "solo", "solo", "solo", "head", "outro"],
}

# Default structure if genre not found
DEFAULT_STRUCTURE = ["intro", "section_a", "section_b", "section_a", "section_b", "bridge", "section_b", "outro"]

class SongExtensionRequest(BaseModel):
    input_file: str
    target_duration: int = 180  # Default 3 minutes
    structure: Optional[List[str]] = None
    genre: Optional[str] = None
    transition_smoothness: float = 0.5  # 0.0 to 1.0, higher means smoother transitions
    variation_amount: float = 0.3  # 0.0 to 1.0, how much to vary repeated sections
    low_resource_mode: bool = False  # Flag for low-resource mode
    regenerate_content: bool = False  # Flag to regenerate content for extension

class SongExtender:
    def __init__(self, low_resource_mode=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.low_resource_mode = low_resource_mode
        self.model = None  # Will be loaded on demand
        
        # If in low resource mode, set some memory-saving parameters
        if self.low_resource_mode and self.device == "cuda":
            # Set smaller chunk sizes for processing
            self.max_chunk_size_mb = 32
            # Use lower precision
            torch.set_default_dtype(torch.float16)
    
    def load_model(self):
        """Load the MusicGen model on demand."""
        if self.model is None:
            # Free memory before loading
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            # Load a smaller model in low resource mode
            model_size = "facebook/musicgen-small" if self.low_resource_mode else "facebook/musicgen-medium"
            self.model = MusicGen.get_pretrained(model_size)
            
            # Set default generation parameters
            self.model.set_generation_params(duration=5)  # Short segments for extension
    
    def unload_model(self):
        """Unload the model to free memory."""
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def get_unique_filename(self, base_name, extension):
        """Generate a unique filename if a file with the same name exists."""
        counter = 1
        new_name = f"{base_name}{extension}"
        
        while Path(new_name).exists():
            new_name = f"{base_name}_{counter}{extension}"
            counter += 1
        
        return new_name
    
    def load_audio(self, file_path):
        """Load audio file and return waveform and sample rate."""
        if self.low_resource_mode:
            # In low resource mode, load audio directly with pydub to avoid torch memory usage
            # and convert to torch tensor only when needed
            audio_segment = AudioSegment.from_file(file_path)
            sample_rate = audio_segment.frame_rate
            
            # Convert to numpy array first (more memory efficient)
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Handle stereo
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2)).T
            else:
                samples = samples.reshape(1, -1)
                
            # Convert to torch tensor only when needed
            waveform = torch.tensor(samples, dtype=torch.float32 if self.device == "cpu" else torch.float16)
            return waveform, sample_rate
        else:
            # Standard loading with torchaudio
            waveform, sample_rate = torchaudio.load(file_path)
            return waveform, sample_rate
    
    def find_best_loop_points(self, waveform, sample_rate, num_candidates=5):
        """Find optimal loop points in the audio using autocorrelation."""
        # Free memory before intensive computation
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            # In low resource mode, process in smaller chunks
            if self.low_resource_mode:
                # Process in chunks of 5 seconds
                chunk_size = 5 * sample_rate
                total_samples = waveform.shape[1]
                chunks = []
                
                for i in range(0, total_samples, chunk_size):
                    end = min(i + chunk_size, total_samples)
                    chunk = torch.mean(waveform[:, i:end], dim=0).cpu().numpy()
                    chunks.append(chunk)
                
                mono_waveform = np.concatenate(chunks)
                del chunks  # Free memory
            else:
                mono_waveform = torch.mean(waveform, dim=0).cpu().numpy()
        else:
            mono_waveform = waveform[0].cpu().numpy()
            
        # Calculate autocorrelation to find repeating patterns
        # In low resource mode, use a simpler approach
        if self.low_resource_mode:
            # Use a simpler approach for finding loop points
            # Divide the audio into equal segments
            min_segment = int(0.5 * sample_rate)  # Minimum 0.5 seconds
            segment_length = len(mono_waveform) // 4  # Divide into 4 segments
            
            if segment_length < min_segment:
                segment_length = min_segment
                
            # Create loop points at regular intervals
            loop_points = []
            for i in range(1, 4):
                point = i * segment_length
                if point > min_segment and point < len(mono_waveform) - min_segment:
                    loop_points.append(point)
                    
            # Add a few more points based on amplitude changes
            # Find points where the amplitude changes significantly
            amplitude = np.abs(mono_waveform)
            diff = np.abs(np.diff(amplitude))
            
            # Find peaks in the difference
            threshold = np.percentile(diff, 90)  # Top 10% of changes
            potential_points = np.where(diff > threshold)[0]
            
            # Filter points to be at least min_segment apart
            filtered_points = []
            last_point = -min_segment
            
            for point in potential_points:
                if point - last_point >= min_segment and min_segment < point < len(mono_waveform) - min_segment:
                    filtered_points.append(point)
                    last_point = point
                    
                    # Limit to 3 additional points
                    if len(filtered_points) >= 3:
                        break
                        
            loop_points.extend(filtered_points)
            
            # Ensure we have at least one loop point
            if not loop_points:
                loop_points = [len(mono_waveform) // 2]  # Middle point
                
            return loop_points[:num_candidates]
        else:
            # Full autocorrelation approach for high-resource mode
            correlation = correlate(mono_waveform, mono_waveform, mode='full')
            correlation = correlation[len(correlation)//2:]
            
            # Minimum segment length (0.5 seconds)
            min_segment = int(0.5 * sample_rate)
            
            # Find peaks in correlation (potential loop points)
            peaks = []
            for i in range(min_segment, len(correlation) - min_segment):
                if correlation[i] > correlation[i-1] and correlation[i] > correlation[i+1]:
                    peaks.append((i, correlation[i]))
            
            # Sort by correlation strength and return top candidates
            peaks.sort(key=lambda x: x[1], reverse=True)
            return [p[0] for p in peaks[:num_candidates]]
    
    def create_crossfade(self, segment1, segment2, crossfade_duration_ms=1000):
        """Create a smooth crossfade between two audio segments."""
        # In low resource mode, use a simpler crossfade
        if self.low_resource_mode and crossfade_duration_ms > 500:
            # Use a shorter crossfade to reduce memory usage
            crossfade_duration_ms = 500
        
        # Check if either segment is too short for the crossfade
        segment1_duration_ms = len(segment1)
        segment2_duration_ms = len(segment2)
        
        # If either segment is too short, reduce the crossfade duration
        if segment1_duration_ms < crossfade_duration_ms or segment2_duration_ms < crossfade_duration_ms:
            # Use the shortest possible crossfade that works with both segments
            max_possible_crossfade = min(segment1_duration_ms, segment2_duration_ms) - 10  # Leave 10ms buffer
            
            # Ensure we don't go below 10ms
            crossfade_duration_ms = max(10, max_possible_crossfade)
            
            # Log the adjustment if it's significant
            if max_possible_crossfade < 100:  # If we had to reduce to less than 100ms
                print(f"Warning: Reduced crossfade from {crossfade_duration_ms}ms to {max_possible_crossfade}ms due to short segment")
        
        # Perform the crossfade
        try:
            return segment1.append(segment2, crossfade=crossfade_duration_ms)
        except Exception as e:
            # If crossfade still fails, try with no crossfade
            print(f"Crossfade failed: {str(e)}. Falling back to direct append.")
            return segment1 + segment2
    
    def apply_variation(self, segment, variation_amount):
        """Apply subtle variations to an audio segment to avoid repetitiveness."""
        # In low resource mode, use simpler variations
        if self.low_resource_mode:
            # Limit the number of effects applied
            varied = segment
            
            # Choose only one effect to apply
            effect_choice = np.random.randint(0, 3)
            
            if effect_choice == 0 and variation_amount > 0:
                # Apply slight volume change (least memory intensive)
                volume_change = (np.random.random() - 0.5) * 6 * variation_amount
                varied = varied + volume_change
            elif effect_choice == 1 and variation_amount > 0:
                # Apply slight tempo change
                tempo_change = 1.0 + (np.random.random() - 0.5) * 0.1 * variation_amount
                varied = varied.speedup(playback_speed=tempo_change)
            
            return varied
        else:
            # Full variation processing for high-resource mode
            # Convert to numpy array for processing
            samples = np.array(segment.get_array_of_samples())
            
            if segment.channels == 2:
                # Reshape for stereo
                samples = samples.reshape((-1, 2))
                
                # Apply subtle EQ changes (boost or cut different frequency ranges)
                if variation_amount > 0:
                    # Create a new AudioSegment with variations
                    varied = segment
                    
                    # Apply slight pitch shift
                    if np.random.random() < variation_amount:
                        octaves = (np.random.random() - 0.5) * 0.1 * variation_amount
                        new_sample_rate = int(segment.frame_rate * (2.0 ** octaves))
                        varied = varied._spawn(varied.raw_data, overrides={
                            "frame_rate": new_sample_rate
                        })
                        varied = varied.set_frame_rate(segment.frame_rate)
                    
                    # Apply slight tempo change
                    if np.random.random() < variation_amount:
                        tempo_change = 1.0 + (np.random.random() - 0.5) * 0.1 * variation_amount
                        varied = varied.speedup(playback_speed=tempo_change)
                    
                    # Apply slight volume change
                    if np.random.random() < variation_amount:
                        volume_change = (np.random.random() - 0.5) * 6 * variation_amount
                        varied = varied + volume_change
                    
                    return varied
            
            return segment
    
    def generate_new_content(self, style_description, duration_sec=5):
        """Generate new content based on style description."""
        try:
            # Load model if not already loaded
            self.load_model()
            
            # Generate new content
            with torch.no_grad(), torch.cuda.amp.autocast():
                output_waveform = self.model.generate([style_description])
            
            # Move to CPU
            output_waveform = output_waveform[0].cpu()
            
            # Save as WAV
            temp_wav = self.get_unique_filename("temp_generated", ".wav")
            torchaudio.save(temp_wav, output_waveform.to(torch.float32), 32000)
            
            # Convert to AudioSegment
            new_segment = AudioSegment.from_file(temp_wav)
            
            # Clean up temp file
            os.remove(temp_wav)
            
            # Unload model to free memory
            self.unload_model()
            
            return new_segment
        except Exception as e:
            print(f"Error generating new content: {str(e)}")
            # Return empty segment if generation fails
            return AudioSegment.silent(duration=duration_sec * 1000)

    def extend_song(self, request: SongExtensionRequest):
        """Extend a short audio clip into a full song with structure."""
        # Set low resource mode from request
        self.low_resource_mode = request.low_resource_mode
        
        input_file = request.input_file
        target_duration = request.target_duration
        
        # Free memory before starting
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Determine song structure based on genre or use provided structure
        structure = request.structure
        if not structure:
            if request.genre and request.genre.lower() in SONG_STRUCTURES:
                structure = SONG_STRUCTURES[request.genre.lower()]
            else:
                structure = DEFAULT_STRUCTURE
        
        # Load the input audio
        try:
            source_audio = AudioSegment.from_file(input_file)
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {str(e)}")
        
        # Get source audio duration in seconds
        source_duration = len(source_audio) / 1000
        
        # If source is already longer than target, just return it
        if source_duration >= target_duration:
            return input_file
        
        # If source is very short (less than 2 seconds), it's too short to work with
        if source_duration < 2:
            raise ValueError(f"Source audio is too short ({source_duration:.2f}s). Minimum 2 seconds required.")
        
        # Ensure source is long enough to be divided into sections
        min_section_duration_ms = 500  # Minimum 500ms per section
        min_total_duration_ms = min_section_duration_ms * 4  # Need at least 4 sections
        
        if len(source_audio) < min_total_duration_ms:
            # If source is too short, loop it to reach minimum length
            loops_needed = (min_total_duration_ms // len(source_audio)) + 1
            source_audio = source_audio * loops_needed
            print(f"Source audio was too short. Looped {loops_needed} times to reach minimum length.")
        
        # Find optimal loop points
        waveform, sample_rate = self.load_audio(input_file)
        loop_points = self.find_best_loop_points(waveform, sample_rate)
        
        # Free memory after intensive computation
        if self.device == "cuda":
            torch.cuda.empty_cache()
            del waveform
        
        # Create segments for each section of the song
        sections = {}
        
        # Create intro (first 25% of source)
        intro_duration = int(len(source_audio) * 0.25)
        sections["intro"] = source_audio[:intro_duration]
        
        # Create outro (last 25% of source)
        outro_duration = int(len(source_audio) * 0.25)
        sections["outro"] = source_audio[-outro_duration:]
        
        # Create main sections using different parts of the source
        source_segments = []
        segment_count = 4  # Divide source into 4 segments
        segment_length = len(source_audio) // segment_count
        
        for i in range(segment_count):
            start = i * segment_length
            end = start + segment_length
            source_segments.append(source_audio[start:end])
        
        # Map segments to section types
        sections["verse"] = source_segments[0]
        sections["chorus"] = source_segments[1]
        sections["bridge"] = source_segments[2]
        sections["solo"] = source_segments[3]
        
        # For electronic music specific sections
        sections["buildup"] = self.apply_variation(source_segments[0], request.variation_amount * 1.5)
        sections["drop"] = self.apply_variation(source_segments[1], request.variation_amount * 2)
        sections["breakdown"] = self.apply_variation(source_segments[2], request.variation_amount)
        
        # For classical/orchestral music
        sections["exposition"] = source_segments[0]
        sections["development"] = self.apply_variation(source_segments[1], request.variation_amount * 1.2)
        sections["recapitulation"] = self.apply_variation(source_segments[0], request.variation_amount * 0.8)
        sections["coda"] = source_segments[3]
        
        # For jazz
        sections["head"] = source_segments[0]
        sections["solo"] = self.apply_variation(source_segments[1], request.variation_amount * 1.5)
        
        # Generic sections
        sections["section_a"] = source_segments[0]
        sections["section_b"] = source_segments[1]
        sections["theme"] = source_segments[0]
        sections["variation"] = self.apply_variation(source_segments[0], request.variation_amount * 1.2)
        sections["climax"] = self.apply_variation(source_segments[1], request.variation_amount * 2)
        sections["resolution"] = self.apply_variation(source_segments[3], request.variation_amount * 0.5)
        
        # Build the extended song following the structure
        extended_song = AudioSegment.empty()
        
        # Keep track of which sections have been used to apply variations to repeats
        used_sections = set()
        
        # In low resource mode, process in smaller batches
        if self.low_resource_mode:
            # Process structure in chunks to avoid memory issues
            chunk_size = 4  # Process 4 sections at a time
            
            for i in range(0, len(structure), chunk_size):
                chunk_structure = structure[i:i+chunk_size]
                chunk_song = AudioSegment.empty()
                
                for section_name in chunk_structure:
                    if section_name in sections:
                        section_audio = sections[section_name]
                        
                        # Apply variation if this section type has been used before
                        if section_name in used_sections:
                            section_audio = self.apply_variation(section_audio, request.variation_amount)
                        
                        # Add to the chunk with crossfade if not the first section
                        if len(chunk_song) > 0:
                            crossfade_ms = int(1000 * request.transition_smoothness)
                            chunk_song = self.create_crossfade(chunk_song, section_audio, crossfade_ms)
                        else:
                            chunk_song += section_audio
                        
                        used_sections.add(section_name)
                
                # Add chunk to extended song with crossfade
                if len(extended_song) > 0 and len(chunk_song) > 0:
                    crossfade_ms = int(1000 * request.transition_smoothness)
                    extended_song = self.create_crossfade(extended_song, chunk_song, crossfade_ms)
                elif len(chunk_song) > 0:
                    extended_song += chunk_song
                
                # Free memory after each chunk
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        else:
            # Standard processing for high-resource mode
            for section_name in structure:
                if section_name in sections:
                    section_audio = sections[section_name]
                    
                    # Apply variation if this section type has been used before
                    if section_name in used_sections:
                        section_audio = self.apply_variation(section_audio, request.variation_amount)
                    
                    # Add to the extended song with crossfade if not the first section
                    if len(extended_song) > 0:
                        crossfade_ms = int(1000 * request.transition_smoothness)
                        extended_song = self.create_crossfade(extended_song, section_audio, crossfade_ms)
                    else:
                        extended_song += section_audio
                    
                    used_sections.add(section_name)
        
        # Check if we've reached the target duration
        current_duration = len(extended_song) / 1000  # in seconds
        
        # If we need more duration, repeat the structure
        while current_duration < target_duration:
            # In low resource mode, process repeats in smaller batches
            if self.low_resource_mode:
                # Take a subset of the structure to repeat (avoid intro/outro)
                repeat_structure = structure[1:-1]
                chunk_size = min(3, len(repeat_structure))  # Smaller chunks for repeats
                
                for i in range(0, len(repeat_structure), chunk_size):
                    if current_duration >= target_duration:
                        break
                        
                    chunk_structure = repeat_structure[i:i+chunk_size]
                    chunk_song = AudioSegment.empty()
                    
                    for section_name in chunk_structure:
                        if current_duration >= target_duration:
                            break
                            
                        if section_name in sections:
                            section_audio = sections[section_name]
                            # Always apply variation for repeated structures
                            section_audio = self.apply_variation(section_audio, request.variation_amount)
                            
                            # Add with crossfade
                            if len(chunk_song) > 0:
                                crossfade_ms = int(1000 * request.transition_smoothness)
                                chunk_song = self.create_crossfade(chunk_song, section_audio, crossfade_ms)
                            else:
                                chunk_song += section_audio
                    
                    # Add chunk to extended song with crossfade
                    if len(chunk_song) > 0:
                        crossfade_ms = int(1000 * request.transition_smoothness)
                        extended_song = self.create_crossfade(extended_song, chunk_song, crossfade_ms)
                        current_duration = len(extended_song) / 1000
                    
                    # Free memory after each chunk
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
            else:
                # Standard processing for high-resource mode
                for section_name in structure[1:-1]:  # Skip intro and outro for repeats
                    if current_duration >= target_duration:
                        break
                        
                    if section_name in sections:
                        section_audio = sections[section_name]
                        # Always apply variation for repeated structures
                        section_audio = self.apply_variation(section_audio, request.variation_amount)
                        
                        # Add with crossfade
                        crossfade_ms = int(1000 * request.transition_smoothness)
                        extended_song = self.create_crossfade(extended_song, section_audio, crossfade_ms)
                        
                        current_duration = len(extended_song) / 1000
        
        # Add outro if we haven't already reached the target duration
        if current_duration < target_duration and "outro" in sections:
            crossfade_ms = int(1000 * request.transition_smoothness)
            extended_song = self.create_crossfade(extended_song, sections["outro"], crossfade_ms)
        
        # Export the extended song
        output_file = self.get_unique_filename("extended_song", ".mp3")
        
        # In low resource mode, use a lower bitrate
        bitrate = "128k" if self.low_resource_mode else "192k"
        extended_song.export(output_file, format="mp3", bitrate=bitrate)
        
        return output_file

    def extend_song_with_regeneration(self, request: SongExtensionRequest, style_description=None):
        """Extend a song by both structuring the original and generating new content."""
        input_file = request.input_file
        target_duration = request.target_duration
        
        # Free memory before starting
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Determine song structure based on genre or use provided structure
        structure = request.structure
        if not structure:
            if request.genre and request.genre.lower() in SONG_STRUCTURES:
                structure = SONG_STRUCTURES[request.genre.lower()]
            else:
                structure = DEFAULT_STRUCTURE
        
        # Load the input audio
        try:
            source_audio = AudioSegment.from_file(input_file)
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {str(e)}")
        
        # Get source audio duration in seconds
        source_duration = len(source_audio) / 1000
        
        # If source is already longer than target, just return it
        if source_duration >= target_duration:
            return input_file
        
        # If source is very short (less than 2 seconds), it's too short to work with
        if source_duration < 2:
            raise ValueError(f"Source audio is too short ({source_duration:.2f}s). Minimum 2 seconds required.")
        
        # Ensure source is long enough to be divided into sections
        min_section_duration_ms = 500  # Minimum 500ms per section
        min_total_duration_ms = min_section_duration_ms * 4  # Need at least 4 sections
        
        if len(source_audio) < min_total_duration_ms:
            # If source is too short, loop it to reach minimum length
            loops_needed = (min_total_duration_ms // len(source_audio)) + 1
            source_audio = source_audio * loops_needed
            print(f"Source audio was too short. Looped {loops_needed} times to reach minimum length.")
        
        # Find optimal loop points
        waveform, sample_rate = self.load_audio(input_file)
        loop_points = self.find_best_loop_points(waveform, sample_rate)
        
        # Free memory after intensive computation
        if self.device == "cuda":
            torch.cuda.empty_cache()
            del waveform
        
        # Create segments for each section of the song
        sections = {}
        
        # Create intro (first 25% of source)
        intro_duration = int(len(source_audio) * 0.25)
        sections["intro"] = source_audio[:intro_duration]
        
        # Create outro (last 25% of source)
        outro_duration = int(len(source_audio) * 0.25)
        sections["outro"] = source_audio[-outro_duration:]
        
        # Create main sections using different parts of the source
        source_segments = []
        segment_count = 4  # Divide source into 4 segments
        segment_length = len(source_audio) // segment_count
        
        for i in range(segment_count):
            start = i * segment_length
            end = start + segment_length
            source_segments.append(source_audio[start:end])
        
        # Map segments to section types
        sections["verse"] = source_segments[0]
        sections["chorus"] = source_segments[1]
        sections["bridge"] = source_segments[2]
        sections["solo"] = source_segments[3]
        
        # For electronic music specific sections
        sections["buildup"] = self.apply_variation(source_segments[0], request.variation_amount * 1.5)
        sections["drop"] = self.apply_variation(source_segments[1], request.variation_amount * 2)
        sections["breakdown"] = self.apply_variation(source_segments[2], request.variation_amount)
        
        # For classical/orchestral music
        sections["exposition"] = source_segments[0]
        sections["development"] = self.apply_variation(source_segments[1], request.variation_amount * 1.2)
        sections["recapitulation"] = self.apply_variation(source_segments[0], request.variation_amount * 0.8)
        sections["coda"] = source_segments[3]
        
        # For jazz
        sections["head"] = source_segments[0]
        sections["solo"] = self.apply_variation(source_segments[1], request.variation_amount * 1.5)
        
        # Generic sections
        sections["section_a"] = source_segments[0]
        sections["section_b"] = source_segments[1]
        sections["theme"] = source_segments[0]
        sections["variation"] = self.apply_variation(source_segments[0], request.variation_amount * 1.2)
        sections["climax"] = self.apply_variation(source_segments[1], request.variation_amount * 2)
        sections["resolution"] = self.apply_variation(source_segments[3], request.variation_amount * 0.5)
        
        # If regeneration is requested, create new content for some sections
        if request.regenerate_content and style_description:
            try:
                # Generate new content for key sections
                print("Generating new content for extension...")
                
                # Generate new verse and chorus variations
                new_verse = self.generate_new_content(f"A verse section in the style of {style_description}", 5)
                new_chorus = self.generate_new_content(f"A chorus section in the style of {style_description}", 5)
                new_bridge = self.generate_new_content(f"A bridge section in the style of {style_description}", 5)
                
                # Add the new sections
                sections["new_verse"] = new_verse
                sections["new_chorus"] = new_chorus
                sections["new_bridge"] = new_bridge
                
                # Create a modified structure that incorporates new content
                enhanced_structure = []
                for section in structure:
                    enhanced_structure.append(section)  # Add original section
                    
                    # Add new content after certain sections
                    if section == "verse":
                        enhanced_structure.append("new_verse")
                    elif section == "chorus":
                        enhanced_structure.append("new_chorus")
                    elif section == "bridge":
                        enhanced_structure.append("new_bridge")
                
                # Use the enhanced structure
                structure = enhanced_structure
                
            except Exception as e:
                print(f"Error during content regeneration: {str(e)}")
                # Continue with original structure if regeneration fails
        
        # Build the extended song following the structure
        extended_song = AudioSegment.empty()
        
        # Keep track of which sections have been used to apply variations to repeats
        used_sections = set()
        
        # In low resource mode, process in smaller batches
        if self.low_resource_mode:
            # Process structure in chunks to avoid memory issues
            chunk_size = 4  # Process 4 sections at a time
            
            for i in range(0, len(structure), chunk_size):
                chunk_structure = structure[i:i+chunk_size]
                chunk_song = AudioSegment.empty()
                
                for section_name in chunk_structure:
                    if section_name in sections:
                        section_audio = sections[section_name]
                        
                        # Apply variation if this section type has been used before
                        if section_name in used_sections:
                            section_audio = self.apply_variation(section_audio, request.variation_amount)
                        
                        # Add to the chunk with crossfade if not the first section
                        if len(chunk_song) > 0:
                            crossfade_ms = int(1000 * request.transition_smoothness)
                            chunk_song = self.create_crossfade(chunk_song, section_audio, crossfade_ms)
                        else:
                            chunk_song += section_audio
                        
                        used_sections.add(section_name)
                
                # Add chunk to extended song with crossfade
                if len(extended_song) > 0 and len(chunk_song) > 0:
                    crossfade_ms = int(1000 * request.transition_smoothness)
                    extended_song = self.create_crossfade(extended_song, chunk_song, crossfade_ms)
                elif len(chunk_song) > 0:
                    extended_song += chunk_song
                
                # Free memory after each chunk
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        else:
            # Standard processing for high-resource mode
            for section_name in structure:
                if section_name in sections:
                    section_audio = sections[section_name]
                    
                    # Apply variation if this section type has been used before
                    if section_name in used_sections:
                        section_audio = self.apply_variation(section_audio, request.variation_amount)
                    
                    # Add to the extended song with crossfade if not the first section
                    if len(extended_song) > 0:
                        crossfade_ms = int(1000 * request.transition_smoothness)
                        extended_song = self.create_crossfade(extended_song, section_audio, crossfade_ms)
                    else:
                        extended_song += section_audio
                    
                    used_sections.add(section_name)
        
        # Check if we've reached the target duration
        current_duration = len(extended_song) / 1000  # in seconds
        
        # If we need more duration, repeat the structure
        while current_duration < target_duration:
            # In low resource mode, process repeats in smaller batches
            if self.low_resource_mode:
                # Take a subset of the structure to repeat (avoid intro/outro)
                repeat_structure = structure[1:-1]
                chunk_size = min(3, len(repeat_structure))  # Smaller chunks for repeats
                
                for i in range(0, len(repeat_structure), chunk_size):
                    if current_duration >= target_duration:
                        break
                        
                    chunk_structure = repeat_structure[i:i+chunk_size]
                    chunk_song = AudioSegment.empty()
                    
                    for section_name in chunk_structure:
                        if current_duration >= target_duration:
                            break
                            
                        if section_name in sections:
                            section_audio = sections[section_name]
                            # Always apply variation for repeated structures
                            section_audio = self.apply_variation(section_audio, request.variation_amount)
                            
                            # Add with crossfade
                            if len(chunk_song) > 0:
                                crossfade_ms = int(1000 * request.transition_smoothness)
                                chunk_song = self.create_crossfade(chunk_song, section_audio, crossfade_ms)
                            else:
                                chunk_song += section_audio
                    
                    # Add chunk to extended song with crossfade
                    if len(chunk_song) > 0:
                        crossfade_ms = int(1000 * request.transition_smoothness)
                        extended_song = self.create_crossfade(extended_song, chunk_song, crossfade_ms)
                        current_duration = len(extended_song) / 1000
                    
                    # Free memory after each chunk
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
            else:
                # Standard processing for high-resource mode
                for section_name in structure[1:-1]:  # Skip intro and outro for repeats
                    if current_duration >= target_duration:
                        break
                        
                    if section_name in sections:
                        section_audio = sections[section_name]
                        # Always apply variation for repeated structures
                        section_audio = self.apply_variation(section_audio, request.variation_amount)
                        
                        # Add with crossfade
                        crossfade_ms = int(1000 * request.transition_smoothness)
                        extended_song = self.create_crossfade(extended_song, section_audio, crossfade_ms)
                        
                        current_duration = len(extended_song) / 1000
        
        # Add outro if we haven't already reached the target duration
        if current_duration < target_duration and "outro" in sections:
            crossfade_ms = int(1000 * request.transition_smoothness)
            extended_song = self.create_crossfade(extended_song, sections["outro"], crossfade_ms)
        
        # Export the extended song
        output_file = self.get_unique_filename("extended_song", ".mp3")
        
        # In low resource mode, use a lower bitrate
        bitrate = "128k" if self.low_resource_mode else "192k"
        extended_song.export(output_file, format="mp3", bitrate=bitrate)
        
        return output_file

# Create an instance of the SongExtender
song_extender = SongExtender()

def extend_audio_file(input_file, target_duration=180, genre=None, variation_amount=0.3, transition_smoothness=0.5, low_resource_mode=False, regenerate_content=False, style_description=None):
    """Function to extend an audio file into a structured song."""
    request = SongExtensionRequest(
        input_file=input_file,
        target_duration=target_duration,
        genre=genre,
        variation_amount=variation_amount,
        transition_smoothness=transition_smoothness,
        low_resource_mode=low_resource_mode,
        regenerate_content=regenerate_content
    )
    
    if regenerate_content:
        return song_extender.extend_song_with_regeneration(request, style_description)
    else:
        return song_extender.extend_song(request) 