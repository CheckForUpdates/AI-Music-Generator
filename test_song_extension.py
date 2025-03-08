import requests
import time
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Test the AI Music Generator with Song Extension')
    parser.add_argument('--prompt', type=str, default='A cheerful pop song with piano and drums',
                        help='Text description of the music to generate')
    parser.add_argument('--extend', action='store_true',
                        help='Whether to extend the clip into a full song')
    parser.add_argument('--duration', type=int, default=180,
                        help='Target duration for the extended song in seconds')
    parser.add_argument('--genre', type=str, default='pop',
                        help='Genre for song structure')
    parser.add_argument('--server', type=str, default='http://localhost:8000',
                        help='Server URL')
    parser.add_argument('--high-performance', action='store_true',
                        help='Use high performance mode (disable low resource mode)')
    parser.add_argument('--compare', action='store_true',
                        help='Generate both low resource and high performance versions for comparison')
    
    args = parser.parse_args()
    
    print(f"Generating music with prompt: '{args.prompt}'")
    
    if args.compare and args.extend:
        # Generate both low resource and high performance versions
        generate_and_compare(args)
    else:
        # Generate a single version
        low_resource_mode = not args.high_performance
        mode_name = "low resource" if low_resource_mode else "high performance"
        print(f"Using {mode_name} mode")
        
        # Start timer
        start_time = time.time()
        
        # Generate music
        response = requests.post(f"{args.server}/generate_music/", json={
            "prompt": args.prompt,
            "extend_to_full_song": args.extend,
            "target_duration": args.duration,
            "genre": args.genre,
            "low_resource_mode": low_resource_mode
        })
        
        # End timer
        end_time = time.time()
        
        process_response(response, args, end_time - start_time, mode_name)

def generate_and_compare(args):
    """Generate both low resource and high performance versions for comparison."""
    print("Generating both low resource and high performance versions for comparison...")
    
    # Low resource version
    print("\n=== LOW RESOURCE MODE ===")
    start_time_low = time.time()
    response_low = requests.post(f"{args.server}/generate_music/", json={
        "prompt": args.prompt,
        "extend_to_full_song": True,
        "target_duration": args.duration,
        "genre": args.genre,
        "low_resource_mode": True
    })
    end_time_low = time.time()
    low_resource_time = end_time_low - start_time_low
    
    # Process and save low resource version
    if response_low.status_code == 200:
        result_low = response_low.json()
        print(f"Generation successful in {low_resource_time:.2f} seconds")
        print(f"Message: {result_low.get('message')}")
        
        # Download the file
        download_url = f"{args.server}/download_extended_music/"
        download_response = requests.get(download_url)
        
        if download_response.status_code == 200:
            output_file = "downloaded_extended_song_low_resource.mp3"
            with open(output_file, 'wb') as f:
                f.write(download_response.content)
            
            print(f"Downloaded file to {output_file}")
            print(f"File size: {Path(output_file).stat().st_size / 1024:.2f} KB")
    else:
        print(f"Error with low resource mode: {response_low.status_code}")
        print(response_low.text)
    
    # High performance version
    print("\n=== HIGH PERFORMANCE MODE ===")
    start_time_high = time.time()
    response_high = requests.post(f"{args.server}/generate_music/", json={
        "prompt": args.prompt,
        "extend_to_full_song": True,
        "target_duration": args.duration,
        "genre": args.genre,
        "low_resource_mode": False
    })
    end_time_high = time.time()
    high_performance_time = end_time_high - start_time_high
    
    # Process and save high performance version
    if response_high.status_code == 200:
        result_high = response_high.json()
        print(f"Generation successful in {high_performance_time:.2f} seconds")
        print(f"Message: {result_high.get('message')}")
        
        # Download the file
        download_url = f"{args.server}/download_extended_music/"
        download_response = requests.get(download_url)
        
        if download_response.status_code == 200:
            output_file = "downloaded_extended_song_high_performance.mp3"
            with open(output_file, 'wb') as f:
                f.write(download_response.content)
            
            print(f"Downloaded file to {output_file}")
            print(f"File size: {Path(output_file).stat().st_size / 1024:.2f} KB")
    else:
        print(f"Error with high performance mode: {response_high.status_code}")
        print(response_high.text)
    
    # Print comparison
    print("\n=== COMPARISON ===")
    print(f"Low resource mode: {low_resource_time:.2f} seconds")
    print(f"High performance mode: {high_performance_time:.2f} seconds")
    print(f"Difference: {high_performance_time - low_resource_time:.2f} seconds")
    
    if os.path.exists("downloaded_extended_song_low_resource.mp3") and os.path.exists("downloaded_extended_song_high_performance.mp3"):
        low_size = Path("downloaded_extended_song_low_resource.mp3").stat().st_size / 1024
        high_size = Path("downloaded_extended_song_high_performance.mp3").stat().st_size / 1024
        print(f"Low resource file size: {low_size:.2f} KB")
        print(f"High performance file size: {high_size:.2f} KB")
        print(f"Size difference: {high_size - low_size:.2f} KB")

def process_response(response, args, elapsed_time, mode_name):
    """Process the API response and download the file."""
    if response.status_code == 200:
        result = response.json()
        print(f"Generation successful in {elapsed_time:.2f} seconds")
        print(f"Message: {result.get('message')}")
        
        # Download the file
        if args.extend:
            download_url = f"{args.server}/download_extended_music/"
            output_file = f"downloaded_extended_song_{mode_name.replace(' ', '_')}.mp3"
        else:
            download_url = f"{args.server}/download_music/"
            output_file = f"downloaded_song_{mode_name.replace(' ', '_')}.mp3"
        
        download_response = requests.get(download_url)
        
        if download_response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(download_response.content)
            
            print(f"Downloaded file to {output_file}")
            print(f"File size: {Path(output_file).stat().st_size / 1024:.2f} KB")
            
            # Try to play the file if on Linux
            if os.name == 'posix':
                try:
                    os.system(f"xdg-open {output_file}")
                except:
                    print(f"File saved to {output_file}. Please open it manually.")
        else:
            print(f"Failed to download file: {download_response.text}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main() 