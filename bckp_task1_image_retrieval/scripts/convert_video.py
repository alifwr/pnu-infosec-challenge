import subprocess
import sys
import os

def convert_video(input_path, output_path=None):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    if output_path is None:
        filename, ext = os.path.splitext(input_path)
        output_path = f"{filename}_h264{ext}"
    
    # ffmpeg command to convert to H.264 (libx264)
    # -i input
    # -c:v libx264 (H.264 video codec)
    # -preset slow (better compression/quality ratio)
    # -crf 23 (standard quality)
    # -c:a aac (AAC audio codec, compatible with MP4)
    # -movflags +faststart (optimizes for web/streaming)
    # -y (overwrite output)
    
    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "22",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-y",
        output_path
    ]
    
    print(f"Converting '{input_path}' to '{output_path}'...")
    print("Running command:", " ".join(command))
    
    try:
        subprocess.run(command, check=True)
        print("\nConversion completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nError during conversion: {e}")
    except FileNotFoundError:
        print("\nError: ffmpeg not found. Please install ffmpeg.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_video.py <input_video_path> [output_video_path]")
    else:
        input_video = sys.argv[1]
        output_video = sys.argv[2] if len(sys.argv) > 2 else None
        convert_video(input_video, output_video)
