import yt_dlp
import sys
import os
import signal

def signal_handler(sig, frame):
    print("\nDownload stopped by user (Ctrl+C). Exiting...")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def download_video(url, output_path="downloads"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading from {url}...")
            ydl.download([url])
            print("Download completed!")
    except KeyboardInterrupt:
        print("\nDownload stopped by user.")
    except Exception as e:
        print(f"Error downloading video: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_youtube.py <youtube_url>")
        # Example URL for testing or default behavior if needed, but better to ask user
        # url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 
    else:
        url = sys.argv[1]
        download_video(url)
