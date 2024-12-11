import yt_dlp
import json
import os

# Get the directory of the script and construct the metadata file path
script_dir = os.path.dirname(os.path.abspath(__file__))
METADATA_FILE = os.path.join(script_dir, "downloaded.json")

def load_metadata():
    """Load metadata from the JSON file if it exists."""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as file:
                print("Loading metadata from JSON file...")
                return json.load(file)
        except json.JSONDecodeError:
            print("Metadata file is empty or invalid. Starting fresh.")
            return {}
    print("Metadata file does not exist. Starting fresh.")
    return {}

def save_metadata(metadata):
    """Save metadata to the JSON file."""
    print("Saving metadata to JSON file...")
    with open(METADATA_FILE, "w") as file:
        json.dump(metadata, file, indent=4)
    print("Metadata saved successfully.")

def download_video(query, metadata):
    """Download video using yt-dlp if not already downloaded."""
    print(f"Searching for '{query}' on YouTube...")
    search_query = f"ytsearch:{query}"
    ydl_opts = {
        'format': 'best',
        # Save the video with its original title initially
        'outtmpl': os.path.join(script_dir, '%(title)s.%(ext)s'),
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        search_results = ydl.extract_info(search_query, download=False)
        first_video = search_results['entries'][0]

        video_id = first_video['id']
        video_title = first_video['title']

        if video_id in metadata:
            print(f"Skipping '{video_title}' (already downloaded).")
            return metadata

        # Sanitize the title and create the output file path
        sanitized_title = video_title.replace(" ", "_").replace("|", "").replace("｜", "").replace(":", "")
        output_file = os.path.join(script_dir, f"{sanitized_title}.mp4")

        print(f"Downloading '{video_title}'...")
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        # Find the actual downloaded file
        actual_files = os.listdir(script_dir)
        downloaded_file = next(
            (os.path.join(script_dir, f) for f in actual_files if video_title.split(" ")[0] in f and f.endswith(".mp4")),
            None
        )

        if downloaded_file:
            print(f"Downloaded file found: {downloaded_file}")
            if downloaded_file != output_file:
                print(f"Renaming from {downloaded_file} to {output_file}")
                os.rename(downloaded_file, output_file)
            else:
                print(f"File already has the correct name: {output_file}")
        else:
            print("No matching downloaded file found.")


        print(f"Download completed: {output_file}")
        metadata[video_id] = {
            'title': sanitized_title,
            'url': f"https://www.youtube.com/watch?v={video_id}",
        }
        return metadata

def main():
    # Search term for the video
    search_term = "super mario movie trailer"
    print(f"Starting video search for: {search_term}")

    # Load metadata from the JSON file
    metadata = load_metadata()
    print("Loaded metadata:", metadata)

    # Download the video if not already downloaded
    updated_metadata = download_video(search_term, metadata)
    save_metadata(updated_metadata)
    print("Updated metadata:", updated_metadata)

    # Get the video path from the metadata
    video_info = list(updated_metadata.values())[0]  # Get the first video's metadata
    video_title = video_info['title']
    video_path = os.path.join(script_dir, f"{video_title.replace(' ', '_').replace('|', '').replace(':', '')}.mp4")
    print(f"Video path: {video_path}")

    # Check if the video file exists in the directory
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at path: {video_path}")
        return
if __name__ == "__main__":
    main()