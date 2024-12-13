import yt_dlp
import json
import os
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
import numpy as np
import moondream as md
from PIL import Image
from rapidfuzz import fuzz

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

def is_black_scene(image_path, threshold=10):
    """Check if the image is predominantly black."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False  # Skip if image couldn't be loaded
    # Calculate the percentage of pixels below the black threshold
    black_pixel_count = np.sum(image < threshold)
    total_pixel_count = image.size
    black_percentage = (black_pixel_count / total_pixel_count) * 100
    return black_percentage > 90  # Consider as black if >90% pixels are black

def detect_scenes(video_path, output_folder):
    """Detect scenes and save images to the output folder."""
    # Check if scenes already exist
    if os.path.exists(output_folder) and any(f.endswith(".jpg") for f in os.listdir(output_folder)):
        print(f"Scenes already exist in {output_folder}. Skipping scene detection.")
        return

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=40.0))  # Adjust threshold here
    video_manager.set_downscale_factor()

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Retrieve the list of detected scenes
    scenes = scene_manager.get_scene_list()
    print(f"Detected {len(scenes)} scenes.")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, scene in enumerate(scenes):
        start_frame, end_frame = scene
        output_path = f"{output_folder}/scene_{i+1}.jpg"
        # OpenCV to extract and save frames
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame.get_frames())  # Set the frame position
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(output_path, frame)  # Save the frame as an image
        cap.release()

        # Remove black scenes
        if is_black_scene(output_path):
            os.remove(output_path)

    video_manager.release()

def load_model():
    print("Loading Moondream model...")
    # Construct the relative path to the model file
    model_path = os.path.join(script_dir, "models", "moondream-2b-int8.mf")
    
    # Ensure the file exists before loading
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Load the model using the relative path
    model = md.vl(model=model_path)
    print("Model loaded successfully.")
    return model

def generate_captions(scene_folder, output_file, model):
    """
    Generate captions for each scene image and save them to a JSON file.
    """
    # Check if captions already exist
    if os.path.exists(output_file):
        print(f"Captions file already exists at {output_file}. Skipping caption generation.")
        return

    captions = {}
    # Sort files numerically by scene number
    scene_files = sorted(
        [f for f in os.listdir(scene_folder) if f.startswith("scene_") and f.endswith(".jpg")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    for scene_file in scene_files:
        scene_number = int(scene_file.split("_")[1].split(".")[0])  # Extract scene number
        scene_path = os.path.join(scene_folder, scene_file)
        print(f"Generating caption for scene {scene_number}...")

        # Load and process the image
        image = Image.open(scene_path)
        encoded_image = model.encode_image(image)

        # Generate caption
        caption = model.caption(encoded_image)["caption"]
        captions[scene_number] = caption

    # Save the captions to a JSON file
    print(f"Saving captions to {output_file}...")
    with open(output_file, "w") as json_file:
        json.dump(captions, json_file, indent=4)
    print("Captions saved successfully.")


def search_scenes_by_word(captions_file, threshold=70):
    """
    Search scenes for a specific word or similar words in captions using RapidFuzz.
    """
    if not os.path.exists(captions_file):
        print(f"Error: Captions file not found at path: {captions_file}")
        return

    # Load captions from the JSON file
    with open(captions_file, "r") as file:
        captions = json.load(file)
    
    print("Search the video using a word or similar words:")
    search_word = input("Enter a word: ").strip().lower()

    if not search_word:
        print("No word entered. Please try again.")
        return

    # Find scenes with captions similar to the input word
    matching_scenes = {}
    for scene, caption in captions.items():
        similarity = fuzz.partial_ratio(search_word, caption.lower())
        if similarity >= threshold:
            matching_scenes[scene] = (caption, similarity)

    if matching_scenes:
        print(f"Scenes with captions similar to '{search_word}' (threshold: {threshold}%):")
        for scene, (caption, similarity) in matching_scenes.items():
            print(f"Scene {scene}: {caption} (Similarity: {similarity}%)")
    else:
        print(f"No scenes found with captions similar to '{search_word}' (threshold: {threshold}%).")

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
    
    # Define the output folder for saving scene images
    output_folder = os.path.join(script_dir, "scenes")  # Folder to save extracted scenes
    print(f"Output folder for scenes: {output_folder}")

    # Detect scenes and save scene images
    print("Detecting scenes...")
    detect_scenes(video_path, output_folder)
    print(f"Scenes saved in {output_folder}")

    # Load the model for image captioning
    model = load_model()

    # Generate captions for scenes
    captions_file = os.path.join(script_dir, "scene_captions.json")
    generate_captions(output_folder, captions_file, model)
    print(f"Captions saved to {captions_file}")

    # Allow the user to search scenes by a word with a threshold for similarity
    search_scenes_by_word(captions_file, threshold=75)  # Adjust the threshold as needed



if __name__ == "__main__":
    main()