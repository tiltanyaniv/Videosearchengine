import yt_dlp
import json
import os
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images
import cv2
import numpy as np
import moondream as md
from PIL import Image
from rapidfuzz import fuzz
from math import ceil, sqrt
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
import google.generativeai as genai
import time


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


def detect_scenes(video_path, output_folder):
    """Detect scenes and save images to the output folder."""
    # Check if scenes already exist
    if os.path.exists(output_folder) and any(f.endswith(".jpg") for f in os.listdir(output_folder)):
        print(f"Scenes already exist in {output_folder}. Skipping scene detection.")
        return

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=45.0))  # Adjust threshold here
    video_manager.set_downscale_factor()

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Retrieve the list of detected scenes
    scenes = scene_manager.get_scene_list()
    print(f"Detected {len(scenes)} scenes.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save images for each scene
    save_images(
        scene_list=scenes,
        video=video_manager,
        num_images=1,  # Adjust to the desired number of images per scene
        frame_margin=1,
        image_extension='jpg',
        encoder_param=95,
        image_name_template='$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER',
        output_dir=output_folder,
        show_progress=True,  # Displays a progress bar if tqdm is installed
    )

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
        [f for f in os.listdir(scene_folder) if f.endswith(".jpg")],  # Ensure only image files are considered
        key=lambda x: int(x.split("-Scene-")[1].split("-")[0])  # Extract the scene number from the filename
    )

    for scene_file in scene_files:
        scene_number = int(scene_file.split("-Scene-")[1].split("-")[0]) # Extract scene number
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


def search_scenes_with_autocomplete(captions_file, threshold=70):
    """
    Search scenes for a specific word or similar words in captions using RapidFuzz with auto-complete.
    """
    if not os.path.exists(captions_file):
        print(f"Error: Captions file not found at path: {captions_file}")
        return

    # Load captions from the JSON file
    with open(captions_file, "r") as file:
        captions = json.load(file)

    # Initialize the auto-complete session
    session = PromptSession(completer=CaptionCompleter(captions))

    print("Search the video using a word or similar words (with auto-complete):")

    try:
        search_word = session.prompt("Enter a word: ").strip().lower()
    except KeyboardInterrupt:
        print("\nSearch canceled.")
        return
    except EOFError:
        print("\nExiting search.")
        return

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
        
        # Create a collage of matching scenes
        scene_folder = os.path.join(script_dir, "image_scenes")
        create_collage_image_model(scene_folder, matching_scenes)
    else:
        print(f"No scenes found with captions similar to '{search_word}' (threshold: {threshold}%).")

def create_collage_image_model(scene_folder, matching_scenes):
    """
    Create a collage of all matching scene images and save it to 'collage.png'.
    """
    if not matching_scenes:
        print("No matching scenes to create a collage.")
        return

    # Get paths of matching scene images
    image_paths = []
    for scene in matching_scenes.keys():
        # Use the new naming convention to match scene files
        pattern = f"-Scene-{str(scene).zfill(3)}-"  # Match Scene numbers padded to 3 digits
        matching_files = [
            f for f in os.listdir(scene_folder)
            if pattern in f and f.endswith(".jpg")
        ]
        if matching_files:
            image_paths.append(os.path.join(scene_folder, matching_files[0]))  # Add the first match

    if not image_paths:
        print("No images found for matching scenes.")
        return

    print(f"Creating a collage of {len(image_paths)} scenes...")

    # Open all images
    images = [Image.open(path) for path in image_paths]

    # Get dimensions for the collage grid
    num_images = len(images)
    grid_size = ceil(sqrt(num_images))  # Determine grid size (square root of total images)
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    # Create a blank image for the collage
    collage_width = grid_size * max_width
    collage_height = grid_size * max_height
    collage = Image.new("RGB", (collage_width, collage_height), (255, 255, 255))

    # Paste each image into the collage
    for idx, img in enumerate(images):
        x = (idx % grid_size) * max_width
        y = (idx // grid_size) * max_height
        collage.paste(img, (x, y))

    # Save the collage to a file
    collage_file = os.path.join(script_dir, "collage.png")
    collage.save(collage_file)
    print(f"Collage created and saved as '{collage_file}'.")

def create_collage_video_model(output_folder, collage_path):
    """
    Create a collage from images in the output folder and save it as a single file.
    :param output_folder: Folder containing scene images.
    :param collage_path: Path to save the collage image.
    """
    image_paths = [
        os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(".jpg")
    ]
    if not image_paths:
        print("No images found to create a collage.")
        return
    # Load images
    images = [Image.open(path) for path in image_paths]
    # Calculate collage grid dimensions
    num_images = len(images)
    grid_size = ceil(sqrt(num_images))
    image_width, image_height = images[0].size
    # Create blank canvas for the collage
    collage_width = grid_size * image_width
    collage_height = grid_size * image_height
    collage = Image.new("RGB", (collage_width, collage_height), (255, 255, 255))
    # Paste images into the collage
    for idx, img in enumerate(images):
        x = (idx % grid_size) * image_width
        y = (idx // grid_size) * image_height
        collage.paste(img, (x, y))
    # Save the collage
    collage.save(collage_path)

class CaptionCompleter(Completer):
    """
    Custom Completer for captions. Suggests words from captions as user types.
    """
    def __init__(self, captions):
        # Extract all unique words from captions
        self.words = set()
        for caption in captions.values():
            self.words.update(caption.lower().split())
    
    def get_completions(self, document, complete_event):
        """
        Provide completions for the current input.
        """
        text = document.text.lower()  # Get the current input text
        for word in sorted(self.words):  # Suggest matching words
            if word.startswith(text):  # Match words starting with the input
                yield Completion(word, start_position=-len(text))


def search_video_with_gemini(video_path):
    """
    Use Google's Generative AI library to analyze a video using Gemini 1.5 Flash.
    """
    # Configure the Gemini API
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    genai.configure(api_key=api_key)

    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Ask the user for a descriptive query
    query = input("Using a video model. What would you like me to find in the video? ").strip()
    if not query:
        print("No query provided. Exiting.")
        return

    # Upload the video file
    try:
        print(f"Uploading video file: {video_path}")
        video_file = genai.upload_file(video_path)
    except Exception as e:
        print(f"Error uploading video: {e}")
        return

    # Wait for the file to become ACTIVE
    if not wait_for_file_to_be_active(video_file):
        print("File did not become ACTIVE. Exiting.")
        return

    # Use the Gemini model to analyze the uploaded file
    print("Analyzing video with Gemini 1.5 Flash...")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            [f"""Analyze the uploaded video and identify the specific timestamps or frame ranges for scenes that match the provided query.
            For each identified scene, provide:
            2. The start time in seconds.
            3. The end time in seconds.
            Return the result as a valid JSON array with no additional text, markers, or formatting. Example of the desired response:
            [
            {{
                "start": 12.5,
                "end": 14.0
            }},
            {{
                "start": 45.3,
                "end": 50.2
            }}
            ]
            Query: {query}""", video_file],
            generation_config=genai.GenerationConfig(
                max_output_tokens=400,  # Limit the response to 400 tokens
                temperature=0.4  # Set the temperature for response creativity
            )
        )

        # Handle the response
        if response and hasattr(response, "text"):
            response_text = response.text.strip()

            # Remove all backticks and markers (generalized cleaning)
            response_text = response_text.replace("```json", "").replace("```", "").strip()

            try:
                # Parse the cleaned response
                timestamps = json.loads(response_text)
                print("Parsed timestamps:")
                print(json.dumps(timestamps, indent=4))  # Nicely formatted output for verification

                # Process timestamps: extract frames and create a collage
                output_folder = os.path.join(script_dir, "video_scenes")
                extract_frames(video_path, timestamps, output_folder)

                collage_path = os.path.join(script_dir, "collage.png")
                create_collage_video_model(output_folder, collage_path)

            except json.JSONDecodeError as e:
                # Log error if JSON parsing fails
                print(f"Error decoding JSON response: {e}")
                print(f"Raw response after cleaning: {response_text}")
                return
        else:
            print("Error: No valid response received from the Gemini API.")
            return
    except Exception as e:
        print(f"An error occurred while analyzing the video: {e}")


def wait_for_file_to_be_active(video_file, wait_time=10, max_wait_time=300):
    """
    Wait for the uploaded file to become ACTIVE using the Google AI API file status check.
    :param video_file: Uploaded video file object.
    :param wait_time: Time to wait between retries (in seconds).
    :param max_wait_time: Maximum time to wait before giving up (in seconds).
    :return: True if the file becomes ACTIVE, False otherwise.
    """
    file_name = video_file.name  # Extract the file name from the video file object
    total_wait_time = 0  # Keep track of total time waited

    while total_wait_time < max_wait_time:
        try:
            # Use the API to check the file's status
            file_status_response = genai.get_file(file_name)
            state = file_status_response.state.name  # Assume 'state' indicates the file's readiness
            if state == "ACTIVE":
                print(f"File {file_name} is ACTIVE.")
                return True
            print(f"File {file_name} is in {state} state. Retrying in {wait_time} seconds...")
        except Exception as e:
            print(f"Error while checking file status: {e}")

        time.sleep(wait_time)
        total_wait_time += wait_time

    print(f"File {file_name} did not become ACTIVE within {max_wait_time} seconds.")
    return False

def extract_frames(video_path, timestamps, output_folder):
    """
    Extract frames corresponding to the given timestamps from the video.
    :param video_path: Path to the video file.
    :param timestamps: List of timestamps with start and end times.
    :param output_folder: Folder to save the extracted frames.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for idx, timestamp in enumerate(timestamps):
        start_time = timestamp["start"]
        end_time = timestamp["end"]

        # Calculate the middle frame between start and end
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        middle_frame = (start_frame + end_frame) // 2

        # Set the video position to the middle frame and read it
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_folder, f"scene_{idx + 1}.jpg")
            cv2.imwrite(output_path, frame)
        else:
            print(f"Warning: Unable to extract frame for scene {idx + 1}")

    cap.release()
    print(f"Frames saved to {output_folder}.")




def main():
    print("Choose the search method:")
    print("1. Search using an image-based model")
    print("2. Search using a video-based model (Google Gemini 1.5 Flash)")
    choice = input("Enter your choice (1 or 2): ").strip()

    # Define the video path for the Mario video
    search_term = "super mario movie trailer"
    print(f"Starting video search for: {search_term}")

    # Load metadata
    metadata = load_metadata()

    # Download video
    updated_metadata = download_video(search_term, metadata)
    save_metadata(updated_metadata)

    # Get video path
    video_info = list(updated_metadata.values())[0]
    video_title = video_info["title"]
    video_path = os.path.join(script_dir, f"{video_title.replace(' ', '_').replace('|', '').replace(':', '')}.mp4")
    print(f"Video path: {video_path}")

    # Check video existence
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at path: {video_path}")
        return

    if choice == "1":
        # Run the existing image-based search
        print("Using the image-based model for search...")

        # Detect scenes
        output_folder = os.path.join(script_dir, "image_scenes")
        detect_scenes(video_path, output_folder)

        # Generate captions
        model = load_model()
        captions_file = os.path.join(script_dir, "scene_captions.json")
        generate_captions(output_folder, captions_file, model)

        # Search captions
        search_scenes_with_autocomplete(captions_file, threshold=75)

    elif choice == "2":
        # Run the video-based search using Gemini
        print("Using the video-based model for search...")
        search_video_with_gemini(video_path)

    else:
        print("Invalid choice. Exiting program.")

if __name__ == "__main__":
    main()