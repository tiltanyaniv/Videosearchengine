# 🎥 VideoSearchEngine

VideoSearchEngine is a Python tool that downloads a YouTube video (e.g., movie trailer), detects scene changes, generates image captions using a vision-language model, and allows users to **search for specific moments** using either:
- An **image-based AI model** (Moondream)
- A **video-based AI model** (Gemini 1.5 Flash from Google)

It supports both fuzzy text matching and natural language queries, and visualizes the matching scenes in a collage.

---

## 🚀 Features

- 🔍 Search YouTube for a video and download it using `yt-dlp`
- 🎬 Automatically detect scenes using `PySceneDetect`
- 🧠 Generate scene captions using a local Moondream model
- 🤖 Search for scenes using fuzzy word matching or Gemini's video understanding
- 🧩 Create a visual collage of matching scenes
- 🗂️ Save metadata, captions, and collages for reuse

---

## 🛠️ Requirements

Install required packages:

	•	yt-dlp
	•	opencv-python
	•	scenedetect
	•	Pillow
	•	rapidfuzz
	•	moondream (custom or local module)
	•	google-generativeai
	•	prompt_toolkit
	•	numpy

 ---

 ## 📁 Folder Structure

 Ex2-Videosearchenginenew/
 ├── models/
 │   └── moondream-2b-int8.mf         # Pretrained Moondream model file
 ├── search_engine.py                 # Main script
 ├── scene_captions.json              # Auto-generated captions
 ├── downloaded.json                  # Metadata of downloaded videos
 ├── image_scenes/                    # Scene images from scene detection
 ├── video_scenes/                    # Gemini-detected scene images
 └── collage.png                      # Collage image of matching scenes

 ---

 ## ⚙️ How to Use

 1.Run the main script:
  
    python search_engine.py
 2.	Choose how you want to search:
	  •	1 → Image-based model (Moondream)
	  •	2 → Video-based model (Gemini 1.5 Flash)
 3.	When using Gemini, you’ll be prompted to type a natural language query (e.g., “show me scenes with Mario jumping”).

## 🔑 Environment Variable (For Gemini)
If you’re using the Gemini-based search, set your API key:
   ```bash
   export GEMINI_API_KEY="your_google_generative_ai_key"
   ```
## 🧠 Caption Output Example
   ```bash
   {
     "1": "A red car speeds through the city.",
     "2": "A man jumps over a building.",
     "3": "Explosion behind a character running away."
   }
   ```
   

## 🖼️ Result: Scene Collage

At the end of a successful search, a file called collage.png will be created in the root directory. It contains a grid of the relevant scene images found.

## 👩‍💻 Created by

Tiltan Yaniv

## 📄 License

This project is for educational use only.
