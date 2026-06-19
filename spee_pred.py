import os
import pandas as pd
import moviepy.editor as mp
import joblib
import logging
import speech_recognition as sr

# Configure logging
logging.basicConfig(filename="speech_recognition.log", level=logging.INFO)

# Load trained ML model & vectorizer using raw strings
model_file = r"C:\Users\gnana\Desktop\Final yr Project\Code\backend\trained_model.pkl"
vectorizer_file = r"C:\Users\gnana\Desktop\Final yr Project\Code\backend\vectorizer.pkl"
csv_file = r"C:\Users\gnana\Desktop\Final yr Project\Code\backend\dataset.csv"
video_folder = r"C:\Users\gnana\Desktop\Final yr Project\Code\backend\video"

# Check if model & vectorizer exist
if not os.path.exists(model_file) or not os.path.exists(vectorizer_file):
    print("⚠️ Error: Model files not found! Train the model first.")
    exit()

model = joblib.load(model_file)
vectorizer = joblib.load(vectorizer_file)

# Load dataset for video mapping
if not os.path.exists(csv_file):
    print(f"⚠️ Error: Dataset file '{csv_file}' not found!")
    exit()

df = pd.read_csv(csv_file)
video_dict = dict(zip(df["Label"], df["Video Name"]))

# Set the video folder path
if not os.path.exists(video_folder):
    print(f"⚠️ Error: Video folder '{video_folder}' not found!")
    exit()

# Initialize the Recognizer
recognizer = sr.Recognizer()

# Function to transcribe speech
def transcribe_speech(max_attempts=3):
    attempts = 0

    while attempts < max_attempts:
        with sr.Microphone() as source:
            print("🎤 Speak now...")
            recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            recognizer.energy_threshold = 300  # Lower values make it more sensitive
            recognizer.pause_threshold = 1.0   # Wait for 1 second of silence to end a phrase

            try:
                # Listen to the audio input
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                print("⏳ Processing...")

                # Use Google Speech-to-Text API for transcription
                text = recognizer.recognize_google(audio)
                logging.info(f"Successful transcription: {text}")
                print(f"📝 Transcribed Text: {text}")
                return text

            except sr.UnknownValueError:
                logging.warning(f"Failed transcription attempt {attempts + 1}: UnknownValueError")
                print(f"⚠️ Could not understand audio. Please try again. Attempt {attempts + 1} of {max_attempts}.")
            except sr.RequestError as e:
                logging.error(f"Speech recognition service error: {e}")
                print(f"⚠️ Speech recognition service error: {e}")
                break

        attempts += 1

    print("⚠️ Maximum attempts reached.")
    return None

# Main Program
input_mode = input("Choose input mode (text/speech): ").strip().lower()
if input_mode == "speech":
    print("Trying online speech recognition...")
    input_text = transcribe_speech()  # Try online speech recognition
else:
    input_text = input("Enter text: ")

# Fallback to text input if speech fails
if not input_text:
    print("⚠️ Switching to text input...")
    input_text = input("Enter text: ")

if not input_text:
    print("⚠️ No input provided. Exiting.")
    exit()

# Predict Labels
words = input_text.split()
predicted_labels = []

for word in words:
    input_vector = vectorizer.transform([word])
    prediction = model.predict(input_vector)
    predicted_labels.extend(prediction)

# Preserve Duplicates by Default
print(f"🔹 Predicted Labels (Ordered): {predicted_labels}")

# Find Matching Videos
video_files = []
for label in predicted_labels:
    if label in video_dict:
        video_path = os.path.join(video_folder, video_dict[label])
        if os.path.exists(video_path):
            video_files.append(video_path)
        else:
            print(f"⚠️ Warning: Video file '{video_dict[label]}' not found in '{video_folder}'.")
    else:
        print(f"⚠️ Warning: Label '{label}' not found in dataset.")

# Merge Videos
if video_files:
    clips = [mp.VideoFileClip(video).resize((1280, 720)) for video in video_files]
    final_clip = mp.concatenate_videoclips(clips, method="compose")

    output_path = "merged_output.mp4"
    final_clip.write_videofile(output_path, codec="libx264", fps=30)
    print(f"✅ Merged video saved as '{output_path}'")
else:
    print("⚠️ No matching videos found!")
    