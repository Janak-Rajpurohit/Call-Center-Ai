import os
import time
import warnings
import torch
import numpy as np
import wave
import pyaudio
from faster_whisper import WhisperModel

class WhisperTranscriber:
    def __init__(self, model_size="large-v2"):
        #self.transcriber = WhisperTranscriber(model_size="large-v2")
        warnings.filterwarnings("ignore")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model_size = model_size
        
        # Measure the time taken to load the model
        start_time = time.time()
        print("Loading Faster Whisper model...")
        
        # Use mixed precision (float16) by setting compute_type to 'float16'
        self.model = WhisperModel(model_size, device=self.device, compute_type="int8")
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds.")

    def transcribe_and_translate(self, audio_file, language_code="hi"):
        start_time = time.time()
        print("Transcribing and translating audio...")

        # Perform transcription and translation using the provided language code
        segments, info = self.model.transcribe(audio_file, beam_size=2, task="translate", language=language_code)
        transcription = "".join([segment.text for segment in segments])
        
        # Measure the time taken
        process_time = time.time() - start_time
        print(f"Process completed in {process_time:.2f} seconds.")
        
        return {"transcription": transcription, "language": language_code}

def get_whisper_languages():
    """
    Returns the list of supported languages in Whisper.
    """
    return {
        "Afrikaans": "af",
        "Arabic": "ar",
        "Assamese": "as",
        "Azerbaijani": "az",
        "Bashkir": "ba",
        "Basque": "eu",
        "Belarusian": "be",
        "Bengali": "bn",
        "Bosnian": "bs",
        "Bulgarian": "bg",
        "Catalan": "ca",
        "Chinese (Simplified)": "zh",
        "Chinese (Traditional)": "zh-TW",
        "Croatian": "hr",
        "Czech": "cs",
        "Danish": "da",
        "Dutch": "nl",
        "English": "en",
        "Estonian": "et",
        "Finnish": "fi",
        "French": "fr",
        "Galician": "gl",
        "Georgian": "ka",
        "German": "de",
        "Greek": "el",
        "Gujarati": "gu",
        "Hebrew": "he",
        "Hindi": "hi",
        "Hungarian": "hu",
        "Icelandic": "is",
        "Indonesian": "id",
        "Italian": "it",
        "Japanese": "ja",
        "Kannada": "kn",
        "Kazakh": "kk",
        "Korean": "ko",
        "Latvian": "lv",
        "Lithuanian": "lt",
        "Macedonian": "mk",
        "Malay": "ms",
        "Malayalam": "ml",
        "Maltese": "mt",
        "Marathi": "mr",
        "Mongolian": "mn",
        "Nepali": "ne",
        "Norwegian": "no",
        "Oriya": "or",
        "Persian": "fa",
        "Polish": "pl",
        "Portuguese": "pt",
        "Punjabi": "pa",
        "Romanian": "ro",
        "Russian": "ru",
        "Serbian": "sr",
        "Slovak": "sk",
        "Slovenian": "sl",
        "Spanish": "es",
        "Swahili": "sw",
        "Swedish": "sv",
        "Tamil": "ta",
        "Telugu": "te",
        "Thai": "th",
        "Turkish": "tr",
        "Ukrainian": "uk",
        "Urdu": "ur",
        "Uzbek": "uz",
        "Vietnamese": "vi",
        "Welsh": "cy",
        "Yiddish": "yi",
        "Zulu": "zu",
    }

# Replace the earlier dynamic call with this function
languages = get_whisper_languages()

def get_language_code():
    """
    Gets the language code from the user, ensuring valid input.
    """
    choice = input("Do you want to see the available language codes? (Type 'yes' to view, or enter your language code directly): ").strip().lower()
    
    if choice == 'yes':
        print("\nAvailable languages:")
        for language, code in languages.items():
            print(f"{language}: {code}")
        
        # Ask the user to input the language code after showing the list
        detected_language = input("\nEnter your language code from the above list (e.g., 'hi', 'en', 'es'): ").strip().lower()
    else:
        detected_language = choice  # Assume the input is the language code itself

    # Validate the entered language code
    if detected_language in languages.values():
        return detected_language
    else:
        print("Invalid language code entered. Please try again.")
        return get_language_code()  # Recursively ask again
def measure_ambient_noise(duration=2, sample_rate=8000):
    """
    Measures the ambient noise level for a specified duration.

    Args:
        duration (int): Duration (in seconds) to measure ambient noise.
        sample_rate (int): The sample rate for audio.

    Returns:
        float: Estimated silence threshold based on ambient noise.
    """
    print("Measuring ambient noise...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)
    
    noise_levels = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        noise_levels.append(np.abs(audio_data).mean())
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    avg_noise = np.mean(noise_levels)
    print(f"Ambient noise level: {avg_noise}")
    return avg_noise * 1.5  # Set threshold slightly above ambient noise

def record_audio(filename="recorded_audio.wav", sample_rate=8000, silence_duration=3, min_recording_time=5):
    """
    Records audio from the microphone and saves it as a .wav file.
    Stops recording when silence is detected for a specified duration.

    Args:
        filename (str): The output file name.
        sample_rate (int): The sample rate for the recording.
        silence_duration (int): Duration (in seconds) of continuous silence to stop recording.
        min_recording_time (int): Minimum time (in seconds) before recording can stop.

    Returns:
        str: Path to the saved audio file.
    """
    silence_threshold = measure_ambient_noise()
    print("Recording... Please speak clearly.")
    p = pyaudio.PyAudio()

    # Set up audio stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    frames = []
    silence_frames = 0
    max_silence_frames = int(sample_rate / 1024 * silence_duration)

    # Track recording start time
    start_time = time.time()

    # Use a buffer for smoothing silence detection
    energy_buffer = []

    while True:
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

        # Convert data to numpy array for analysis
        audio_data = np.frombuffer(data, dtype=np.int16)
        audio_energy = np.abs(audio_data).mean()

        # Append to energy buffer and calculate smoothed energy
        energy_buffer.append(audio_energy)
        if len(energy_buffer) > 5:  # Smooth over 5 frames
            energy_buffer.pop(0)

        smoothed_energy = np.mean(energy_buffer)

        # Check for silence
        if smoothed_energy < silence_threshold:
            silence_frames += 1
        else:
            silence_frames = 0

        # Ensure minimum recording time is met
        elapsed_time = time.time() - start_time
        if silence_frames > max_silence_frames and elapsed_time > min_recording_time:
            print("Silence detected. Stopping recording.")
            break

    print("Recording complete. Saving audio...")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as {filename}")
    return filename

if __name__ == "__main__":
    #detected_language = input("Select the language code (e.g., 'en', 'es', 'fr'): ")  # Language code is provided as input
    detected_language = get_language_code()
    transcriber = WhisperTranscriber(model_size="large-v2")

    while True:
        choice = input("Please Record Your Query [Press 'R/r'] or Provide An Existing File Path [Press F/f] ? Type 'exit' to quit: ").lower()
        
        if choice.lower().strip() == "exit":
            print("Exiting program.")
            break
        
        if choice.lower().strip() in ["r"]:
            print("Recording audio...")
            audio_file = record_audio()
        elif choice.lower().strip() in ["f"]:
            audio_file = input("Enter audio file path: ")
            if not os.path.exists(audio_file):
                print("File not found. Please enter a valid file path.")
                continue
        
        total_start_time = time.time()
        result = transcriber.transcribe_and_translate(audio_file, detected_language)
        total_time = time.time() - total_start_time

        print("\n--- Speech-to-Text Results ---")
        print(f"Transcription: {result['transcription']}")
        print(f"Language: {result['language']}")
        print(f"Total time for processing: {total_time:.2f} seconds.")
