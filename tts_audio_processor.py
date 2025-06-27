from streamlit_webrtc import AudioProcessorBase
import pyttsx3
import threading
import queue
import base64
import time
import av
import streamlit as st

class TTSManager:
    def __init__(self):
        self.engine = None
        self.audio_queue = queue.Queue()
        self.initialize_tts()

    def initialize_tts(self):
        """Initialize TTS engine with error handling."""
        try:
            self.engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to set a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice
                    self.engine.setProperty('voice', voices[0].id)
            
            # Set speech rate and volume
            self.engine.setProperty('rate', 180)  # Speed of speech
            self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            
        except Exception as e:
            st.error(f"TTS initialization failed: {e}")
            self.engine = None

    def speak(self, text):
        """Convert text to speech."""
        if self.engine and text:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                st.error(f"TTS error: {e}")

    def speak_async(self, text):
        """Speak text in a separate thread."""
        if text:
            thread = threading.Thread(target=self.speak, args=(text,))
            thread.daemon = True
            thread.start()

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Store audio frames for processing
        self.audio_frames.append(frame.to_ndarray())
        return frame