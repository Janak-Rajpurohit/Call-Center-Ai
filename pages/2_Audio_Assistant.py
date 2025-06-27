# import streamlit as st
# import pandas as pd
# import os
# import tempfile
# from datetime import datetime
# from io import BytesIO
# import soundfile as sf
# import numpy as np
# from streamlit_webrtc import webrtc_streamer, WebRtcMode

# # Import your custom modules
# from improved_call_center_ai import CallCenterAI
# from live_sst_updated import WhisperTranscriber
# from tts_audio_processor import TTSManager, AudioProcessor

# def initialize_session_state():
#     """Initialize Streamlit session state variables."""
#     if 'ai_agent' not in st.session_state:
#         st.session_state.ai_agent = None
#     if 'tts_manager' not in st.session_state:
#         st.session_state.tts_manager = TTSManager()
#     if 'whisper_transcriber' not in st.session_state:
#         st.session_state.whisper_transcriber = None
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
#     if 'audio_input_method' not in st.session_state:
#         st.session_state.audio_input_method = "Upload File"

# def setup_ai_agent():
#     """Setup the AI agent with configuration."""
#     with st.spinner("Setting up AI Agent..."):
#         try:
#             # Get API key
#             google_api_key = "AIzaSyBEvK-CIpTTKOT-ErOvMTZm6W8UuHvI8NY"
#             if not google_api_key:
#                 st.error("Google API Key not found. Please set it in secrets or environment variables.")
#                 return False
            
#             # Initialize AI agent
#             ai_agent = CallCenterAI(
#                 google_api_key=google_api_key,
#                 data_path="data/Ecommerce_FAQs.csv",  # Update path as needed
#                 crm_path="data/CRM.csv",  # Update path as needed
#                 persist_directory="data/vector_db",
#                 operations_log_path="data/operations_log.csv"
#             )
            
#             # Setup AI components
#             ai_agent.setup_qa_chain()
#             ai_agent.initialize_tools()
#             ai_agent.initialize_agent()
            
#             st.session_state.ai_agent = ai_agent
#             st.success("AI Agent initialized successfully!")
#             return True
            
#         except Exception as e:
#             st.error(f"Failed to initialize AI Agent: {e}")
#             return False

# def setup_whisper():
#     """Setup Whisper transcriber."""
#     if st.session_state.whisper_transcriber is None:
#         with st.spinner("Loading Whisper model..."):
#             try:
#                 st.session_state.whisper_transcriber = WhisperTranscriber(model_size="base")
#                 st.success("Whisper model loaded successfully!")
#             except Exception as e:
#                 st.error(f"Failed to load Whisper model: {e}")

# def process_audio_file(audio_file):
#     """Process uploaded audio file."""
#     if st.session_state.whisper_transcriber is None:
#         st.error("Whisper transcriber not initialized.")
#         return None
    
#     try:
#         # Save uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#             tmp_file.write(audio_file.read())
#             tmp_file_path = tmp_file.name
        
#         # Transcribe audio
#         with st.spinner("Transcribing audio..."):
#             result = st.session_state.whisper_transcriber.transcribe_and_translate(tmp_file_path)
        
#         # Clean up temporary file
#         os.unlink(tmp_file_path)
        
#         return result['transcription']
        
#     except Exception as e:
#         st.error(f"Audio processing error: {e}")
#         return None

# def display_chat_history():
#     """Display chat history in a clean format."""
#     if st.session_state.chat_history:
#         st.subheader("Conversation History")
        
#         for i, (user_msg, ai_msg, timestamp) in enumerate(st.session_state.chat_history):
#             with st.container():
#                 col1, col2 = st.columns([1, 1])
                
#                 with col1:
#                     st.markdown(f"**You** ({timestamp}):")
#                     st.markdown(f"🗣️ {user_msg}")
                
#                 with col2:
#                     st.markdown(f"**AI Agent** ({timestamp}):")
#                     st.markdown(f"🤖 {ai_msg}")
                
#                 st.divider()
        

# def save_chat_to_csv(user_msg, ai_msg, timestamp):
#     log_file = "data/chat_log.csv"
#     new_row = {"User": user_msg, "AI": ai_msg, "Timestamp": timestamp}
#     if not os.path.exists(log_file):
#         df = pd.DataFrame([new_row])
#         df.to_csv(log_file, index=False)
#     else:
#         df = pd.read_csv(log_file)
#         df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
#         df.to_csv(log_file, index=False)

# def process_user_query(query, tts_enabled=True):
#     """Process user query and generate response."""
#     if not st.session_state.ai_agent:
#         st.error("AI Agent not initialized.")
#         return
    
#     try:
#         with st.spinner("Processing your request..."):
#             # Get AI response
#             response = st.session_state.ai_agent.process_query(query)
        
#         # Add to chat history
#         timestamp = datetime.now().strftime("%H:%M:%S")
#         st.session_state.chat_history.append((query, response, timestamp))
#         save_chat_to_csv(query, response, timestamp)
        

        
#         # Display response
#         st.success("AI Response:")
#         st.markdown(f"🤖 {response}")
        
#         # Text-to-speech
#         if tts_enabled and st.session_state.tts_manager:
#             st.session_state.tts_manager.speak_async(response)
        
#         # Auto-refresh to show updated chat
#         st.rerun()
        
#     except Exception as e:
#         st.error(f"Error processing query: {e}")

# def main():
#     st.set_page_config(
#         page_title="Call Center AI Assistant",
#         page_icon="🎧",
#         layout="wide"
#     )
    
#     # Initialize session state
#     initialize_session_state()
    
#     # Sidebar for configuration
          
#     if not st.session_state.ai_agent:
#         setup_ai_agent()
        
#         # Initialize Whisper
#         # if st.button("Load Whisper Model"):
#     if not st.session_state.whisper_transcriber:
#         setup_whisper()
        
        
#     st.markdown("---")
        
#         # Audio Input Method
#     st.subheader("Audio Input")
#     audio_method = st.radio(
#         "Choose input method:",
#         ["Upload File", "Record Audio", "Live Recording"],
#         key="audio_method"
#     )
    
#     st.markdown("---")
    
#     # Operations Log
#     if st.session_state.ai_agent:
#         if st.button("View Operations Log"):
#             log_df = st.session_state.ai_agent.get_operations_log()
#             if not log_df.empty:
#                 st.subheader("Recent Operations")
#                 st.dataframe(log_df.tail(10))
#             else:
#                 st.info("No operations logged yet.")
    
#     # Reset Session
#     if st.button("Reset Conversation"):
#         st.session_state.chat_history = []
#         if st.session_state.ai_agent:
#             st.session_state.ai_agent.reset_user_session()
#         st.success("Conversation reset!")
#         st.rerun()

#     # Main interface
#     st.title("🎧 Call Center AI Assistant")
#     st.markdown("Welcome to our AI-powered customer service! I can help you with FAQs, process refunds, handle replacements, and more.")
    
#     # Check if AI agent is initialized
#     if st.session_state.ai_agent is None:
#         st.warning("Please initialize the AI Agent from the sidebar to get started.")
#         return
    
#     # Input methods
    
#     # Audio input section
#     st.subheader("🎤 Voice Input")
    
#     if audio_method == "Upload File":
#         uploaded_audio = st.file_uploader(
#             "Upload audio file",
#             type=['wav', 'mp3', 'ogg', 'flac'],
#             help="Upload an audio file to transcribe"
#         )
        
#         if uploaded_audio and st.button("Process Audio"):
#             transcription = process_audio_file(uploaded_audio)
#             if transcription:
#                 st.success(f"Transcribed: {transcription}")
#                 process_user_query(transcription, tts_enabled)
    
#     elif audio_method == "Live Recording":
#         st.info("Live recording feature - Click to start/stop")
        
#         webrtc_ctx = webrtc_streamer(
#             key="audio-recording",
#             mode=WebRtcMode.SENDONLY,
#             audio_processor_factory=AudioProcessor,
#             media_stream_constraints={"audio": True, "video": False},
#             rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
#         )
        
#         if webrtc_ctx.audio_processor and st.button("Process Recording"):
#             # Process recorded audio frames
#             st.info("Processing recorded audio...")

#     # Display chat history
#     display_chat_history()
    
#     # Footer with helpful information
#     st.markdown("---")
#     with st.expander("ℹ️ How to use this assistant"):
#         st.markdown("""
#         **Getting Help:**
#         - Ask general questions about products, policies, or services
#         - Request refunds or replacements (you'll need your Order ID)
#         - Use voice input for hands-free interaction
        
#         **Order Operations:**
#         - For refunds/replacements, provide your Order ID (e.g., "ORD123")
#         - The AI will verify your order before processing requests
#         - All operations are logged for tracking
        
#         **Voice Features:**
#         - Upload audio files or use live recording
#         - Enable TTS to hear AI responses
#         - Supports multiple audio formats
#         """)



# if __name__ == "__main__":
#     main()
    

import streamlit as st
import pandas as pd
import os
import tempfile
from datetime import datetime
import threading
import time
import queue
import numpy as np
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, WebRtcStreamerContext
import av
import logging
import asyncio
from typing import Optional
import speech_recognition as sr
import pyttsx3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'call_active' not in st.session_state:
        st.session_state.call_active = False
    if 'processing_audio' not in st.session_state:
        st.session_state.processing_audio = False
    if 'silence_threshold' not in st.session_state:
        st.session_state.silence_threshold = 0.02
    if 'call_status' not in st.session_state:
        st.session_state.call_status = "Disconnected"
    if 'audio_processor' not in st.session_state:
        st.session_state.audio_processor = None
    if 'tts_engine' not in st.session_state:
        # Initialize TTS engine
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Speed of speech
            engine.setProperty('volume', 0.8)  # Volume level
            st.session_state.tts_engine = engine
        except:
            st.session_state.tts_engine = None
    if 'speech_recognizer' not in st.session_state:
        st.session_state.speech_recognizer = sr.Recognizer()
        st.session_state.speech_recognizer.energy_threshold = 300
        st.session_state.speech_recognizer.dynamic_energy_threshold = True
        st.session_state.speech_recognizer.pause_threshold = 0.8

class SimpleAIAgent:
    """Simple AI agent for demonstration."""
    
    def __init__(self):
        self.conversation_history = []
        
    def process_query(self, query: str) -> str:
        """Process user query and return response."""
        # Simple rule-based responses for demonstration
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            response = "Hello! I'm your AI assistant. How can I help you today?"
        elif any(word in query_lower for word in ['order', 'purchase', 'buy']):
            response = "I can help you with orders. What would you like to order today?"
        elif any(word in query_lower for word in ['help', 'support', 'problem']):
            response = "I'm here to help! Please describe your issue and I'll do my best to assist you."
        elif any(word in query_lower for word in ['price', 'cost', 'money']):
            response = "I can provide pricing information. What product are you interested in?"
        elif any(word in query_lower for word in ['bye', 'goodbye', 'see you']):
            response = "Goodbye! Thank you for contacting us. Have a great day!"
        else:
            response = f"I understand you said: '{query}'. I'm here to help with orders, support, and general questions. What would you like to know?"
        
        # Store in conversation history
        self.conversation_history.append({"user": query, "ai": response, "timestamp": datetime.now()})
        
        return response

class RealTimeAudioProcessor:
    """Real-time audio processor for continuous speech recognition."""
    
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=10)
        self.sample_rate = 16000
        self.buffer = []
        self.buffer_duration_seconds = 2.0  # Process every 2 seconds
        self.min_buffer_size = int(self.sample_rate * self.buffer_duration_seconds)
        self.last_process_time = time.time()
        self.processing = False
        
    def add_audio_data(self, audio_data: np.ndarray):
        """Add audio data to buffer."""
        if audio_data is None or len(audio_data) == 0:
            return
            
        # Convert to float32 and ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        audio_data = audio_data.astype(np.float32)
        self.buffer.extend(audio_data)
        
        # Process if buffer is large enough
        if len(self.buffer) >= self.min_buffer_size and not self.processing:
            self._process_buffer()
    
    def _process_buffer(self):
        """Process the current buffer."""
        if not self.buffer or self.processing:
            return
            
        # Check for activity (simple energy-based)
        audio_array = np.array(self.buffer[:self.min_buffer_size], dtype=np.float32)
        energy = np.sqrt(np.mean(audio_array ** 2))
        
        if energy > st.session_state.silence_threshold:
            try:
                # Add to queue for processing
                if not self.audio_queue.full():
                    self.audio_queue.put(audio_array.copy())
            except:
                pass  # Queue full, skip this chunk
        
        # Clear processed portion of buffer
        self.buffer = self.buffer[self.min_buffer_size:]
        self.last_process_time = time.time()
    
    def get_audio_for_transcription(self) -> Optional[np.ndarray]:
        """Get audio data for transcription."""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    """Callback for processing audio frames."""
    if not st.session_state.call_active or st.session_state.audio_processor is None:
        return frame
    
    try:
        # Convert frame to numpy array
        audio_data = frame.to_ndarray()
        
        # Add to processor
        st.session_state.audio_processor.add_audio_data(audio_data)
        
    except Exception as e:
        logger.error(f"Error in audio callback: {e}")
    
    return frame

def transcribe_audio(audio_data: np.ndarray) -> str:
    """Transcribe audio using speech_recognition."""
    try:
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, audio_data, 16000)
            
            # Use speech_recognition
            with sr.AudioFile(tmp_file.name) as source:
                audio = st.session_state.speech_recognizer.record(source)
                
                try:
                    # Try Google Speech Recognition (free tier)
                    text = st.session_state.speech_recognizer.recognize_google(audio)
                    return text
                except sr.UnknownValueError:
                    # Try alternative: recognize_sphinx (offline, but less accurate)
                    try:
                        text = st.session_state.speech_recognizer.recognize_sphinx(audio)
                        return text
                    except:
                        return ""
                except sr.RequestError:
                    # If online services fail, return empty
                    return ""
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_file.name)
        except:
            pass

def speak_text(text: str):
    """Convert text to speech."""
    if st.session_state.tts_engine and text.strip():
        try:
            def _speak():
                st.session_state.tts_engine.say(text)
                st.session_state.tts_engine.runAndWait()
            
            # Run TTS in separate thread to avoid blocking
            threading.Thread(target=_speak, daemon=True).start()
        except Exception as e:
            logger.error(f"TTS error: {e}")

def process_audio_continuously():
    """Background thread to process audio continuously."""
    ai_agent = SimpleAIAgent()
    
    while st.session_state.call_active:
        try:
            if st.session_state.audio_processor:
                audio_data = st.session_state.audio_processor.get_audio_for_transcription()
                
                if audio_data is not None:
                    st.session_state.processing_audio = True
                    
                    # Transcribe
                    transcription = transcribe_audio(audio_data)
                    
                    if transcription and len(transcription.strip()) > 2:
                        # Process with AI
                        response = ai_agent.process_query(transcription)
                        
                        # Add to chat history
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state.chat_history.append({
                            "user": transcription,
                            "ai": response,
                            "timestamp": timestamp
                        })
                        
                        # Speak response
                        speak_text(response)
                    
                    st.session_state.processing_audio = False
                    
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            st.session_state.processing_audio = False
        
        time.sleep(0.1)

def display_call_interface():
    """Display the main call interface."""
    st.markdown("---")
    
    # Call status
    status_color = "🔴" if not st.session_state.call_active else "🟢"
    st.markdown(f"### {status_color} Status: {st.session_state.call_status}")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if not st.session_state.call_active:
            if st.button("📞 Start Call", type="primary", use_container_width=True):
                st.session_state.call_active = True
                st.session_state.call_status = "Connected"
                st.session_state.audio_processor = RealTimeAudioProcessor()
                
                # Start background processing
                threading.Thread(target=process_audio_continuously, daemon=True).start()
                st.rerun()
    
    with col2:
        if st.session_state.call_active:
            if st.button("📞 End Call", type="secondary", use_container_width=True):
                st.session_state.call_active = False
                st.session_state.call_status = "Disconnected"
                st.session_state.audio_processor = None
                st.rerun()
    
    with col3:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

def display_chat():
    """Display chat history."""
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation")
        
        # Show recent messages
        for i, msg in enumerate(st.session_state.chat_history[-5:]):  # Last 5 messages
            # User message
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                <strong>👤 You ({msg['timestamp']}):</strong><br>
                {msg['user']}
            </div>
            """, unsafe_allow_html=True)
            
            # AI response
            st.markdown(f"""
            <div style="background-color: #f3e5f5; padding: 10px; border-radius: 10px; margin: 5px 0;">
                <strong>🤖 AI ({msg['timestamp']}):</strong><br>
                {msg['ai']}
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("💡 Start the call and speak to begin conversation!")

def main():
    """Main application."""
    initialize_session_state()
    
    st.set_page_config(
        page_title="Real-Time Voice AI",
        page_icon="📞",
        layout="wide"
    )
    
    st.title("📞 Real-Time Voice AI Assistant")
    st.markdown("**Speak naturally and get AI responses in real-time**")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("## 🎛️ Audio Settings")
        
        # Silence threshold
        st.session_state.silence_threshold = st.slider(
            "Voice Detection Sensitivity",
            min_value=0.005,
            max_value=0.1,
            value=st.session_state.silence_threshold,
            step=0.005,
            help="Lower = more sensitive to quiet speech"
        )
        
        # Status indicators
        if st.session_state.processing_audio:
            st.markdown("🎤 **Processing Speech...**")
        elif st.session_state.call_active:
            st.markdown("🎤 **Listening...**")
        else:
            st.markdown("🎤 **Inactive**")
        
        # TTS Status
        if st.session_state.tts_engine:
            st.markdown("🔊 **Text-to-Speech: Ready**")
        else:
            st.markdown("🔊 **Text-to-Speech: Error**")
    
    # Main interface
    display_call_interface()
    
    # Audio streaming interface
    if st.session_state.call_active:
        st.markdown("### 🎙️ Voice Stream")
        
        webrtc_ctx = webrtc_streamer(
            key="voice-ai",
            mode=WebRtcMode.SENDONLY,
            audio_frame_callback=audio_frame_callback,
            media_stream_constraints={
                "audio": {
                    "echoCancellation": True,
                    "noiseSuppression": True,
                    "autoGainControl": True,
                    "sampleRate": 16000,
                    "channelCount": 1
                },
                "video": False
            },
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            async_processing=True
        )
        
        if webrtc_ctx.state.playing:
            st.success("🎙️ **Microphone Active** - Speak now!")
        else:
            st.warning("🎙️ **Click START to activate microphone**")
    
    # Display conversation
    display_chat()
    
    # Auto-refresh when active
    if st.session_state.call_active and st.session_state.chat_history:
        time.sleep(2)
        st.rerun()
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        **Instructions:**
        1. Click "Start Call" to begin
        2. Allow microphone access when prompted
        3. Click "START" on the audio stream
        4. Speak naturally - the AI will respond automatically
        5. Wait for responses before speaking again
        
        **Requirements:**
        ```bash
        pip install streamlit streamlit-webrtc speechrecognition pyttsx3 soundfile numpy pandas
        ```
        
        **Tips:**
        - Speak clearly at normal volume
        - Pause between sentences
        - Adjust sensitivity if needed
        - Check microphone permissions
        """)

if __name__ == "__main__":
    main()