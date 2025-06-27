import streamlit as st
import pandas as pd
import os
import tempfile
from datetime import datetime
from io import BytesIO
import soundfile as sf
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Import your custom modules
from utils.improved_call_center_ai import CallCenterAI
from utils.live_sst_updated import WhisperTranscriber
from tts_audio_processor import TTSManager, AudioProcessor

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'ai_agent' not in st.session_state:
        st.session_state.ai_agent = None
    if 'tts_manager' not in st.session_state:
        st.session_state.tts_manager = TTSManager()
    if 'whisper_transcriber' not in st.session_state:
        st.session_state.whisper_transcriber = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'audio_input_method' not in st.session_state:
        st.session_state.audio_input_method = "Upload File"

def setup_ai_agent():
    """Setup the AI agent with configuration."""
    with st.spinner("Setting up AI Agent..."):
        try:
            # Get API key
            google_api_key = "AIzaSyBEvK-CIpTTKOT-ErOvMTZm6W8UuHvI8NY"
            if not google_api_key:
                st.error("Google API Key not found. Please set it in secrets or environment variables.")
                return False
            
            # Initialize AI agent
            ai_agent = CallCenterAI(
                google_api_key=google_api_key,
                data_path="data/Ecommerce_FAQs.csv",  # Update path as needed
                crm_path="data/CRM.csv",  # Update path as needed
                persist_directory="data/vector_db",
                operations_log_path="data/operations_log.csv"
            )
            
            # Setup AI components
            ai_agent.setup_qa_chain()
            ai_agent.initialize_tools()
            ai_agent.initialize_agent()
            
            st.session_state.ai_agent = ai_agent
            st.success("AI Agent initialized successfully!")
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize AI Agent: {e}")
            return False

def setup_whisper():
    """Setup Whisper transcriber."""
    if st.session_state.whisper_transcriber is None:
        with st.spinner("Loading Whisper model..."):
            try:
                st.session_state.whisper_transcriber = WhisperTranscriber(model_size="base")
                st.success("Whisper model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load Whisper model: {e}")


def display_chat_history():
    """Display chat history in a clean format."""
    if st.session_state.chat_history:
        st.subheader("Conversation History")
        
        for i, (user_msg, ai_msg, timestamp) in enumerate(st.session_state.chat_history):
            with st.container():
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"**You** ({timestamp}):")
                    st.markdown(f"🗣️ {user_msg}")
                
                with col2:
                    st.markdown(f"**AI Agent** ({timestamp}):")
                    st.markdown(f"🤖 {ai_msg}")
                
                st.divider()

def save_chat_to_csv(user_msg, ai_msg, timestamp):
    log_file = "data/chat_log.csv"
    new_row = {"User": user_msg, "AI": ai_msg, "Timestamp": timestamp}
    if not os.path.exists(log_file):
        df = pd.DataFrame([new_row])
        df.to_csv(log_file, index=False)
    else:
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(log_file, index=False)

def process_user_query(query, tts_enabled=True):
    """Process user query and generate response."""
    if not st.session_state.ai_agent:
        st.error("AI Agent not initialized.")
        return
    
    try:
        with st.spinner("Processing your request..."):
            # Get AI response
            response = st.session_state.ai_agent.process_query(query)
        
        # Add to chat history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append((query, response, timestamp))
        save_chat_to_csv(query, response, timestamp)
        

        
        # Display response
        st.success("AI Response:")
        st.markdown(f"🤖 {response}")
        
        # Text-to-speech
        if tts_enabled and st.session_state.tts_manager:
            st.session_state.tts_manager.speak_async(response)
        
        # Auto-refresh to show updated chat
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing query: {e}")



        
def main():
    st.set_page_config(
        page_title="Call Center AI Assistant",
        page_icon="🎧",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for configuration
    
    if not st.session_state.ai_agent:
        setup_ai_agent()
        
        # Initialize Whisper
        # if st.button("Load Whisper Model"):
    if not st.session_state.whisper_transcriber:
        setup_whisper()
        
        
        # TTS Settings
        
    tts_enabled = st.checkbox("Enable AI Voice Response", value=True)
    
    # Main interface
    st.title("🎧 Call Center AI Assistant")
    st.markdown("Welcome to our AI-powered customer service! I can help you with FAQs, process refunds, handle replacements, and more.")
    
    # Check if AI agent is initialized
    if st.session_state.ai_agent is None:
        st.warning("Please initialize the AI Agent from the sidebar to get started.")
        return
    
    # Input methods
    
    
    
    # Text input
    st.subheader("💬 Chat with AI")
    user_input = st.text_input(
        "Type your message:",
        placeholder="e.g., I want to return my order ORD123",
        key="text_input"
    )
    
    if st.button("Send Message", type="primary") and user_input:
        process_user_query(user_input, tts_enabled)

    # Display chat history
    display_chat_history()
    
    # Footer with helpful information
    st.markdown("---")
    with st.expander("ℹ️ How to use this assistant"):
        st.markdown("""
        **Getting Help:**
        - Ask general questions about products, policies, or services
        - Request refunds or replacements (you'll need your Order ID)
        - Use voice input for hands-free interaction
        
        **Order Operations:**
        - For refunds/replacements, provide your Order ID (e.g., "ORD123")
        - The AI will verify your order before processing requests
        - All operations are logged for tracking
        
        **Voice Features:**
        - Upload audio files or use live recording
        - Enable TTS to hear AI responses
        - Supports multiple audio formats
        """)





if __name__ == "__main__":
    main()
    