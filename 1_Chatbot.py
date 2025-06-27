import streamlit as st
import pandas as pd
import os
import tempfile
from datetime import datetime
from io import BytesIO
import soundfile as sf
import numpy as np
# from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Import your custom modules
from utils.improved_call_center_ai import CallCenterAI
from utils.live_sst_updated import WhisperTranscriber
from tts_audio_processor import TTSManager, AudioProcessor

# Custom CSS for beautiful UI
def load_custom_css():
    st.markdown("""
    <style>
        /* Main app styling */
        .main {
            padding-top: 1rem;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Chat container */
        .chat-container {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            border-radius: 15px;
            padding: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            margin-bottom: 1rem;
            min-height: 400px;
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        
        /* Message styling */
        .user-message {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 20px 20px 5px 20px;
            margin: 0.5rem 0 0.5rem auto;
            max-width: 80%;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            word-wrap: break-word;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .ai-message {
            background: linear-gradient(135deg, #32364A 0%, #2A2D3E 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 20px 20px 20px 5px;
            margin: 0.5rem auto 0.5rem 0;
            max-width: 80%;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            word-wrap: break-word;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        /* Input container */
        .input-container {
            position: sticky;
            bottom: 0;
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            padding: 1rem;
            border-radius: 15px;
            box-shadow: 0 -4px 20px rgba(0,0,0,0.2);
            border: 1px solid rgba(255,255,255,0.1);
            margin-top: 1rem;
            backdrop-filter: blur(10px);
        }
        
        /* Status indicators */
        .status-success {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            color: #4facfe;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            text-align: center;
            border: 1px solid rgba(79,172,254,0.3);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            color: white;
            border: 1px solid rgba(255,255,255,0.1);
            padding: 0.5rem 2rem;
            border-radius: 25px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        /* Input field styling */
        .stTextInput > div > div > input {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            color: white;
            border-radius: 25px;
            border: 1px solid rgba(255,255,255,0.1);
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }
        
        /* Welcome message */
        .welcome-message {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 2rem 0;
            color: white;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        /* Scrollbar styling */
        .chat-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: #2A2D3E;
            border-radius: 10px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        .chat-container::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
    </style>
    """, unsafe_allow_html=True)

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
                data_path="data/Ecommerce_FAQs.csv",
                crm_path="data/CRM.csv",
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
    """Display chat history in a modern chat interface."""
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="welcome-message">
            <h3>👋 Welcome to our AI Customer Service!</h3>
            <p>I'm here to help you with FAQs, refunds, replacements, and more. How can I assist you today?</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display in a container with max height and scrolling
        with st.container():
            for user_msg, ai_msg, timestamp in st.session_state.chat_history:
                # User message
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin: 1rem 0;">
                    <div class="user-message">
                        <div><strong>You</strong> - {timestamp}</div>
                        <div style="margin-top: 0.5rem;">{user_msg}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # AI message
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin: 1rem 0;">
                    <div class="ai-message">
                        <div><strong>🤖 AI Assistant</strong> - {timestamp}</div>
                        <div style="margin-top: 0.5rem;">{ai_msg}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

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
        
        # Text-to-speech
        if tts_enabled and st.session_state.tts_manager:
            st.session_state.tts_manager.speak_async(response)
        
        # Auto-refresh to show updated chat
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing query: {e}")

def main():
    st.set_page_config(
        page_title="AI Customer Service Assistant",
        page_icon="🎧",
        layout="wide"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Auto-initialize AI agent and Whisper like original
    if not st.session_state.ai_agent:
        setup_ai_agent()
        
    if not st.session_state.whisper_transcriber:
        setup_whisper()
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # TTS Settings
        tts_enabled = st.checkbox("Enable AI Voice Response", value=True)
        
        st.markdown("---")
        
        # Chat Controls
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🎧 AI Customer Service Assistant</h1>
        <p>Your intelligent companion for customer support and instant assistance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if AI agent is initialized
    if st.session_state.ai_agent is None:
        st.warning("Please wait while the AI Agent initializes...")
        return
    
    # Chat display area
    display_chat_history()
    
    # Input area at the bottom
    st.markdown("---")
    
    # Create input container
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message:",
            placeholder="e.g., I want to return my order ORD123",
            key="text_input"
        )
    
    with col2:
        send_button = st.button("Send Message", type="primary")
    
    # Process input
    if send_button and user_input:
        if user_input.strip():
            process_user_query(user_input.strip(), tts_enabled)

if __name__ == "__main__":
    main()