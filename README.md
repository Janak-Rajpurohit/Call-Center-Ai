# 🎧 Call Center AI

An intelligent, voice-enabled customer service assistant powered by Google Gemini, LangChain, and Whisper. It answers FAQs, processes refund/replacement requests, and provides a real-time analytics dashboard — all through a Streamlit web interface.

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Text Chatbot** | Conversational AI for customer support queries |
| 🎙️ **Voice Input** | Real-time speech-to-text via OpenAI Whisper |
| 🔊 **Text-to-Speech** | AI voice responses using pyttsx3/gTTS |
| 🛒 **Order Management** | Verify orders and process refunds/replacements |
| 📚 **FAQ Retrieval** | RAG-based answers from an e-commerce FAQ knowledge base |
| 📊 **Analytics Dashboard** | Sentiment analysis, timelines, word clouds, and emoji insights |
| 💾 **CRM Integration** | Customer lookup and session tracking |
| 🧠 **Memory** | Conversation buffer memory across a session |

---

## 🏗️ Architecture

```
Call-Center-Ai/
├── 1_Chatbot.py                  # Main page – text-based AI chat
├── tts_audio_processor.py        # TTS (pyttsx3) & audio utilities
├── pages/
│   ├── 2_Audio_Assistant.py      # Real-time voice AI assistant (WebRTC)
│   └── 3_Analysis.py             # Analytics dashboard (sentiment, word cloud, emoji)
├── utils/
│   ├── improved_call_center_ai.py # Core CallCenterAI class (RAG + LangChain agent)
│   ├── live_sst_updated.py        # Whisper-based speech-to-text transcriber
│   └── sentiment.py               # Sentiment analysis helpers
├── data/
│   ├── Ecommerce_FAQs.csv         # FAQ knowledge base
│   ├── CRM.csv                    # Customer records
│   ├── operations_log.csv         # Refund/replacement audit log
│   ├── chat_log.csv               # Conversation history
│   └── vector_db/                 # ChromaDB persisted vector store
├── .streamlit/
│   ├── config.toml                # Streamlit theme & server config
│   └── secrets.toml               # API keys (not committed)
└── requirements.txt
```

### Core Components

- **CallCenterAI** (`utils/improved_call_center_ai.py`) — orchestrates the LangChain agent with tools for FAQ retrieval (ChromaDB + HuggingFace embeddings), CRM lookup, and operations logging. Backed by Google Gemini (`gemini-pro`) via `langchain-google-genai`.
- **WhisperTranscriber** (`utils/live_sst_updated.py`) — wraps `faster-whisper` for efficient local speech recognition.
- **TTSManager / AudioProcessor** (`tts_audio_processor.py`) — async text-to-speech and WebRTC audio processing.
- **Analytics** (`pages/3_Analysis.py`) — Plotly/Matplotlib dashboards over the chat log and operations log, including VADER/transformer sentiment scoring.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- A [Google AI Studio](https://aistudio.google.com/) API key (Gemini)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Janak-Rajpurohit/Call-Center-Ai.git
cd Call-Center-Ai

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

Set your Google API key before launching. The recommended approach is via Streamlit secrets:

```toml
# .streamlit/secrets.toml
GOOGLE_API_KEY = "your-api-key-here"
```

Alternatively, set an environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

> **Note:** Never commit your API key to version control.

### Run the App

```bash
streamlit run 1_Chatbot.py
```

The app will open at `http://localhost:8501` with three pages in the sidebar:

| Page | Description |
|---|---|
| **1 – Chatbot** | Type messages and chat with the AI agent |
| **2 – Audio Assistant** | Speak directly using your microphone |
| **3 – Analysis** | View analytics over past conversations |

---

## 🧩 How It Works

1. **FAQ RAG Pipeline** — E-commerce FAQs are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in a local ChromaDB vector store. Relevant chunks are retrieved at query time and passed to Gemini via a `RetrievalQA` chain.

2. **Agentic Tool Use** — The LangChain agent selects from tools (FAQ search, CRM lookup, refund/replacement processing) based on the user's intent. Actions are logged to `data/operations_log.csv`.

3. **Voice Flow** — Audio is captured via `streamlit-webrtc`, decoded with `av`, transcribed by Whisper locally, fed to the same AI agent, and the response is spoken back with pyttsx3.

4. **Analytics** — `3_Analysis.py` reads `data/chat_log.csv` and `data/operations_log.csv` to generate interactive Plotly charts for sentiment trends, operation summaries, word clouds, and emoji usage.

---

## 📦 Key Dependencies

| Library | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `langchain` / `langchain-community` | Agent & chain orchestration |
| `langchain-google-genai` | Gemini LLM integration |
| `chromadb` | Local vector database |
| `sentence-transformers` | Text embeddings |
| `faster-whisper` | Local speech recognition |
| `pyttsx3` / `gTTS` | Text-to-speech |
| `streamlit-webrtc` | Real-time audio/video in browser |
| `plotly` / `matplotlib` | Data visualizations |
| `transformers` | Sentiment analysis models |

Full list: [`requirements.txt`](requirements.txt)

---

## 📁 Data Files

| File | Description |
|---|---|
| `data/Ecommerce_FAQs.csv` | Question/Answer/Category rows for the knowledge base |
| `data/CRM.csv` | Customer records (order ID, name, contact, etc.) |
| `data/operations_log.csv` | Audit log for refunds and replacements |
| `data/chat_log.csv` | Auto-generated conversation history |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source. See the repository for license details.
