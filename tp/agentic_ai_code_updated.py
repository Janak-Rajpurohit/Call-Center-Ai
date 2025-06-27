"""import os
# os.environ['GOOGLE_API_KEY'] = 'AIzaSyDTLJ9oECPbo1f3Ru-Tf8ebzUVjRuBPLFY';
os.environ['GOOGLE_API_KEY'] = 'AIzaSyBEvK-CIpTTKOT-ErOvMTZm6W8UuHvI8NY';

if os.environ["GOOGLE_API_KEY"] == 'AIzaSyBEvK-CIpTTKOT-ErOvMTZm6W8UuHvI8NY':
    print({ "error": '''
        To get started, get an GOOGLE_API_KEY and enter it in the first step
    '''.replace('\n', '') })"""


import os
import time
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from google.api_core.exceptions import ResourceExhausted
from langchain.agents import initialize_agent,AgentType
from utils.live_sst_updated import *
from langchain.memory import ConversationBufferMemory
from langchain_core.caches import InMemoryCache
import re
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

class CallCenterAI:
    def __init__(self, google_api_key, data_path, crm_path, persist_directory, embedding_model="sentence-transformers/all-MiniLM-L6-v2",memory_limit=5):
        os.environ['GOOGLE_API_KEY'] = google_api_key
        self.data_path = data_path
        self.crm_path = crm_path
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vector_db = None
        self.qa_chain = None
        self.tools = []
        self.lang = 'English'
        self.agent = None
        self.transcriber = WhisperTranscriber(model_size="large-v2")
        self.lang = get_language_code()
        self.memory_limit = memory_limit
        self.cache = InMemoryCache()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Add memory here
        self.order_verified = False  # Track if order ID has been verified


    def load_faq_data(self):
        """Load FAQ data from a CSV file."""
        df = pd.read_csv(self.data_path, encoding='windows-1252')
        df['qa'] = df['Question'] + " " + df['Answer'] + " " + df['Category']
        return df['qa'].tolist()

    def convert_to_documents(self, data):
        """Convert data into LangChain-compatible Document objects."""
        return [Document(page_content=content, metadata={}) for content in data]

    def split_text_into_chunks(self, documents, chunk_size=500, chunk_overlap=20):
        """Split documents into smaller chunks."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(documents)

    def download_hf_embeddings(self):
        """Download and load HuggingFace embedding model."""
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

    def create_vector_db(self, chunks):
        """Create and persist the vector database."""
        self.vector_db = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=self.persist_directory)
        self.vector_db.persist()

    def setup_qa_chain(self):
        system_prompt = (
            "You are a highly intelligent, empathetic, and professional call center agent specializing in solving e-commerce FAQs. "
            "Your role is to assist users by providing clear, concise, and conversational responses tailored to their needs while maintaining context throughout the conversation. "
            "always listen to user first and react accordingly like a human."
            "Detect user language from user input and always Respond in the hybrid script of English and user's language, even if input query is in pure language reply in their own language, using transliterated Roman characters (the English alphabet). Write it as it sounds and do not repeat the sentences."
            "Speak casually and respectfully, incorporating expressions to mimic human speech but don't overreact too much; be professional. "
            "first listen to user's problem ask then how you can help."
            "Before processing anything like (replacement request) you need to authenticate user, for that you have to check order id exsist in crm data or not"
            "Ask for order id in respective manner and verify it."                
            "you have to authenticate user only ones using provided tools"
            "Always keep your responses very short/brief, human-like, and solution-oriented. "
            "If you need any information then ask for it to user dont ask them personal questions"
            "you have to use qa retrival all the time to answer user query for authenticating you have to use other tool ones"
            "If information is unavailable in the provided context, use reasoning and general knowledge to assist. "
            #"Always verify the query before responding to ensure accuracy and prevent misunderstandings. "
            "Avoid assumptions about the user’s gender or preferences. Craft responses that are universally respectful and inclusive. "
            "*Context Provided:* "
            "{context} "
            "Your ultimate goal is to make the user feel understood, valued, and satisfied with their interaction, ensuring a smooth and professional resolution to their concerns. "
            "Respond as a highly trained, empathetic, and human-like agent in every interaction."
        )

        prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])

        self.download_hf_embeddings()
        self.vector_db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",cache=self.cache)
        question_answer_chain = create_stuff_documents_chain(model, prompt)
        self.qa_chain = create_retrieval_chain(self.vector_db.as_retriever(search_kwargs={'k': 5}), question_answer_chain)

    def query(self, user_query: str) -> str:
        """Handle user queries using the QA chain."""
        result = self.qa_chain.invoke({"input":user_query})

        return result["answer"]

    def compare_order_id_with_crm(self, user_query: str) -> str:
     """Compare order ID against CRM data."""
     crm_data = self.load_crm_data()
    
     if crm_data is None or 'Order_ID' not in crm_data:
        self.order_verified = False
        return "CRM data unavailable or invalid."

     # Remove all whitespace from user query and convert to lowercase
     clean_query = ''.join(user_query.split()).lower()

     # Updated regex to handle potential spaces between characters
     order_id = re.search(r'o\s*r\s*d\s*\d\s*\d\s*\d', clean_query, re.IGNORECASE)
    
     if order_id:
        # Clean the matched order ID by removing spaces and normalizing to lowercase
        clean_order_id = ''.join(order_id.group(0).split()).lower()

        # Clean the CRM data order IDs similarly and normalize to lowercase
        clean_crm_orders = crm_data['Order_ID'].str.replace(r'\s+', '', regex=True).str.lower()

        if clean_order_id in clean_crm_orders.values:
            self.order_verified = True
            return "Order ID found"
    
     self.order_verified = False
     return "Order ID not found."


    def load_crm_data(self):
        """Load CRM data from a CSV file."""
        return pd.read_csv(self.crm_path)

    def initialize_tools(self):
        """Initialize tools for the agent."""

        self.tools = [
            Tool(
                name="Answer User",
                func=self.query,
                description= "Use this tool all the time to answer all user queries and Respond in the hybrid script of English and user's language, even if input query is in pure language, reply in in their own language, using transliterated Roman characters (the English alphabet) briefly. Write it as it sounds and do not repeat the sentences." 
                ),
            Tool(
                name="Authenticating User",
                func=self.compare_order_id_with_crm,
                description="When user ask for any operation process example(replacement),you need to verify user,so call this tool of authenticating user which return you the order id exsist or not. you have to only call this ones when you asked for order id from user. Always listen to user query first reply accordingly if needed then only ask for order id.After getting order id from 'Extract Order ID' tool, Use this tool To authenticate user ask when you need order id, before processing operations, to Check if an order ID exists in the CRM system. you have to only call this ones when you asked for order id from user to authenticate it. Respond in the user's language but write the text in English letters (transliteration). Do not use the native script."
            )

        ]

    def initialize_agent(self):
        """Initialize the agent with tools and prompt."""
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",cache=self.cache)

        self.agent = initialize_agent(llm=model, tools=self.tools, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True,memory=self.memory, handle_parsing_errors=True)


    def process_query(self, user_query: str):
        """Process a user query using the agent executor."""
        try:
            # Access chat memory directly
            response = self.agent.run(user_query)

            # Check if the order ID is verified
            if "Order ID found" in response:
                self.order_verified = True

            return response

        except ResourceExhausted as e:
            return f"Quota exceeded: {e}"

    def process_audio_input(self, audio_file: str):
        try:
            print("Starting audio processing...")
            start_time = time.time()
            result = self.transcriber.transcribe_and_translate(audio_file)
            print(f"Transcription result: {result}")
            print(f"Transcription time: {time.time() - start_time:.2f} seconds")
            return result
        except Exception as e:
            print(f"Error in process_audio_input: {e}")
            raise RuntimeError(f"Error processing audio: {e}")

    def run_interactive_mode(self):
        recorded_queries = []

        while True:
            choice = input("\nChoose an option: Record Query [R/r], Provide File [F/f], Type Query [T/t], or Exit [E/e]: ").lower().strip()

            if choice in ["e", "exit"]:
                print("Thank you!")
                break

            try:
                query = None
                if choice in ["r", "record"]:
                    print("Recording audio...")
                    try:
                        audio_file = record_audio()
                        print(f"Audio file created: {audio_file}")
                        result = self.process_audio_input(audio_file)
                        if result and result['transcription']:
                            query = result['transcription']
                            print(f"\nTranscribed Query: {query}")
                        else:
                            print("No transcription received")
                            continue

                    except Exception as e:
                        print(f"Recording error: {e}")
                        continue

                elif choice in ["f", "file"]:
                    audio_file = input("Enter audio file path: ").strip()
                    if not os.path.exists(audio_file):
                        print("File not found. Please provide a valid path.")
                        continue
                    result = self.process_audio_input(audio_file)
                    if result and result['transcription']:
                        query = result['transcription']
                        print(f"\nTranscribed Query: {query}")
                    else:
                        print("No transcription received")
                        continue

                elif choice in ["t", "type"]:
                    query = input("Type your query (including Order ID if applicable): ").strip()
                    if not query:
                        print("Query cannot be empty. Please try again.")
                        continue
                else:
                    print("Invalid choice. Please try again.")
                    continue

                # Process the query (whether from audio or text)
                if query:
                    print(f"Processing query: {query}")
                    recorded_queries.append(query)
                    try:
                        response = self.process_query(query)
                        print(f"\nResponse: {response}")
                    except Exception as e:
                        print(f"\nError in query processing: {e}")
                else:
                    print("No valid query to process. Please try again.")

            except Exception as e:
                print(f"\nError in run_interactive_mode: {e}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()

        return recorded_queries
# Example usage:
api = google_api_key
ai = CallCenterAI(google_api_key=api, data_path="D:\D4X\Ecommerce_FAQs.csv", crm_path="D:\D4X\CRM.csv", persist_directory="data")
ai.setup_qa_chain()
ai.initialize_tools()
ai.initialize_agent()
ai.run_interactive_mode()

# ques = ["my order id is ORD007", "i ordered a shirt and it was unstitched i want replacement","my order id is ORD007"]