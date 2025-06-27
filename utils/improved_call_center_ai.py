# import os
# import time
# import pandas as pd
# from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.schema import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.tools import Tool
# from langchain.agents import initialize_agent, AgentType
# from google.api_core.exceptions import ResourceExhausted
# from langchain.memory import ConversationBufferWindowMemory
# from langchain_core.caches import InMemoryCache
# import re
# from dotenv import load_dotenv
# from datetime import datetime
# import json

import os
import time
import pandas as pd
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

# LangChain Core (now separate)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.caches import InMemoryCache
from langchain_core.documents import Document  # updated Document path

# LangChain Community Components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# LangChain Chains (now restructured)
from langchain.chains.retrieval_qa.base import RetrievalQA  # if using RetrievalQA directly
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# LangChain Google GenAI (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain Agents
from langchain.agents import initialize_agent, AgentType

# LangChain Tools
from langchain.tools import Tool

# LangChain Memory
from langchain.memory import ConversationBufferWindowMemory

# Text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

class CallCenterAI:
    def __init__(self, google_api_key, data_path, crm_path, persist_directory, 
                 operations_log_path="operations_log.csv", 
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2", 
                 memory_limit=10):
        
        os.environ['GOOGLE_API_KEY'] = google_api_key
        self.data_path = data_path
        self.crm_path = crm_path
        self.persist_directory = persist_directory
        self.operations_log_path = operations_log_path
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vector_db = None
        self.qa_chain = None
        self.tools = []
        self.agent = None
        self.cache = InMemoryCache()
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", 
            return_messages=True, 
            k=memory_limit
        )
        
        # User session state
        self.current_user = {
            'order_id': None,
            'verified': False,
            'customer_data': None
        }
        
        # Initialize operations log
        self._initialize_operations_log()

    def _initialize_operations_log(self):
        """Initialize the operations log CSV file if it doesn't exist."""
        if not os.path.exists(self.operations_log_path):
            df = pd.DataFrame(columns=[
                'timestamp', 'order_id', 'customer_name', 'operation_type', 
                'reason', 'status', 'agent_notes'
            ])
            df.to_csv(self.operations_log_path, index=False)

    def load_faq_data(self):
        """Load FAQ data from a CSV file."""
        try:
            df = pd.read_csv(self.data_path, encoding='windows-1252')
            if 'Question' in df.columns and 'Answer' in df.columns:
                df['qa'] = df['Question'] + " " + df['Answer']
                if 'Category' in df.columns:
                    df['qa'] += " " + df['Category']
                return df['qa'].tolist()
            else:
                print("Warning: Expected columns 'Question' and 'Answer' not found in FAQ data")
                return []
        except Exception as e:
            print(f"Error loading FAQ data: {e}")
            return []

    def convert_to_documents(self, data):
        """Convert data into LangChain-compatible Document objects."""
        return [Document(page_content=content, metadata={}) for content in data]

    def split_text_into_chunks(self, documents, chunk_size=500, chunk_overlap=50):
        """Split documents into smaller chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)

    def download_hf_embeddings(self):
        """Download and load HuggingFace embedding model."""
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

    def create_vector_db(self, chunks):
        """Create and persist the vector database."""
        self.vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            persist_directory=self.persist_directory
        )
        self.vector_db.persist()

    def setup_vector_database(self):
        """Setup the complete vector database from FAQ data."""
        print("Setting up vector database...")
        faq_data = self.load_faq_data()
        if not faq_data:
            print("No FAQ data found. Using empty database.")
            faq_data = ["Default FAQ: Please contact support for assistance."]
        
        documents = self.convert_to_documents(faq_data)
        chunks = self.split_text_into_chunks(documents)
        self.download_hf_embeddings()
        self.create_vector_db(chunks)
        print("Vector database setup complete.")

    def setup_qa_chain(self):
        """Setup the QA retrieval chain."""
        system_prompt = (
            "You are a highly intelligent, empathetic, and professional call center agent for an e-commerce company. "
            "Your role is to assist customers with their queries, authenticate them when needed, and help with operations like refunds and replacements. "
            
            "IMPORTANT GUIDELINES: "
            "1. Always be polite, professional, and empathetic "
            "2. Listen to the customer's problem first before asking for any information "
            "3. For operations like refund/replacement, you MUST authenticate the customer first by asking for their Order ID "
            "4. Keep responses concise and solution-oriented "
            "5. Use the provided context to answer FAQ-related questions "
            "6. If you don't have specific information, guide the customer appropriately "
            "7. Dont assume anything like what customer want listen to them first"

            "AUTHENTICATION PROCESS: "
            "- Before processing any refund, replacement, or sensitive operation, ask for the Order ID "
            "- Use the authentication tool to verify the Order ID "
            "- Only proceed with operations after successful authentication "
            
            "Context from FAQ database: {context} "
            
            "Always maintain a helpful and professional tone throughout the conversation."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # Load existing vector database or create new one
        self.download_hf_embeddings()
        if os.path.exists(self.persist_directory):
            self.vector_db = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )
        else:
            self.setup_vector_database()

        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", cache=self.cache)
        question_answer_chain = create_stuff_documents_chain(model, prompt)
        self.qa_chain = create_retrieval_chain(
            self.vector_db.as_retriever(search_kwargs={'k': 3}), 
            question_answer_chain
        )

    def query_faq(self, user_query: str) -> str:
        """Handle user queries using the QA chain."""
        try:
            result = self.qa_chain.invoke({"input": user_query})
            return result["answer"]
        except Exception as e:
            return f"I apologize, but I'm having trouble accessing the FAQ information right now. Could you please rephrase your question or contact our support team directly?"

    def authenticate_user(self, user_query: str) -> str:
        """Authenticate user by checking Order ID against CRM data."""
        try:
            crm_data = self.load_crm_data()
            if crm_data is None or crm_data.empty:
                return "I'm sorry, but I'm unable to access the customer database right now. Please try again later."

            # Extract order ID from query using multiple patterns
            order_patterns = [
                r'(?:order\s*(?:id|number)?\s*(?:is|:)?\s*)?([a-zA-Z]{2,4}\d{3,6})',
                r'(?:ord|order)\s*(\d{3,6})',
                r'([a-zA-Z]{2,4}\d{3,6})'
            ]
            
            order_id = None
            for pattern in order_patterns:
                match = re.search(pattern, user_query, re.IGNORECASE)
                if match:
                    order_id = match.group(1).upper()
                    break
            
            if not order_id:
                return "I couldn't find an Order ID in your message. Please provide your Order ID in the format like 'ORD123' or 'ORDER123'."

            # Check if order exists in CRM
            if 'Order_ID' in crm_data.columns:
                crm_order_ids = crm_data['Order_ID'].astype(str).str.upper()
                if order_id in crm_order_ids.values:
                    # Get customer data
                    customer_row = crm_data[crm_order_ids == order_id].iloc[0]
                    self.current_user = {
                        'order_id': order_id,
                        'verified': True,
                        'customer_data': customer_row.to_dict()
                    }
                    customer_name = customer_row.get('Customer_Name', 'Valued Customer')
                    return f"Thank you! I've successfully verified your Order ID {order_id}. How can I assist you today, {customer_name}?"
                else:
                    self.current_user['verified'] = False
                    return f"I'm sorry, but I couldn't find Order ID {order_id} in our system. Please check the Order ID and try again, or contact our support team if you believe this is an error."
            else:
                return "I'm having trouble accessing the order database. Please contact our support team directly."
                
        except Exception as e:
            print(f"Authentication error: {e}")
            return "I'm experiencing technical difficulties with authentication. Please try again or contact our support team."

    def process_refund_request(self, user_query: str) -> str:
        """Process refund request after authentication."""
        if not self.current_user['verified']:
            return "I need to verify your identity first. Please provide your Order ID to proceed with the refund request."
        
        
        # Log the operation
        self._log_operation(
            order_id=self.current_user['order_id'],
            customer_name=self.current_user['customer_data'].get('Customer_Name', 'Unknown'),
            operation_type='refund',
            reason=user_query,
            status='initiated'
        )
        
        return f"I've initiated your refund request for Order ID {self.current_user['order_id']}. Your refund will be processed within 5-7 business days. You'll receive a confirmation email shortly with the refund details. Is there anything else I can help you with?"

    def process_replacement_request(self, user_query: str) -> str:
        """Process replacement request after authentication."""
        
        if not self.current_user['verified']:
            return "I need to verify your identity first. Please provide your Order ID to proceed with the replacement request."
        
        # Log the operation
        self._log_operation(
            order_id=self.current_user['order_id'],
            customer_name=self.current_user['customer_data'].get('Customer_Name', 'Unknown'),
            operation_type='replacement',
            reason=user_query,
            status='initiated'
        )
        
        return f"I've initiated your replacement request for Order ID {self.current_user['order_id']}. We'll arrange for a replacement to be sent to you within 3-5 business days. You'll receive tracking information once the replacement is dispatched. Is there anything else I can help you with?"

    def _log_operation(self, order_id: str, customer_name: str, operation_type: str, reason: str, status: str):
        """Log operation to CSV file."""
        try:
            # Read existing log
            if os.path.exists(self.operations_log_path):
                df = pd.read_csv(self.operations_log_path)
            else:
                df = pd.DataFrame(columns=[
                    'timestamp', 'order_id', 'customer_name', 'operation_type', 
                    'reason', 'status', 'agent_notes'
                ])
            
            # Add new entry
            new_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'order_id': order_id,
                'customer_name': customer_name,
                'operation_type': operation_type,
                'reason': reason,
                'status': status,
                'agent_notes': f"Processed via AI agent"
            }
            
            new_df = pd.DataFrame([new_entry])
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(self.operations_log_path, index=False)
            
        except Exception as e:
            print(f"Error logging operation: {e}")

    def load_crm_data(self):
        """Load CRM data from CSV file."""
        try:
            return pd.read_csv(self.crm_path)
        except Exception as e:
            print(f"Error loading CRM data: {e}")
            return None

    def initialize_tools(self):
        """Initialize tools for the agent."""
        self.tools = [
            Tool(
                name="Answer_FAQ",
                func=self.query_faq,
                description="Use this tool to answer general FAQ questions about products, services, policies, etc. This should be your primary tool for answering customer questions."
            ),
            Tool(
                name="Authenticate_User",
                func=self.authenticate_user,
                description="Use this tool when customer provides an Order ID or when you need to verify customer identity before processing refunds, replacements, or other sensitive operations."
            ),
            Tool(
                name="Process_Refund",
                func=self.process_refund_request,
                description="Use this tool to process refund requests. Only use after customer has been authenticated with their Order ID. Provide details of what and why need refund request in the user query."
            ),
            Tool(
                name="Process_Replacement",
                func=self.process_replacement_request,
                description="Use this tool to process replacement requests. Only use after customer has been authenticated with their Order ID. Provide details of the what and why need replacement request in the user query."
            )
        ]

    def initialize_agent(self):
        """Initialize the conversational agent."""
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", cache=self.cache)
        
        self.agent = initialize_agent(
            llm=model,
            tools=self.tools,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=3
        )

    def process_query(self, user_query: str) -> str:
        """Process user query through the agent."""
        try:
            response = self.agent.run(user_query)
            return response
        except ResourceExhausted as e:
            return "I'm experiencing high traffic right now. Please try again in a moment."
        except Exception as e:
            print(f"Error processing query: {e}")
            return "I apologize, but I'm having technical difficulties. Please try rephrasing your question or contact our support team directly."

    def reset_user_session(self):
        """Reset user session data."""
        self.current_user = {
            'order_id': None,
            'verified': False,
            'customer_data': None
        }
        
    def get_operations_log(self):
        """Get the operations log as DataFrame."""
        try:
            if os.path.exists(self.operations_log_path):
                return pd.read_csv(self.operations_log_path)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading operations log: {e}")
            return pd.DataFrame()