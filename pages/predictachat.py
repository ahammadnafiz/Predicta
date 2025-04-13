import streamlit as st
import io
import time
import pandas as pd
import os
import shutil
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
# Add model_rebuild call to fix Pydantic model configuration
ChatGroq.model_rebuild()
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from Theme import theme

class ChatPredicta:
    def __init__(self, df, groq_api_key):
        self.df = df
        self.groq_api_key = groq_api_key
        self.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
        self.vector_store = None
        self.vector_store_path = "faiss_index"

        # Initialize Groq Langchain chat object
        self.groq_chat = ChatGroq(groq_api_key=self.groq_api_key, model_name='llama3-8b-8192')

        # Create vector store from dataframe
        self.create_vector_store()

        # Construct a chat prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="As an advanced virtual assistant, you excel in data analysis and code generation, providing comprehensive insights and actionable recommendations based on CSV data. \
                    You are a code wizard, crafting starter code templates for users' projects, giving them a head start in their coding endeavors. \
                    Use the relevant information from the vector database to provide more accurate answers."),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        # Create a conversation chain
        self.conversation = LLMChain(
            llm=self.groq_chat,
            prompt=self.prompt,
            verbose=False,
            memory=self.memory,
        )
    
    def create_vector_store(self):
        """Create a FAISS vector store from the dataframe"""
        # Save dataframe to a temporary CSV file
        temp_csv_path = "temp_data.csv"
        self.df.to_csv(temp_csv_path, index=False)
        
        # Load the CSV data using Langchain's CSVLoader
        loader = CSVLoader(file_path=temp_csv_path)
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create vector store
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Save vector store to disk
        self.vector_store.save_local(self.vector_store_path)
        
        # Clean up temporary CSV file
        os.remove(temp_csv_path)
    
    def clear_vector_store(self):
        """Clear the vector store from disk"""
        if os.path.exists(self.vector_store_path):
            shutil.rmtree(self.vector_store_path)
            return True
        return False
    
    def search_vector_store(self, query, k=3):
        """Search the vector store for relevant documents"""
        if self.vector_store:
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        return []

    def display_dataset(self):
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>", unsafe_allow_html=True)
        st.dataframe(self.df, width=800)

    def display_question_input(self):
        st.markdown("<h2 style='text-align: center; font-size: 25px;'>Ask a question about the data</h2>", unsafe_allow_html=True)
        question = st.chat_input(placeholder="Type your question here...")
        return question

    def get_user_message(self, question):
        # Search for relevant context in vector store
        relevant_docs = self.search_vector_store(question)
        relevant_context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
        
        user_message = {
            "role": "user",
            "content": f"**Relevant Context from Vector DB**:\n{relevant_context}\n\n**Question**: {question}"
        }
        return user_message

    def send_message_to_groq(self, user_message):
        response = self.conversation.predict(human_input=user_message['content'])
        return response

    def chat_with_predicta(self):
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Chat with Predicta</h1>", unsafe_allow_html=True)
        st.markdown("---")

        self.display_dataset()

        question = self.display_question_input()

        if question and not self.groq_api_key:
            st.info("Please add your Groq API key to continue.")

        if question and self.groq_api_key:
            user_message = self.get_user_message(question)

            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
            
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)
            response = self.send_message_to_groq(user_message)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write("Thinking...")
                time.sleep(0.5)
                st.write(response)

def main():
    # Initialize app state
    if "vector_store_created" not in st.session_state:
        st.session_state.vector_store_created = False
    if "chat_predicta" not in st.session_state:
        st.session_state.chat_predicta = None
    if "previous_file_hash" not in st.session_state:
        st.session_state.previous_file_hash = None
    if "uploaded_file" not in st.session_state:  # Store uploaded file in session
        st.session_state.uploaded_file = None

    st.set_page_config(layout="centered", page_icon="🤖", page_title="ChatPredicta")
    theme.init_styling()
    st.image("assets/Hero.png")
    st.markdown("---")
    st.sidebar.title("PredictaChat App")
    st.sidebar.markdown("---")
    
    # File uploader that updates session state
    new_upload = st.file_uploader("Choose a CSV file", type="csv")
    if new_upload:
        st.session_state.uploaded_file = new_upload
        
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
    
    # Add buttons for clearing history and vector store
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear History"):
            st.session_state["messages"] = []
    
    with col2:
        if st.button("Clear Vector DB"):
            if st.session_state.chat_predicta:
                if st.session_state.chat_predicta.clear_vector_store():
                    st.session_state.vector_store_created = False
                    st.session_state.chat_predicta = None
                    st.success("Vector database cleared!")
                    # Preserve other session state elements
                    st.rerun()
    
    st.sidebar.markdown("---")
    theme.contributor_info()
    
    if not groq_api_key:
        st.info('Please add your Groq API key first')
        return

    if st.session_state.uploaded_file and groq_api_key is not None:
        # Check if the uploaded file is new or different
        current_file_hash = hash(st.session_state.uploaded_file.getvalue())
        if st.session_state.previous_file_hash != current_file_hash:
            # File has changed, clear existing data
            if st.session_state.chat_predicta:
                st.session_state.chat_predicta.clear_vector_store()
                st.session_state.vector_store_created = False
                st.session_state.chat_predicta = None
            st.session_state.previous_file_hash = current_file_hash

        try:
            df = pd.read_csv(st.session_state.uploaded_file)
        except Exception as e:
            st.info(f"Please upload a valid CSV file. {str(e)}")
            theme.show_footer()
            return
        
        # Create new ChatPredicta instance if necessary
        if not st.session_state.vector_store_created or st.session_state.chat_predicta is None:
            try:
                st.session_state.chat_predicta = ChatPredicta(df, groq_api_key)
                st.session_state.vector_store_created = True
            except Exception as e:
                st.error(f"Error initializing Predicta: {str(e)}")
                return
        else:
            # Update API key if it changed
            st.session_state.chat_predicta.groq_api_key = groq_api_key
        
        st.session_state.chat_predicta.chat_with_predicta()
    else:
        st.markdown(
            "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to Chat.</div>",
            unsafe_allow_html=True,
        )
        st.image("assets/uploadfile.png", width=None)
        theme.show_footer()

# Register auto-cleanup function when app exits
def handle_app_exit():
    if st.session_state.chat_predicta:
        st.session_state.chat_predicta.clear_vector_store()

# Run the main function
if __name__ == "__main__":
    try:
        main()
    finally:
        # This will be called when the app is closed/restarted
        if "chat_predicta" in st.session_state and st.session_state.chat_predicta:
            st.session_state.chat_predicta.clear_vector_store()