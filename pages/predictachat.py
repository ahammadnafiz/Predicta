import streamlit as st
import io
import time
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from Theme import theme

class ChatPredicta:
    def __init__(self, df, groq_api_key):

        self.df = df
        self.groq_api_key = groq_api_key
        self.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

        # Initialize Groq Langchain chat object
        self.groq_chat = ChatGroq(groq_api_key=self.groq_api_key, model_name='llama3-8b-8192')

        # Construct a chat prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="As an advanced virtual assistant, you excel in data analysis and code generation, providing comprehensive insights and actionable recommendations based on CSV data. \
                    You are a code wizard, crafting starter code templates for users' projects, giving them a head start in their coding endeavors."),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        # Create a conversation chain
        self.conversation = LLMChain(
            llm=self.groq_chat,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory,
        )
    
    def display_dataset(self):
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>", unsafe_allow_html=True)
        st.dataframe(self.df, width=800)

    def display_question_input(self):
        st.markdown("<h2 style='text-align: center; font-size: 25px;'>Ask a question about the data</h2>", unsafe_allow_html=True)
        question = st.chat_input(placeholder="Type your question here...")
        return question

    def get_user_message(self, question):
        csv_data = self.df.sample(20).to_csv(index=False)
        csv_file = io.StringIO(csv_data)
        final_csv = csv_file.read()
        
        user_message = {
            "role": "user",
            "content": f"**CSV Data**:\n```\n{final_csv}\n```\n\n**Question**: {question}"
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
    
    st.set_page_config(layout="centered", page_icon="🤖", page_title="ChatPredicta")
    theme.init_styling()
    st.image("assets/Hero.png")
    st.markdown("---")
    st.sidebar.title("PredictaChat App")
    st.sidebar.markdown("---")
        
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
    
    if st.sidebar.button("Clear History"):
        st.session_state["messages"] = []
    
    st.sidebar.markdown("---")
    theme.contributor_info()
    
    
    if not groq_api_key:
        st.info('Please add your api key first')
        return

    if uploaded_file and groq_api_key is not None:
        df = pd.read_csv(uploaded_file)
        chat_predicta = ChatPredicta(df, groq_api_key)
        chat_predicta.chat_with_predicta()
    else:
        st.markdown(
            "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to Chat.</div>",
            unsafe_allow_html=True,
        )
        st.image("assets/uploadfile.png", use_container_width=True)
        theme.show_footer()
    

if __name__ == "__main__":
    main()