import streamlit as st
import anthropic
import io
import time

class ChatPredicta:
    def __init__(self, df, anthropic_api_key):
        self.df = df
        self.anthropic_api_key = anthropic_api_key

    def display_dataset(self):
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>", unsafe_allow_html=True)
        st.dataframe(self.df, width=800)

    def display_question_input(self):
        st.markdown("<h2 style='text-align: center; font-size: 25px;'>Ask a question about the data</h2>", unsafe_allow_html=True)
        question = st.chat_input(placeholder="Type your question here...")
        return question

    def get_user_message(self, question):
        csv_data = self.df.sample(300).to_csv(index=False)
        csv_file = io.StringIO(csv_data)
        final_csv = csv_file.read()
        
        user_message = {
            "role": "user",
            "content": f"**CSV Data**:\n```\n{final_csv}\n```\n\n**Question**: {question}"
        }
        return user_message

    def send_message_to_predicta(self, user_message):
        client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0,
                messages=[user_message],
                system="As an advanced virtual assistant, you excel in data analysis and code generation, providing comprehensive insights and actionable recommendations based on CSV data. \
                    You are a code wizard, crafting starter code templates for users' projects, giving them a head start in their coding endeavors. \
                    Drawing from your expertise in machine learning algorithms, you guide users in selecting models that best suit their problems and data.\
                    Whether it's classification or regression, supervised or unsupervised learning, you offer insights into the most effective approaches and why they are suitable for their specific scenarios. \
                    Specializing in data evaluation and preprocessing, you assist users in preparing their datasets for machine learning models. \
                    From data cleaning to augmentation, you provide suggestions to optimize their datasets for accurate analysis. \
                    Furthermore, you are an expert in understanding and defining machine learning problems. Your aim is to extract a clear, \
                    concise problem statement from users' input, ensuring their projects begin with a solid foundation for success."
        )

        return message

    def chat_with_predicta(self):
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Chat with Predicta</h1>", unsafe_allow_html=True)
        st.markdown("---")

        self.display_dataset()

        question = self.display_question_input()

        if question and not self.anthropic_api_key:
            st.info("Please add your Anthropic API key to continue.")

        if question and self.anthropic_api_key:
            user_message = self.get_user_message(question)
            # message = self.send_message_to_predicta(user_message)
            
            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
            
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)
            response = self.send_message_to_predicta(user_message)
            st.session_state.messages.append({"role": "assistant", "content": response.content[0].text})
            with st.chat_message("assistant"):
                st.write("Thinking...")
                time.sleep(0.5)
                st.write(response.content[0].text)
