import sys
import logging

# Configure logging to ignore specific PyTorch errors
class TorchErrorFilter(logging.Filter):
    def filter(self, record):
        if "torch._C._get_custom_class_python_wrapper" in str(record.getMessage()):
            return False
        return True

# Apply the log filter to root logger
root_logger = logging.getLogger()
root_logger.addFilter(TorchErrorFilter())

# Monkeypatch Streamlit's module path extraction to handle PyTorch modules
from streamlit.watcher import local_sources_watcher

original_get_module_paths = local_sources_watcher.get_module_paths

def patched_get_module_paths(module):
    try:
        return original_get_module_paths(module)
    except RuntimeError as e:
        if "torch._C._get_custom_class_python_wrapper" in str(e):
            return []
        raise e

# Apply the patch
local_sources_watcher.get_module_paths = patched_get_module_paths

# Suppress asyncio loop warnings
import asyncio
original_get_running_loop = asyncio.get_running_loop

def patched_get_running_loop():
    try:
        return original_get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Apply the patch
asyncio.get_running_loop = patched_get_running_loop

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import re
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import numpy as np

# LangChain Imports
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq
# Add model_rebuild call to fix Pydantic model configuration
ChatGroq.model_rebuild()
from langchain_experimental.tools import PythonAstREPLTool
from Theme import theme


class CodeUtils:
    """Utility class for code extraction and execution"""
    
    @staticmethod
    def extract_code_from_response(response):
        """Extract Python code from LLM response using various patterns"""
        # Pattern 1: Code within python code blocks
        pattern1 = r"```python\n(.*?)```"
        match1 = re.search(pattern1, response, re.DOTALL)
        if match1:
            return match1.group(1).strip()
        
        # Pattern 2: Code from python_repl_ast Action Input
        pattern2 = r"Action: python_repl_ast\nAction Input: (.*?)(?=\n[A-Z]|$)"
        match2 = re.search(pattern2, response, re.DOTALL)
        if match2:
            return match2.group(1).strip()
        
        # Pattern 3: Just look for the typical pandas/matplotlib patterns
        if "df[" in response and (".plot" in response or "plt." in response):
            lines = response.split('\n')
            code_lines = []
            capture = False
            
            for line in lines:
                # Start capturing when we see typical code patterns
                if "df[" in line or "plt." in line or ".plot" in line:
                    capture = True
                
                # Stop capturing when we hit text that looks like a sentence
                if capture and line and line[0].isupper() and "." in line and not any(x in line for x in ["df", "plt", "pd", "np", "import"]):
                    break
                
                if capture and line.strip():
                    code_lines.append(line)
            
            if code_lines:
                return "\n".join(code_lines)
        
        return None
    
    @staticmethod
    def remove_code_from_response(response, code):
        """Remove the executed code from the response text"""
        # Pattern 1: Remove python code blocks
        cleaned = re.sub(r"```python\n.*?```", "", response, flags=re.DOTALL)
        
        # Pattern 2: Remove python_repl_ast sections
        cleaned = re.sub(r"Action: python_repl_ast\nAction Input: .*?(?=\n[A-Z]|$)", "", cleaned, flags=re.DOTALL)
        
        # Remove any double newlines that might be left
        cleaned = re.sub(r"\n\n+", "\n\n", cleaned)
        
        return cleaned.strip()
    
    @staticmethod
    def sanitize_code(code):
        """Clean up code by removing plt.show() calls"""
        if not code:
            return code
        return re.sub(r'plt\.show\(\)', '', code)


class VisualizationHandler:
    """Centralized class to handle all visualization execution"""
    
    @staticmethod
    def get_execution_context(df=None):
        """Get the standard execution context for Python code"""
        context = {
            'plt': plt,
            'np': np,
            'pd': pd,
            'sns': sns,
        }
        
        if df is not None:
            context['df'] = df
            
        return context
    
    @staticmethod
    def execute_visualization_code(code, df=None, display=True):
        """Execute visualization code and optionally display in Streamlit"""
        try:
            # Sanitize code
            code = CodeUtils.sanitize_code(code)
            
            # Create a new figure
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get execution context
            exec_globals = VisualizationHandler.get_execution_context(df)
            exec_globals.update({
                'ax': ax,
                'fig': fig
            })
            
            # Execute the code
            exec(code, exec_globals)
            
            # Add styling improvements
            plt.tight_layout()
            
            # Check if we need to rotate x-axis labels
            if hasattr(ax, 'get_xticklabels') and ax.get_xticklabels():
                longest_label = max([len(str(label.get_text())) for label in ax.get_xticklabels()])
                if longest_label > 5:
                    plt.xticks(rotation=45, ha='right')
            
            # Display the figure if requested
            if display:
                st.pyplot(fig)
                
            return True, "Visualization successfully displayed."
        except Exception as e:
            error_message = f"Error executing visualization code: {str(e)}"
            if display:
                st.error(error_message)
                st.code(code, language="python")
            return False, error_message


class CustomPythonAstREPLTool(PythonAstREPLTool):
    """Custom Python AST REPL Tool that captures and displays matplotlib figures in Streamlit"""
    
    def __init__(self, locals=None):
        super().__init__(locals=locals)
    
    def _run(self, query: str) -> str:
        """Run the query in the Python REPL and capture the result."""
        try:
            # Ensure locals is initialized
            if self.locals is None:
                self.locals = {}
            
            # Execute the code using parent method
            result = super()._run(query)
            
            # Capture the matplotlib figure if one was created
            if plt.get_fignums():
                current_fig = plt.gcf()
                
                # Create a placeholder for the figure and display it
                fig_placeholder = st.empty()
                fig_placeholder.pyplot(current_fig)
                
                # Add a success message to the result
                result += "\n\nVisualization successfully displayed."
            
            return result
        
        except Exception as e:
            error_message = f"Error executing code: {str(e)}"
            st.error(error_message)
            return error_message


class ResponseProcessor:
    """Class to process and execute Python code from agent responses"""
    
    def __init__(self, df):
        self.df = df
    
    def process_response(self, response):
        """Process agent response to execute Python code visualizations."""
        
        # Look for Python code in the response
        python_code = CodeUtils.extract_code_from_response(response)
        
        if python_code:
            try:
                # Execute the Python code
                success, message = VisualizationHandler.execute_visualization_code(python_code, self.df)
                
                # Clean up response by removing the executed code
                cleaned_response = CodeUtils.remove_code_from_response(response, python_code)
                return cleaned_response
            except Exception as e:
                st.error(f"Error executing Python code: {str(e)}")
        
        # No Python code found or execution failed
        return response


class LLMVisualizer:
    """Class to handle data visualization through LLM code generation"""
    
    def __init__(self, df):
        self.df = df
    
    def generate_visualization(self, query, llm):
        """Generate visualization code using LLM based on user query and dataframe columns"""
        # Create a prompt with column information
        columns_info = "\n".join([f"- {col} ({self.df[col].dtype})" for col in self.df.columns])
        
        visualization_prompt = f"""
        Generate Python code to visualize the following query: "{query}"
        
        The dataframe 'df' has the following columns:
        {columns_info}
        
        Return ONLY valid Python code (using matplotlib, seaborn or pandas plotting) that will run directly.
        Include proper labels, titles, and use plt.tight_layout() for better display.
        Do not include explanations or markdown - just the Python code.
        """
        
        try:
            # Get code from LLM
            response = llm.invoke(visualization_prompt)
            # Extract the actual code
            code = CodeUtils.extract_code_from_response(response.content)
            return code
        except Exception as e:
            st.error(f"Error generating visualization code: {str(e)}")
            return None
    
    def execute_visualization(self, code):
        """Execute the generated visualization code"""
        success, message = VisualizationHandler.execute_visualization_code(code, self.df)
        return success


class LLMAgent:
    """Base class for LLM agents with common functionality"""
    
    # Common system template portions
    COMMON_SYSTEM_TEMPLATE = """
    You are a data analysis and visualization expert that helps users analyze CSV data using Python, pandas, and matplotlib.
    
    You have access to a pandas DataFrame named 'df' with the following columns:
    {df_schema}
    
    IMPORTANT INSTRUCTIONS:
    
    1. When asked to create visualizations, ALWAYS generate valid Python code using matplotlib or pandas plotting.
    2. ALWAYS prefix your code with ```python and end with ``` so it can be executed.
    3. Use 'df' as the DataFrame variable name in your code.
    4. Keep your code simple, focused and complete - it will be executed exactly as written.
    5. Include proper labels, titles, and styling in your matplotlib code.
    6. NEVER include plt.show() in your code - Streamlit will display the figure automatically.
    """
    
    def __init__(self, groq_api_key=None):
        self.groq_api_key = groq_api_key
        self.llm = None
    
    def initialize_llm(self):
        """Initialize the LLM with API key"""
        if not self.groq_api_key:
            return False
            
        try:
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.6
            )
            return True
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            return False


class DataAnalysisAgent(LLMAgent):
    """Class to handle LLM agent interactions for data analysis"""
    
    # Extended system template for the data analysis agent
    SYSTEM_TEMPLATE = LLMAgent.COMMON_SYSTEM_TEMPLATE + """
    7. IMPORTANT: Only attempt to create a visualization ONCE. Do not retry if you don't see the output immediately.
       The visualization will be shown to the user automatically after your code executes.
    8. For common plots:
       - For counts/distributions: `df['column'].value_counts().plot(kind='bar')`
       - For pie charts: `df['column'].value_counts().plot(kind='pie', autopct='%1.1f%%')`
       - For time series: `df.plot(x='date_column', y='value_column')`
       - For scatter plots: `df.plot.scatter(x='column1', y='column2')`
       - For heatmaps: `sns.heatmap(df.corr(), annot=True, cmap='coolwarm')`
       - For stacked bars: `pd.crosstab(df['col1'], df['col2']).plot(kind='bar', stacked=True)`
    
    First provide a brief text analysis, then include executable Python code for visualization.
    After executing the visualization, provide a brief interpretation of the results and finalize your answer.
    
    When generating visualization code, use the python_repl_ast tool to execute Python code.
    To use this tool, specify the action as "python_repl_ast" followed by the code to execute.
    For example:
    
    Action: python_repl_ast
    Action Input: 
    **ALWAYS include these standard imports**:
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   ```
   
   Visualization recipe guide:
    Analysis Type | Recommended Code Pattern
    ------------- | -----------------------
    Distribution | sns.histplot(df['column'], kde=True)
    Count/Category | sns.countplot(y=df['column'].sort_values(), color='#3498db')
    Correlation | sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    Time Series | sns.lineplot(x='date_column', y='value_column', data=df, marker='o')
    Comparison | sns.barplot(x='category', y='value', data=df, palette='viridis')
    Relationship | sns.scatterplot(x='column1', y='column2', hue='category', data=df)
    Multi-Variable | sns.pairplot(df[['col1', 'col2', 'col3']], hue='category')
    Grouped Analysis | df.groupby('category')['value'].mean().sort_values(ascending=False).plot(kind='bar')
   
   Create professional visualizations with:

    Descriptive titles, axis labels, and legends
    Appropriate color schemes (use colorblind-friendly palettes)
    Proper figure sizing and layout adjustments
    Data annotations where helpful
    
    Follow this pattern for all visualizations:
    python# Set styling options
    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')

    # Create visualization
    [VISUALIZATION CODE]

    # Add appropriate labels and styling
    plt.title('Descriptive Title', fontsize=14, fontweight='bold')
    plt.xlabel('X-Axis Label', fontsize=12)
    plt.ylabel('Y-Axis Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    Your output will be displayed automatically. After using the python_repl_ast tool ONCE, 
    proceed directly to interpreting the results and providing your final answer.
    
    
    EXECUTION INSTRUCTIONS
    When generating visualization code, use the python_repl_ast tool with this exact format:
    Action: python_repl_ast
    Action Input:
    [YOUR COMPLETE PYTHON CODE]
    IMPORTANT: Only attempt to create a visualization ONCE. The framework will display the output automatically.
"""
    
    def __init__(self, df, response_processor, groq_api_key=None):
        super().__init__(groq_api_key)
        self.df = df
        self.response_processor = response_processor
        self.agent = None
    
    def setup_agent(self, file_path):
        """Set up the CSV agent with Groq LLM."""
        # Create system prompt with dataframe schema
        df_schema = "\n".join([f"- {col} ({self.df[col].dtype})" for col in self.df.columns])
        system_prompt = self.SYSTEM_TEMPLATE.format(df_schema=df_schema)
        
        # Make sure LLM is initialized
        if not self.llm and not self.initialize_llm():
            return None
            
        try:
            # Initialize conversation memory
            memory = ConversationBufferMemory(memory_key="chat_history")
            
            # Create custom Python REPL tool with the dataframe
            python_repl_tool = CustomPythonAstREPLTool(locals=VisualizationHandler.get_execution_context(self.df))
            
            # Create CSV agent with Python REPL tool
            self.agent = create_csv_agent(
                self.llm, 
                file_path, 
                verbose=True, 
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                memory=memory,
                prefix=system_prompt,
                allow_dangerous_code=True,
                extra_tools=[python_repl_tool]
            )
            
            st.success('Ready to analyze your data!')
            return self.agent
        
        except Exception as e:
            st.error(f"Error setting up the agent: {str(e)}")
            return None
    
    def handle_chat_input(self, prompt):
        """Process chat input and handle agent responses."""
        try:
            st_callback = StreamlitCallbackHandler(st.container())
            
            try:
                # First try with the callback
                raw_response = self.agent.run(prompt, callbacks=[st_callback])
            except ValueError as e:
                # Handle parsing errors
                error_msg = str(e)
                st.warning(f"Got an error: {error_msg}")
                
                # Extract the LLM output from the error if possible
                if "Could not parse LLM output:" in error_msg or "Parsing LLM output produced both a final answer and a parse-able action" in error_msg:
                    # Try to extract the actual output from the error message
                    output_start = error_msg.find("`") + 1
                    output_end = error_msg.rfind("`")
                    if output_start > 0 and output_end > output_start:
                        raw_response = error_msg[output_start:output_end]
                        st.info("Extracted response from error message")
                    else:
                        # If extraction fails, try without callbacks
                        raw_response = self.agent.run(prompt)
                else:
                    # For other errors, try without the callback
                    raw_response = self.agent.run(prompt)
            
            # Process response for visualization
            processed_response = self.response_processor.process_response(raw_response)
            
            # Display the processed response
            st.write(processed_response)
            
            return raw_response
        
        except Exception as e:
            error_msg = str(e)
            st.error(f"Error processing your question: {error_msg}")
            
            # Try to extract useful information from the error
            if "agent_scratchpad" in error_msg:
                st.warning("The AI had difficulty processing your request with the available data.")
                st.info("Try asking a simpler question or provide more context.")
            
            return f"I encountered an error processing your request: {error_msg}"
    
    def display_question_input(self):
        """Display the question input field with nice styling"""
        st.markdown("<h2 style='text-align: center; font-size: 25px;'>Ask a question about the data</h2>", unsafe_allow_html=True)
        question = st.chat_input(placeholder="Type your question here...")
        return question


class DataFrameUtils:
    """Utility class for dataframe operations"""
    
    @staticmethod
    def display_dataframe_info(df):
        """Display information about the dataframe."""
        with st.expander("Dataset Details", expanded=False):
            st.dataframe(df.head())
            st.write(f"Shape: {df.shape}")
            
            # Show column information with types and sample values
            st.write("**Column Information:**")
            col_info = []
            for col in df.columns:
                # Get column type
                col_type = df[col].dtype
                
                # Get sample values (first 3 non-null values)
                sample_values = df[col].dropna().head(3).tolist()
                sample_str = ", ".join([str(val) for val in sample_values])
                
                # Check for missing values
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                
                # Get unique values count
                unique_count = df[col].nunique()
                
                col_info.append({
                    "Column": col,
                    "Type": col_type,
                    "Sample Values": sample_str,
                    "Missing": f"{missing} ({missing_pct:.1f}%)",
                    "Unique Values": unique_count
                })
            
            # Display column info as a table
            st.table(pd.DataFrame(col_info))


class DataApp:
    """Main application class that orchestrates all components"""
    
    def __init__(self):
        self.df = None
        self.file_path = None
        self.response_processor = None
        self.analysis_agent = None
        
    def process_uploaded_file(self, file, groq_api_key=None):
        """Process the uploaded CSV file and return dataframe and file path."""
        with st.spinner(text="Loading dataset..."):
            try:
                # Create temporary file
                with NamedTemporaryFile(delete=False) as f:
                    f.write(file.getbuffer())
                    self.file_path = f.name
                
                # Load dataframe with error handling
                try:
                    self.df = pd.read_csv(self.file_path)
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    st.info("Make sure your file is a valid CSV with proper formatting.")
                    return None
                
                # Display dataframe information
                DataFrameUtils.display_dataframe_info(self.df)
                
                # Initialize components
                self.response_processor = ResponseProcessor(self.df)
                self.analysis_agent = DataAnalysisAgent(self.df, self.response_processor, groq_api_key)
                
                return self.df
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return None
    
    def run(self):
        """Run the main application"""
        # Configure Streamlit page
        st.set_page_config(layout="centered", page_icon="🤖", page_title="PredicaVIZ")
        theme.init_styling()

        # Display app header
        st.image("assets/Hero.png")
        st.markdown("---")
        
        # Sidebar
        groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
        
        # File uploader in sidebar
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
        
        # Initialize or reset session state if needed
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Process file if uploaded
        if uploaded_file and (self.df is None or uploaded_file.name != getattr(st.session_state, 'last_file', None)):
            self.process_uploaded_file(uploaded_file, groq_api_key)
            st.session_state.last_file = uploaded_file.name if self.df is not None else None
            
            # Reset chat history when new file is uploaded
            st.session_state.messages = []
        
        # Main app flow
        if self.df is not None and groq_api_key:
            # Setup agent if not already set up
            if self.analysis_agent and self.analysis_agent.agent is None:
                # Update API key if changed
                self.analysis_agent.groq_api_key = groq_api_key
                self.analysis_agent.setup_agent(self.file_path)
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Get user input
            prompt = self.analysis_agent.display_question_input() if self.analysis_agent else None
            
            # Process user input
            if prompt:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    if self.analysis_agent and self.analysis_agent.agent:
                        response = self.analysis_agent.handle_chat_input(prompt)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Agent not available. Please check your API key and try again.")
                        theme.show_footer()
        
        elif self.df is not None and not groq_api_key:
            # Display message when API key is missing
            st.warning("Please enter your Groq API key to analyze the data.")
            theme.show_footer()
            
        elif not uploaded_file:
            # Display welcome message when no file is uploaded
            st.markdown("""
                ## Welcome to PredicaVIZ!
                
                Upload a CSV file and enter your Groq API key to start analyzing your data with AI.
                
                ### Features:
                - **Intelligent Visualizations**: Generate insightful charts and graphs automatically
                - **Natural Language Queries**: Ask questions about your data in plain English
                - **Advanced Data Analysis**: Get deep insights from your data visualized instantly
                - **AI-Powered Exploration**: Let AI find patterns and trends in your data
                
                Get started by uploading a CSV file in the sidebar.
            """)
            theme.show_footer()


# Run the application
if __name__ == "__main__":
    app = DataApp()
    app.run()