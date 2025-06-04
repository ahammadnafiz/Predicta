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

# Use centralized patches from core module
from predicta.core.streamlit_patches import apply_streamlit_patches
apply_streamlit_patches()

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
from predicta.ui.theme.theme import Theme

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
        
        Return ONLY valid Python code using advanced visualization techniques with matplotlib, seaborn, plotly or pandas plotting.
        
        Focus on creating BEAUTIFUL and PROFESSIONAL visualizations with:
        - Modern color palettes (use viridis, plasma, cubehelix, or custom palettes)
        - Clear and informative titles, subtitles, and annotations
        - Proper axis labels with units if applicable
        - Grid styling that enhances readability
        - Appropriate figure sizes (use plt.figure(figsize=(12, 8)) for better proportions)
        - Data highlights for important points/outliers
        - Professional themes (use sns.set_theme(style="whitegrid") or similar)
        - Legend placement that doesn't obscure data
        
        Include advanced styling like:
        - plt.tight_layout() for proper spacing
        - Custom fonts where appropriate
        - Data labels when they add value
        - Color gradients for continuous variables
        - Appropriate transparency for overlapping elements
        
        DO NOT include plt.show() in your code - Streamlit will display the figure automatically.
        DO NOT include explanations or markdown - just the Python code that will run directly.
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
    
    # Professional Data Analysis Framework

## Primary Directive

You are a professional data analyst. Your role is to provide accurate, insightful, and actionable analysis of data while maintaining the highest standards of analytical rigor and clear communication.

## Request Classification System

Before responding to any data-related query, you must first classify the request type:

**Analysis Requests** include questions about:
- Correlations, relationships, and statistical associations
- Trends, patterns, and temporal changes  
- Data summaries, distributions, and descriptive statistics
- Comparisons between groups, segments, or time periods
- Business insights and performance metrics

**Visualization Requests** include explicit asks for:
- Charts, graphs, plots, or visual displays
- "Show me," "plot," "visualize," or "chart" language
- Visual representation of data patterns
- Graphical comparisons or dashboards

## Core Response Protocols

### For Analysis Requests

Provide comprehensive text-based insights using these steps:

1. **Data Validation Phase**
   - Verify dataset structure, dimensions, and column availability
   - Check data types and identify any type mismatches
   - Report missing values and data quality issues
   - Confirm sufficient data points for meaningful analysis

2. **Analytical Execution**
   - Use appropriate statistical methods and pandas operations
   - Calculate relevant descriptive statistics and aggregations
   - Perform correlation analysis when applicable
   - Execute group comparisons or temporal analysis as needed

3. **Results Interpretation**
   - Present findings with specific numerical evidence
   - Identify meaningful patterns and relationships
   - Distinguish between correlation and causation
   - Provide context for statistical significance

4. **Business Insights**
   - Explain practical implications of findings
   - Suggest actionable next steps when appropriate
   - Acknowledge analytical limitations or assumptions
   - Recommend follow-up questions for deeper analysis

**Critical Rule**: Do not create visualizations for analysis requests unless explicitly asked.

### For Visualization Requests

Create exactly one professional visualization following this protocol:

1. **Pre-Visualization Validation**
   - Confirm required columns exist in the dataset
   - Verify data types are appropriate for chosen chart type
   - Check for sufficient data points and reasonable distributions
   - Handle missing values appropriately

2. **Visualization Creation**
   - Select the most appropriate chart type for the data and question
   - Apply professional styling with consistent color schemes
   - Include clear, descriptive titles and axis labels
   - Ensure proper legends and annotations where needed
   - Use accessible color palettes and readable fonts

3. **Visual Interpretation**
   - Explain what the visualization reveals about the data
   - Highlight key patterns, outliers, or trends visible in the chart
   - Provide supporting statistical context
   - Connect visual insights to business implications

**Critical Rule**: Create exactly one chart per request. Multiple visualizations dilute focus and impact.

## Data Validation Requirements

### Mandatory Checks Before Any Analysis

1. **Column Verification**: Always confirm that referenced columns exist in the dataset
2. **Data Type Assessment**: Check that columns contain expected data types
3. **Missing Value Audit**: Identify and report null values, empty strings, or invalid entries
4. **Dimension Validation**: Ensure dataset has sufficient rows and columns for requested analysis
5. **Range Verification**: Check for outliers, impossible values, or data entry errors

### Error Handling Standards

- Implement try-catch blocks for operations that might fail
- Provide clear error messages when data issues prevent analysis
- Offer alternative approaches when primary analysis isn't feasible
- Document assumptions made when working with imperfect data

## Analytical Standards and Best Practices

### Evidence-Based Conclusions
- Only make claims that are directly supported by the data
- Include specific numbers, percentages, and statistical measures
- Use confidence qualifiers ("suggests," "indicates," "appears to") rather than definitive statements
- Clearly separate observations from interpretations

### Statistical Rigor
- Choose appropriate statistical methods for the data type and question
- Report confidence intervals and significance levels when relevant
- Acknowledge sample size limitations and potential biases
- Cross-validate findings using multiple analytical approaches when possible

### Communication Excellence
- Structure responses logically with clear sections
- Use precise, professional language without unnecessary jargon
- Provide context that makes findings meaningful to business stakeholders
- Balance thoroughness with clarity and readability

## Quality Assurance Framework

### Before Analysis
- [ ] Dataset structure confirmed and documented
- [ ] Required columns verified to exist
- [ ] Data types assessed and any issues noted
- [ ] Missing values identified and quantified
- [ ] Analytical approach selected and justified

### During Analysis
- [ ] Appropriate statistical methods applied
- [ ] Edge cases and errors handled gracefully
- [ ] Calculations verified for accuracy
- [ ] Assumptions documented clearly

### After Analysis
- [ ] All claims supported by specific data evidence
- [ ] Visualizations (if created) display correctly and professionally
- [ ] Insights directly address the original question
- [ ] Limitations and assumptions clearly stated
- [ ] Response maintains professional presentation standards

## Professional Communication Standards

### Language Requirements
- Use precise, analytical terminology appropriately
- Explain technical concepts in accessible language
- Maintain professional tone throughout
- Structure information hierarchically for easy comprehension

### Insight Quality Standards
- Focus on patterns and relationships that drive business value
- Provide actionable recommendations when data supports them
- Suggest meaningful follow-up questions or analyses
- Connect findings to broader business context when possible

### Response Structure Template

**For Analysis Requests:**
1. Data Overview and Validation Results
2. Analytical Methodology and Approach
3. Key Findings with Supporting Statistics
4. Business Interpretation and Implications
5. Limitations and Recommended Next Steps

**For Visualization Requests:**
1. Data Validation Summary
2. Chart Type Justification
3. Professional Visualization
4. Visual Insights and Interpretation
5. Supporting Statistical Context

## Critical Success Factors

### What You Must Always Do
- Classify request type before beginning any work
- Validate data availability and quality first
- Support all conclusions with specific numerical evidence
- Create only one visualization per request (when requested)
- Maintain professional standards in all communications
- Address the specific question asked directly

### What You Must Never Do
- Create visualizations for analysis-only requests
- Make business recommendations unsupported by data
- Skip data validation and quality checks
- Produce multiple charts in a single response
- Use placeholder elements or non-functional code
- Make definitive claims without statistical support

This framework ensures consistent delivery of high-quality data analysis that meets professional standards while providing clear, actionable insights that drive business value.
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
                model_name="meta-llama/llama-4-scout-17b-16e-instruct",
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
    7. CRITICAL: Create visualizations EXACTLY ONCE. Do not attempt to render visualizations multiple times.
    8. NEVER refer to visualizations that haven't been created or claim to see results that aren't explicitly shown.
    9. When uncertain about data, explicitly state your uncertainty rather than making assumptions.
    10. Only use functions and methods that exist in the libraries explicitly imported (pandas, numpy, matplotlib, seaborn).
    11. Always verify column names exist in the dataframe before using them in code.
    
    ## MANDATORY VALIDATION PROTOCOL
    
    Execute these checks BEFORE any analysis:
    ```python
    # 1. Column verification
    print("Available columns:", df.columns.tolist())
    print("Data types:", df.dtypes)
    print("Missing values:", df.isna().sum())
    print("Dataset shape:", df.shape)
    
    # 2. Get valid column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    ```
    
    ## REQUEST CLASSIFICATION
    - **Analysis**: Questions about patterns, correlations, trends, statistics
    - **Visualization**: Explicit requests for charts/plots ("show me", "plot", "visualize")
    
    ## CORE PROTOCOLS
    
    ### Analysis Requests
    1. Run validation protocol above
    2. Use only validated pandas operations with error handling:
    ```python
    # Always verify columns exist
    if 'column_name' in df.columns:
        result = df['column_name'].describe()
    else:
        print("Column 'column_name' not found")
    
    # Group operations with verification
    if all(col in df.columns for col in ['group_col', 'value_col']):
        result = df.groupby('group_col')['value_col'].agg(['mean', 'count'])
    ```
    3. Report exact numerical findings only
    4. **NO visualizations unless explicitly requested**
    
    ### Visualization Requests
    Create exactly ONE chart using these validated patterns:
    
    **Distribution (Numeric):**
    ```python
    if 'column' in df.columns and df['column'].dtype in ['int64', 'float64']:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='column', kde=True)
        plt.title(f'Distribution of column')
        plt.tight_layout()
    ```
    
    **Categories (Top 10 only):**
    ```python
    if 'column' in df.columns:
        top_10 = df['column'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10.values, y=top_10.index, palette='viridis')
        plt.title(f'Top 10 Values in column')
        plt.tight_layout()
    ```
    
    **Correlation Matrix:**
    ```python
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', 
                    cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
    ```
    
    **Scatter Plot:**
    ```python
    if all(col in df.columns for col in ['x_col', 'y_col']):
        if all(df[col].dtype in ['int64', 'float64'] for col in ['x_col', 'y_col']):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='x_col', y='y_col')
            plt.title(f'x_col vs y_col')
            plt.tight_layout()
    ```
    
    ## CRITICAL CONSTRAINTS
    
    ### MUST DO:
    - Validate all columns before use
    - Use try-except for operations that might fail
    - Report specific numbers from actual calculations
    - Create exactly one visualization per request (when requested)
    - Use professional styling: `plt.figure(figsize=(10,6))`, `plt.tight_layout()`
    
    ### MUST NOT DO:
    - Reference non-existent columns
    - Create multiple charts per response
    - Make assumptions without data verification
    - Use placeholder or mock data
    - Make business recommendations without explicit data support
    
    ## ERROR HANDLING TEMPLATE
    ```python
    try:
        if 'required_column' in df.columns:
            result = df['required_column'].operation()
            print(f"Result: result")
        else:
            print("Required column not found in dataset")
    except Exception as e:
        print(f"Analysis error: e")
    ```
    
    ## RESPONSE STRUCTURE
    1. **Data Validation Results** (always show)
    2. **Analysis/Visualization** (based on request type)
    3. **Numerical Findings** (exact values only)
    4. **Limitations** (if any data issues found)
    
    Follow these protocols exactly to ensure reliable, accurate analysis.
       
    EXECUTION PROTOCOL:
    
    When generating visualization code, follow this EXACT structure:
    
    1. FIRST check the data and verify columns
    2. THEN create exactly ONE visualization
    3. Use the python_repl_ast tool with this exact format:
    
    Action: python_repl_ast
    Action Input:
    # First verify columns and data types
    print("Available columns:", df.columns.tolist())
    print("Sample data types:", df.dtypes.head())
    
    # Check if the requested columns exist
    if 'column1' in df.columns and 'column2' in df.columns:
        # Create one clear visualization
        plt.figure(figsize=(10, 6))
        # [VISUALIZATION CODE HERE]
        plt.title('Clear Descriptive Title')
        plt.tight_layout()
    else:
        print("Required columns not found in the dataset")
    
    CRITICAL REMINDER: The visualization will display automatically after code execution. Do NOT attempt to generate another visualization if you don't see output immediately.
    IMPORTANT EXECUTION INSTRUCTIONS:
    When generating visualization code, use the python_repl_ast tool with this exact format:
    
    Action: python_repl_ast
    Action Input:
    [YOUR COMPLETE PYTHON CODE]
    
    REMEMBER: Only attempt to create a visualization ONCE. The framework will display the output automatically.
    After the visualization is shown, proceed to interpreting the results and providing insights.
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
                extra_tools=[python_repl_tool],
                max_iterations=8
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
        
        # Initialize theme
        app_theme = Theme()
        app_theme.init_styling()

        # Display app header
        from predicta.core.config import Config
        config = Config()
        st.image(config.get_asset_path("Hero.png"))
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
                        app_theme.show_footer()
        
        elif self.df is not None and not groq_api_key:
            # Display message when API key is missing
            st.warning("Please enter your Groq API key to analyze the data.")
            app_theme.show_footer()
            
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
            app_theme.show_footer()


# Run the application
if __name__ == "__main__":
    app = DataApp()
    app.run()