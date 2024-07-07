import inspect
import streamlit as st

class ShowCode:
    def __init__(self):
        self.data = None
        self.target_class = None

    def set_target_class(self, target_class):
        self.target_class = target_class

    def _generate_code(self, func_name, *args, **kwargs):
        """Generate code for the given function with its arguments."""
        if self.target_class is None:
            raise ValueError("Target class not set. Use set_target_class() method.")
        
        func = getattr(self.target_class, func_name)
        source = inspect.getsource(func)
        
        # Remove the decorator and function definition
        code_lines = source.split('\n')[1:]
        code = '\n'.join(code_lines)
        
        # Replace self with the class name
        code = code.replace('self.', f'{self.target_class.__name__}.')
        
        # Add function call with actual arguments
        arg_strings = [repr(arg) for arg in args]
        kwarg_strings = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        all_args = arg_strings + kwarg_strings
        func_call = f"{self.target_class.__name__}.{func_name}(data, {', '.join(all_args)})"
        
        return f"import pandas as pd\nimport plotly.express as px\nimport streamlit as st\n\ndata = pd.read_csv('your_data.csv')  # Replace with your data loading method\n\n{code}\n\n{func_call}"

    def _display_code(self, func_name, *args, **kwargs):
        """Display the code for the given function."""
        code = self._generate_code(func_name, *args, **kwargs)
        if st.button("See Code"):
            st.code(code, language='python')