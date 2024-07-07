import inspect
import streamlit as st

class ShowCode:
    def __init__(self):
        self.target_class = None
    
    def set_target_class(self, target_class):
        self.target_class = target_class
    
    def _generate_code(self, func_name):
        if self.target_class is None:
            raise ValueError("Target class not set. Use set_target_class() method.")
        
        func = getattr(self.target_class, func_name)
        source = inspect.getsource(func)
        
        return source
    
    def _display_code(self, func_name):
        code = self._generate_code(func_name)
        st.code(code, language='python')