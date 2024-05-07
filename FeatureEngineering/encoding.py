import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import streamlit as st

class DataEncoder:
    def __init__(self, data):
        self.data = data
    
    def label_encoding(self, data, col):
        label_encoder = LabelEncoder()
        data[col] = label_encoder.fit_transform(data[col])
        return data
    
    def one_hot_encoding(self, data, col):
        if len(data[col].unique()) > 2:  
            encoder = OneHotEncoder(sparse=False, drop='first')
            encoded_values = encoder.fit_transform(data[[col]])
            new_cols = [col + '_' + str(i) for i in range(encoded_values.shape[1])]
            encoded_df = pd.DataFrame(encoded_values, columns=new_cols, index=data.index)
            data = pd.concat([data, encoded_df], axis=1)
            data.drop(columns=[col], inplace=True)
        return data
    
    def mean_encoding(self, data, col):
        if len(data[col].unique()) > 2:  # Apply mean encoding only if more than 2 unique values
            mean_encoded = data.groupby(col).mean().to_dict()
            data[col] = data[col].map(mean_encoded)
            return data
        else:
            return data
    
    def encoder(self):
        st.markdown(
            "<h1 style='text-align: center; font-size: 30px;'>Encode Data</h1>", 
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h1>", unsafe_allow_html=True)
        st.dataframe(self.data, width=800)

        option = st.sidebar.selectbox("Select an Encoding Method", [
            "Label Encoding",
            "One-Hot Encoding",
            "Mean Encoding"
        ])

        if option == "Label Encoding":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Label Encoding</h1>", unsafe_allow_html=True)
            column_to_encode = st.sidebar.selectbox("Select column for Label Encoding", self.data.columns)
            if st.button("Encode"):
                if not pd.api.types.is_numeric_dtype(self.data[column_to_encode]):
                    self.data = self.label_encoding(self.data, column_to_encode)
                    st.dataframe(self.data)

        elif option == "One-Hot Encoding":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>One-Hot Encoding</h1>", unsafe_allow_html=True)
            column_to_encode = st.sidebar.selectbox("Select column for One-Hot Encoding", self.data.columns)
            if st.button("Encode"):
                if not pd.api.types.is_numeric_dtype(self.data[column_to_encode]):
                    self.data = self.one_hot_encoding(self.data, column_to_encode)
                    st.dataframe(self.data) 

        elif option == "Mean Encoding":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Mean Encoding</h1>", unsafe_allow_html=True)
            column_to_encode = st.sidebar.selectbox("Select column for Mean Encoding", self.data.columns)
            if st.button("Encode"):
                if not pd.api.types.is_numeric_dtype(self.data[column_to_encode]):
                    self.data = self.mean_encoding(self.data, column_to_encode)
                    st.dataframe(self.data)

        return self.data
