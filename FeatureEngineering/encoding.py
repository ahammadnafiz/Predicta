import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import streamlit as st
import category_encoders as ce

class DataEncoder:
    def __init__(self, data):
        self.data = data
    
    def label_encoding(self, col):
        label_encoder = LabelEncoder()
        self.data[col] = label_encoder.fit_transform(self.data[col])
        return self.data
    
    def one_hot_encoding(self, col):
        if len(self.data[col].unique()) > 2:  
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_values = encoder.fit_transform(self.data[[col]])
            new_cols = [col + '_' + str(i) for i in range(encoded_values.shape[1])]
            encoded_df = pd.DataFrame(encoded_values, columns=new_cols, index=self.data.index)
            self.data = pd.concat([self.data, encoded_df], axis=1)
            self.data.drop(columns=[col], inplace=True)
        return self.data
    
    def mean_encoding(self, col, target_col):
        if len(self.data[col].unique()) > 2:
            mean_encoded = self.data.groupby(col)[target_col].mean()
            self.data[col + '_mean_encoded'] = self.data[col].map(mean_encoded)
        return self.data
    
    def target_encoding(self, col, target_col):
        if len(self.data[col].unique()) > 2:
            target_encoder = ce.TargetEncoder(cols=[col])
            self.data[col + '_target_encoded'] = target_encoder.fit_transform(self.data[col], self.data[target_col])
        return self.data
    
    def frequency_encoding(self, col):
        freq_encoded = self.data[col].value_counts().to_dict()
        self.data[col + '_freq_encoded'] = self.data[col].map(freq_encoded)
        return self.data
    
    def binary_encoding(self, col):
        binary_encoder = ce.BinaryEncoder(cols=[col])
        self.data = binary_encoder.fit_transform(self.data)
        return self.data
    
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
            "Mean Encoding",
            "Target Encoding",
            "Frequency Encoding",
            "Binary Encoding"
        ])

        column_to_encode = st.sidebar.selectbox("Select column for Encoding", self.data.columns)
        
        if option in ["Mean Encoding", "Target Encoding"]:
            target_col = st.sidebar.selectbox("Select target column", self.data.columns)
        
        if st.button("Encode"):
            if not pd.api.types.is_numeric_dtype(self.data[column_to_encode]):
                if option == "Label Encoding":
                    self.data = self.label_encoding(column_to_encode)
                elif option == "One-Hot Encoding":
                    self.data = self.one_hot_encoding(column_to_encode)
                elif option == "Mean Encoding":
                    self.data = self.mean_encoding(column_to_encode, target_col)
                elif option == "Target Encoding":
                    self.data = self.target_encoding(column_to_encode, target_col)
                elif option == "Frequency Encoding":
                    self.data = self.frequency_encoding(column_to_encode)
                elif option == "Binary Encoding":
                    self.data = self.binary_encoding(column_to_encode)
                st.dataframe(self.data)

        return self.data