import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from ...utils.code_display import ShowCode
from ...core.logging_config import get_logger


class DataEncoder:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data
        self.view_code = ShowCode()
        self.view_code.set_target_class(DataEncoder)
        self.logger = get_logger(__name__)

    def label_encoding(self, col):
        encoder = LabelEncoder()
        self.data[col] = encoder.fit_transform(self.data[col])
        return self.data

    def one_hot_encoding(self, col):
        if len(self.data[col].unique()) > 2:
            # Initialize the OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False)

            # Fit and transform the data
            encoded_array = encoder.fit_transform(self.data[[col]])

            # Get feature names
            feature_names = [f"{col}_{val}" for val in encoder.categories_[0]]

            # Create a DataFrame with the encoded values
            encoded_df = pd.DataFrame(
                encoded_array, columns=feature_names, index=self.data.index
            )

            # Concatenate with original data and drop the original column
            self.data = pd.concat([self.data, encoded_df], axis=1)
            self.data.drop(columns=[col], inplace=True)

        return self.data

    def mean_encoding(self, col, target_col):
        if len(self.data[col].unique()) > 2:
            mean_encoded = self.data.groupby(col)[target_col].mean()
            self.data[col + "_mean_encoded"] = self.data[col].map(mean_encoded)
        return self.data

    def target_encoding(self, col, target_col):
        if len(self.data[col].unique()) > 2:
            # Custom implementation of target encoding without category_encoders
            # Calculate the mean of target variable for each category
            target_means = self.data.groupby(col)[target_col].mean()
            
            # Replace each category with its corresponding target mean
            self.data[col] = self.data[col].map(target_means)
        return self.data

    def frequency_encoding(self, col):
        freq_encoded = self.data[col].value_counts().to_dict()
        self.data[col + "_freq_encoded"] = self.data[col].map(freq_encoded)
        return self.data

    def binary_encoding(self, col):
        # Custom implementation of binary encoding without category_encoders
        # Get unique values
        unique_values = self.data[col].unique()
        
        # Create mapping dictionary
        mapping_dict = {val: format(i, 'b').zfill(len(bin(len(unique_values))[2:])) 
                        for i, val in enumerate(unique_values)}
        
        # Apply binary encoding
        max_digits = max(len(value) for value in mapping_dict.values())
        
        # Create binary columns
        for digit_idx in range(max_digits):
            col_name = f"{col}_bin_{digit_idx}"
            self.data[col_name] = self.data[col].map(
                lambda x: int(mapping_dict[x][digit_idx]) if x in mapping_dict else None
            )
            
        # Drop original column if needed
        # self.data.drop(columns=[col], inplace=True)
        
        return self.data

    def encoder(self):
        st.markdown(
            "<h1 style='text-align: center; font-size: 30px;'>Encode Data</h1>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            "<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>",
            unsafe_allow_html=True,
        )
        st.dataframe(self.data, width=800)

        options = [
            "Label Encoding",
            "One-Hot Encoding",
            "Mean Encoding",
            "Target Encoding",
            "Frequency Encoding",
            "Binary Encoding",
        ]

        option = st.sidebar.selectbox("Select an Encoding Method", options)

        column_to_encode = st.sidebar.selectbox(
            "Select column for Encoding", self.data.columns
        )

        if option in ["Mean Encoding", "Target Encoding"]:
            target_col = st.sidebar.selectbox("Select target column", self.data.columns)
        else:
            target_col = None

        if "encoded_data" not in st.session_state:
            st.session_state.encoded_data = None

        if "show_code" not in st.session_state:
            st.session_state.show_code = False

        if st.button("Encode"):
            if not pd.api.types.is_numeric_dtype(self.data[column_to_encode]):
                if option == "Label Encoding":
                    st.session_state.encoded_data = self.label_encoding(
                        column_to_encode
                    )
                elif option == "One-Hot Encoding":
                    st.session_state.encoded_data = self.one_hot_encoding(
                        column_to_encode
                    )
                elif option == "Mean Encoding":
                    st.session_state.encoded_data = self.mean_encoding(
                        column_to_encode, target_col
                    )
                elif option == "Target Encoding":
                    st.session_state.encoded_data = self.target_encoding(
                        column_to_encode, target_col
                    )
                elif option == "Frequency Encoding":
                    st.session_state.encoded_data = self.frequency_encoding(
                        column_to_encode
                    )
                elif option == "Binary Encoding":
                    st.session_state.encoded_data = self.binary_encoding(
                        column_to_encode
                    )
            else:
                st.warning(
                    f"The selected column '{column_to_encode}' is already numeric. Encoding is typically applied to categorical columns."
                )

        if st.session_state.encoded_data is not None:
            st.write("Encoded Data:")
            st.dataframe(st.session_state.encoded_data)

            st.session_state.show_code = st.checkbox(
                "Show Code", value=st.session_state.show_code
            )

            if st.session_state.show_code:
                if option == "Label Encoding":
                    self.view_code._display_code("label_encoding")
                elif option == "One-Hot Encoding":
                    self.view_code._display_code("one_hot_encoding")
                elif option == "Mean Encoding":
                    self.view_code._display_code("mean_encoding")
                elif option == "Target Encoding":
                    self.view_code._display_code("target_encoding")
                elif option == "Frequency Encoding":
                    self.view_code._display_code("frequency_encoding")
                elif option == "Binary Encoding":
                    self.view_code._display_code("binary_encoding")

        return self.data