import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, PolynomialFeatures, RobustScaler, QuantileTransformer, PowerTransformer
import streamlit as st
import numpy as np
import plotly.express as px
from show_code import ShowCode

class DataTransformer:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data
        self.view_code = ShowCode()
        self.view_code.set_target_class(DataTransformer)

    def standard_scaling(self, col):
        scaler = StandardScaler()
        self.data[col + '_scaled'] = scaler.fit_transform(self.data[[col]])
        return self.data
    
    def min_max_scaling(self, col):
        scaler = MinMaxScaler()
        self.data[col + '_minmax_scaled'] = scaler.fit_transform(self.data[[col]])
        return self.data
    
    def normalization(self, col):
        normalizer = Normalizer()
        self.data[col + '_normalized'] = normalizer.fit_transform(self.data[[col]])
        return self.data
    
    def log_transformation(self, col):
        self.data[col + '_log_transformed'] = np.log1p(self.data[col])
        return self.data
    
    def polynomial_features(self, col, degree):
        poly = PolynomialFeatures(degree)
        poly_features = poly.fit_transform(self.data[[col]])
        poly_feature_names = [f"{col}_poly_{i}" for i in range(poly_features.shape[1])]
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=self.data.index)
        self.data = pd.concat([self.data, poly_df], axis=1)
        return self.data
    
    def box_cox_transformation(self, col):
        pt = PowerTransformer(method='box-cox')
        self.data[col + '_boxcox_transformed'] = pt.fit_transform(self.data[[col]])
        return self.data
    
    def yeo_johnson_transformation(self, col):
        pt = PowerTransformer(method='yeo-johnson')
        self.data[col + '_yeojohnson_transformed'] = pt.fit_transform(self.data[[col]])
        return self.data
    
    def robust_scaling(self, col):
        scaler = RobustScaler()
        self.data[col + '_robust_scaled'] = scaler.fit_transform(self.data[[col]])
        return self.data
    
    def quantile_transformation(self, col, output_distribution='uniform'):
        qt = QuantileTransformer(output_distribution=output_distribution)
        self.data[col + '_quantile_transformed'] = qt.fit_transform(self.data[[col]])
        return self.data
    
    def plot_histogram(self, col, title):
        fig = px.histogram(self.data, x=col, title=title)
        st.plotly_chart(fig)

    def transformer(self):
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Transform Data</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h1>", unsafe_allow_html=True)
        st.dataframe(self.data, width=800)

        options = [
            "Standard Scaling",
            "Min-Max Scaling",
            "Normalization",
            "Log Transformation",
            "Polynomial Features",
            "Box-Cox Transformation",
            "Yeo-Johnson Transformation",
            "Robust Scaling",
            "Quantile Transformation"
        ]

        option = st.sidebar.selectbox("Select a Transformation Method", options)

        column_to_transform = st.sidebar.selectbox("Select column for Transformation", self.data.columns)
        
        if option == "Polynomial Features":
            degree = st.sidebar.slider("Select degree for Polynomial Features", 2, 5, 2)
        else:
            degree = None
        
        if option == "Quantile Transformation":
            output_distribution = st.sidebar.selectbox("Select output distribution for Quantile Transformation", ["uniform", "normal"])
        else:
            output_distribution = None

        st.markdown(f"### Original Data: {column_to_transform}")
        self.plot_histogram(column_to_transform, f"Original Data - {column_to_transform}")

        # Initialize session state
        if 'transformed_data' not in st.session_state:
            st.session_state.transformed_data = None
        if 'show_code' not in st.session_state:
            st.session_state.show_code = False

        if st.button("Transform"):
            if pd.api.types.is_numeric_dtype(self.data[column_to_transform]):
                st.session_state.transformed_data = self._apply_transformation(option, column_to_transform, degree, output_distribution)
            else:
                st.warning(f"The selected column '{column_to_transform}' is not numeric. Transformation can only be applied to numeric columns.")

        if st.session_state.transformed_data is not None:
            st.markdown(f"### Transformed Data: {st.session_state.transformed_data.columns[-1]}")
            self.plot_histogram(st.session_state.transformed_data.columns[-1], f"Transformed Data - {st.session_state.transformed_data.columns[-1]}")

            st.session_state.show_code = st.checkbox('Show Code', value=st.session_state.show_code)
            
            if st.session_state.show_code:
                self._display_code(option)

        return self.data

    def _apply_transformation(self, option, column, degree=None, output_distribution=None):
        if option == "Standard Scaling":
            return self.standard_scaling(column)
        elif option == "Min-Max Scaling":
            return self.min_max_scaling(column)
        elif option == "Normalization":
            return self.normalization(column)
        elif option == "Log Transformation":
            return self.log_transformation(column)
        elif option == "Polynomial Features":
            return self.polynomial_features(column, degree)
        elif option == "Box-Cox Transformation":
            return self.box_cox_transformation(column)
        elif option == "Yeo-Johnson Transformation":
            return self.yeo_johnson_transformation(column)
        elif option == "Robust Scaling":
            return self.robust_scaling(column)
        elif option == "Quantile Transformation":
            return self.quantile_transformation(column, output_distribution)

    def _display_code(self, option):
        method_name = option.lower().replace('-', '_').replace(' ', '_')
        self.view_code._display_code(method_name)