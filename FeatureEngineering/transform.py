import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, PolynomialFeatures, RobustScaler, QuantileTransformer, PowerTransformer
import streamlit as st
import numpy as np
import plotly.express as px

class DataTransformer:
    def __init__(self, data):
        self.data = data
    
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
        st.markdown(
            "<h1 style='text-align: center; font-size: 30px;'>Transform Data</h1>", 
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h1>", unsafe_allow_html=True)
        st.dataframe(self.data, width=800)

        option = st.sidebar.selectbox("Select a Transformation Method", [
            "Standard Scaling",
            "Min-Max Scaling",
            "Normalization",
            "Log Transformation",
            "Polynomial Features",
            "Box-Cox Transformation",
            "Yeo-Johnson Transformation",
            "Robust Scaling",
            "Quantile Transformation"
        ])

        column_to_transform = st.sidebar.selectbox("Select column for Transformation", self.data.columns)
        
        if option == "Polynomial Features":
            degree = st.sidebar.slider("Select degree for Polynomial Features", 2, 5, 2)
        
        if option == "Quantile Transformation":
            output_distribution = st.sidebar.selectbox("Select output distribution for Quantile Transformation", ["uniform", "normal"])

        st.markdown(f"### Original Data: {column_to_transform}")
        self.plot_histogram(column_to_transform, f"Original Data - {column_to_transform}")

        if st.button("Transform"):
            if pd.api.types.is_numeric_dtype(self.data[column_to_transform]):
                if option == "Standard Scaling":
                    self.data = self.standard_scaling(column_to_transform)
                    transformed_col = column_to_transform + '_scaled'
                elif option == "Min-Max Scaling":
                    self.data = self.min_max_scaling(column_to_transform)
                    transformed_col = column_to_transform + '_minmax_scaled'
                elif option == "Normalization":
                    self.data = self.normalization(column_to_transform)
                    transformed_col = column_to_transform + '_normalized'
                elif option == "Log Transformation":
                    self.data = self.log_transformation(column_to_transform)
                    transformed_col = column_to_transform + '_log_transformed'
                elif option == "Polynomial Features":
                    self.data = self.polynomial_features(column_to_transform, degree)
                    transformed_col = column_to_transform + '_poly_1'
                elif option == "Box-Cox Transformation":
                    self.data = self.box_cox_transformation(column_to_transform)
                    transformed_col = column_to_transform + '_boxcox_transformed'
                elif option == "Yeo-Johnson Transformation":
                    self.data = self.yeo_johnson_transformation(column_to_transform)
                    transformed_col = column_to_transform + '_yeojohnson_transformed'
                elif option == "Robust Scaling":
                    self.data = self.robust_scaling(column_to_transform)
                    transformed_col = column_to_transform + '_robust_scaled'
                elif option == "Quantile Transformation":
                    self.data = self.quantile_transformation(column_to_transform, output_distribution)
                    transformed_col = column_to_transform + '_quantile_transformed'

                st.markdown(f"### Transformed Data: {transformed_col}")
                self.plot_histogram(transformed_col, f"Transformed Data - {transformed_col}")

        return self.data