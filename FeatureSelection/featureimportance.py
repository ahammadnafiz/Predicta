import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go


class FeatureImportanceAnalyzer:
    def __init__(self, data):
        self.data = data

    def linear_regression_importance(self, X_train, y_train):
        try:
            model = LinearRegression()
            model.fit(X_train, y_train)

            coefficients = model.coef_
            feat_labels = X_train.columns

            # Create Plotly figure for feature importance
            fig = go.Figure()
            fig.add_trace(go.Bar(x=feat_labels, y=coefficients, marker_color='crimson'))
            fig.update_layout(title="Linear Regression Coefficients",
                            xaxis_title="Feature Name", yaxis_title="Coefficient")
            st.plotly_chart(fig)

            return model
        
        except ValueError:
            st.info("Error occurred during linear regression feature importance analysis.")
            return None    

    def random_forest_regression_importance(self, X_train, y_train, top_n=15):
        try:
            model = RandomForestRegressor(random_state=0)
            model.fit(X_train, y_train)

            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feat_labels = X_train.columns

            # Create Plotly figure for feature importance
            fig = go.Figure()
            fig.add_trace(go.Bar(x=feat_labels[indices[:top_n]], y=importances[indices[:top_n]], marker_color='crimson'))
            fig.update_layout(title=f"Random Forest Feature Importances (Top {top_n})",
                            xaxis_title="Feature Name", yaxis_title="Importance")
            st.plotly_chart(fig)

            return model
        
        except ValueError:
            st.info("Error occurred during random forest feature importance analysis.")
            return None

    def random_forest_importance(self, X_train, y_train, top_n=15):
        try:
            model = RandomForestClassifier(random_state=0)
            model.fit(X_train, y_train)

            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feat_labels = X_train.columns

            # Create Plotly figure for feature importance
            fig = go.Figure()
            fig.add_trace(go.Bar(x=feat_labels[indices[:top_n]], y=importances[indices[:top_n]], marker_color='crimson'))
            fig.update_layout(title=f"Random Forest Feature Importances (Top {top_n})",
                            xaxis_title="Feature Name", yaxis_title="Importance")
            st.plotly_chart(fig)

            return model
        
        except ValueError:
            st.info("Please use categorical labels (e.g., classes) for classification tasks.")
            return None

    def gradient_boosting_importance(self, X_train, y_train, top_n=15):
        try:
            model = GradientBoostingClassifier(random_state=0)
            model.fit(X_train, y_train)

            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feat_labels = X_train.columns

            # Create Plotly figure for feature importance
            fig = go.Figure()
            fig.add_trace(go.Bar(x=feat_labels[indices[:top_n]], y=importances[indices[:top_n]], marker_color='crimson'))
            fig.update_layout(title=f"Gradient Boosting Feature Importances (Top {top_n})",
                            xaxis_title="Feature Name", yaxis_title="Importance")
            st.plotly_chart(fig)

            return model
        
        except ValueError:
            st.info("Please use categorical labels (e.g., classes) for classification tasks.")
            return None

    # Method for chi-square test feature selection
    def chi_square_test(self, target_column, select_k=10):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        n_features = X.shape[1]  # Number of features in X

        try:
            
            if select_k >= 1:
                if select_k > n_features:
                    raise ValueError(f"k should be <= n_features = {n_features}; got {select_k}. Use k='all' to return all features.")
                sel_ = SelectKBest(chi2, k=select_k).fit(X, y)
                col = X.columns[sel_.get_support()]
                

            elif 0 < select_k < 1:
                sel_ = SelectPercentile(chi2, percentile=select_k * 100).fit(X, y)
                col = X.columns[sel_.get_support()]
                
            else:
                raise ValueError("select_k must be a positive number")

            st.write("Selected Features based on chi-square test:")
            st.write(col.tolist())
            
            return col
        
        except ValueError:
            st.info("Please use categorical labels (e.g., classes) for classification tasks.")
                
            return None

    # Method for univariate MSE feature selection
    def univariate_mse(self, X_train, y_train, X_test, y_test, threshold):
        mse_values = []
        for feature in X_train.columns:
            clf = DecisionTreeRegressor()
            clf.fit(X_train[feature].to_frame(), y_train)
            y_scored = clf.predict(X_test[feature].to_frame())
            mse_values.append(mean_squared_error(y_test, y_scored))
        mse_values = pd.Series(mse_values)
        mse_values.index = X_train.columns
        st.write(mse_values.sort_values(ascending=False))
        st.write(f"{len(mse_values[mse_values > threshold])} out of the {len(X_train.columns)} features are kept")
        keep_col = mse_values[mse_values > threshold]
        
        st.write("Selected Features based on univariate mse:")
        st.write(keep_col.tolist())  
        
        return keep_col


    def analyze_features(self):
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Feature Importance Analysis</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>", unsafe_allow_html=True)
        st.dataframe(self.data, width=800)

        target_column = st.sidebar.selectbox("Select Target Column", self.data.columns)

        option = st.sidebar.selectbox("Select a Model for Feature Importance Analysis", [
            "Linear Regression Importance",
            "Random Forest Regression Feature Importance",
            "Random Forest Classifier Importance",
            "Gradient Boosting Classifier Importance",
            "Chi-Square Test Feature Selection",
            "Univariate MSE Feature Selection"
        ])
        
        if option == "Linear Regression Importance":
            st.markdown("<h2 style='text-align: center; font-size: 25px;'>Linear Regression Feature Importance</h2>", unsafe_allow_html=True)

            if st.button("Analyze"):
                X = self.data.drop(columns=[target_column])
                y = self.data[target_column]
                self.linear_regression_importance(X, y)

        elif option == "Random Forest Regression Feature Importance":
            st.markdown("<h2 style='text-align: center; font-size: 25px;'>Random Forest Regression Feature Importance</h2>", unsafe_allow_html=True)
            if st.button("Analyze"):
                X = self.data.drop(columns=[target_column])
                y = self.data[target_column]
                self.random_forest_regression_importance(X, y)

        elif option == "Random Forest Classifier Importance":
            st.markdown("<h2 style='text-align: center; font-size: 25px;'>Random Forest Feature Importance</h2>", unsafe_allow_html=True)

            if st.button("Analyze"):
                X = self.data.drop(columns=[target_column])
                y = self.data[target_column]
                self.random_forest_importance(X, y)

        elif option == "Gradient Boosting Classifier Importance":
            st.markdown("<h2 style='text-align: center; font-size: 25px;'>Gradient Boosting Feature Importance</h2>", unsafe_allow_html=True)

            if st.button("Analyze"):
                X = self.data.drop(columns=[target_column])
                y = self.data[target_column]
                self.gradient_boosting_importance(X, y)
                
        elif option == "Chi-Square Test Feature Selection":
            select_k = st.slider("Select K Features", min_value=1, max_value=len(self.data.columns), value=10)
            
            if st.button("Analyze"):
                self.chi_square_test(target_column, select_k)
        
        elif option == "Univariate MSE Feature Selection":
            threshold = st.slider("MSE Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            
            if st.button("Analyze"):
                X = self.data.drop(columns=[target_column])
                y = self.data[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                self.univariate_mse(X_train, y_train, X_test, y_test, threshold)

