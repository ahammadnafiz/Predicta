import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go


class FeatureImportanceAnalyzer:
    def __init__(self, data):
        self.data = data
    
    def random_forest_importance(self, X_train, y_train, max_depth=10, class_weight='balanced', top_n=15, n_estimators=50, random_state=0):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=random_state, class_weight=class_weight,
                                       n_jobs=-1)
        model.fit(X_train, y_train)
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feat_labels = X_train.columns
        
        # Create Plotly figure for feature importance
        fig = go.Figure()
        fig.add_trace(go.Bar(x=feat_labels[indices[:top_n]], y=importances[indices[:top_n]],
                             error_y=dict(type='data', array=np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)[indices[:top_n]]),
                             marker_color='crimson'))
        fig.update_layout(title=f"Random Forest Feature Importances (Top {top_n})",
                          xaxis_title="Feature Name", yaxis_title="Importance")
        st.plotly_chart(fig)

        return model

    def gradient_boosting_importance(self, X_train, y_train, max_depth=10, top_n=15, n_estimators=50, random_state=0):
        model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           random_state=random_state)
        model.fit(X_train, y_train)
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feat_labels = X_train.columns
        
        # Create Plotly figure for feature importance
        fig = go.Figure()
        fig.add_trace(go.Bar(x=feat_labels[indices[:top_n]], y=importances[indices[:top_n]],
                             error_y=dict(type='data', array=np.std([tree[0].feature_importances_ for tree in model.estimators_], axis=0)[indices[:top_n]]),
                             marker_color='crimson'))
        fig.update_layout(title=f"Gradient Boosting Feature Importances (Top {top_n})",
                          xaxis_title="Feature Name", yaxis_title="Importance")
        st.plotly_chart(fig)

        return model

    # Method to analyze feature importance through shuffling
    def feature_shuffle_rf_importance(self, X_train, y_train, max_depth=None, class_weight=None, top_n=15, n_estimators=50, random_state=0):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=random_state, class_weight=class_weight,
                                       n_jobs=-1)
        model.fit(X_train, y_train)
        
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        feature_dict = {}

        for feature in X_train.columns:
            X_train_c = X_train.copy().reset_index(drop=True)
            X_train_c[feature] = X_train_c[feature].sample(frac=1, random_state=random_state).reset_index(drop=True)
            shuff_auc = roc_auc_score(y_train, model.predict_proba(X_train_c)[:, 1])
            feature_dict[feature] = train_auc - shuff_auc
        
        auc_drop = pd.Series(feature_dict, name='auc_drop').sort_values(ascending=False)
        selected_features = auc_drop[auc_drop > 0].index.tolist()

        st.write("Selected Features based on AUC Drop:")
        st.write(selected_features)

        # Create Plotly figure for feature importance based on AUC drop
        top_features = auc_drop.head(top_n)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top_features.index, y=top_features.values,
                             marker_color='crimson'))
        fig.update_layout(title=f"Feature Importance based on AUC Drop (Top {top_n})",
                          xaxis_title="Feature Name", yaxis_title="AUC Drop")
        st.plotly_chart(fig)

        return auc_drop, selected_features

    # Method for chi-square test feature selection
    def chi_square_test(self, target_column, select_k=10):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        if select_k >= 1:
            sel_ = SelectKBest(chi2, k=select_k).fit(X, y)
            col = X.columns[sel_.get_support()]

        elif 0 < select_k < 1:
            sel_ = SelectPercentile(chi2, percentile=select_k * 100).fit(X, y)
            col = X.columns[sel_.get_support()]

        else:
            raise ValueError("select_k must be a positive number")

        st.write("Selected Features based on chi square test:")
        st.write(col.tolist())
        
        return col

    # Method for univariate ROC-AUC feature selection
    def univariate_roc_auc(self, X_train, y_train, X_test, y_test, threshold):
        roc_values = []
        for feature in X_train.columns:
            clf = DecisionTreeClassifier()
            clf.fit(X_train[feature].to_frame(), y_train)
            y_scored = clf.predict_proba(X_test[feature].to_frame())
            roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))
        roc_values = pd.Series(roc_values)
        roc_values.index = X_train.columns
        st.write(roc_values.sort_values(ascending=False))
        st.write(f"{len(roc_values[roc_values > threshold])} out of the {len(X_train.columns)} features are kept")
        keep_col = roc_values[roc_values > threshold]
        
        st.write("Selected Features based on univariate roc auc:")
        st.write(keep_col.tolist())        
        
        return keep_col

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
            "Random Forest",
            "Gradient Boosting",
            "Feature Shuffle RF",
            "Chi-Square Test Feature Selection",
            "Univariate ROC-AUC Feature Selection",
            "Univariate MSE Feature Selection"
        ])

        if option == "Random Forest":
            st.markdown("<h2 style='text-align: center; font-size: 25px;'>Random Forest Feature Importance</h2>", unsafe_allow_html=True)
            st.write("Configure parameters:")
            max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=10)
            n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, value=50)
            class_weight = 'balanced'
            top_n = st.slider("Top N Features", min_value=5, max_value=30, value=15)

            if st.button("Analyze"):
                X = self.data.drop(columns=[target_column])
                y = self.data[target_column]
                self.random_forest_importance(X, y, max_depth=max_depth, class_weight=class_weight, top_n=top_n, n_estimators=n_estimators)

        elif option == "Gradient Boosting":
            st.markdown("<h2 style='text-align: center; font-size: 25px;'>Gradient Boosting Feature Importance</h2>", unsafe_allow_html=True)
            st.write("Configure parameters:")
            max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=10)
            n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, value=50)
            top_n = st.slider("Top N Features", min_value=5, max_value=30, value=15)

            if st.button("Analyze"):
                X = self.data.drop(columns=[target_column])
                y = self.data[target_column]
                self.gradient_boosting_importance(X, y, max_depth=max_depth, top_n=top_n, n_estimators=n_estimators)

        elif option == "Feature Shuffle RF":
            st.markdown("<h2 style='text-align: center; font-size: 25px;'>Feature Shuffle RF Feature Importance</h2>", unsafe_allow_html=True)
            st.write("Configure parameters:")
            max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=10)
            n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, value=50)
            top_n = st.slider("Top N Features", min_value=5, max_value=30, value=15)

            if st.button("Analyze"):
                X = self.data.drop(columns=[target_column])
                y = self.data[target_column]
                self.feature_shuffle_rf_importance(X, y, max_depth=max_depth, top_n=top_n, n_estimators=n_estimators)
                
        elif option == "Chi-Square Test Feature Selection":
            select_k = st.slider("Select K Features", min_value=1, max_value=len(self.data.columns), value=10)
            
            if st.button("Analyze"):
                self.chi_square_test(target_column, select_k)

        elif option == "Univariate ROC-AUC Feature Selection":
            threshold = st.slider("ROC-AUC Threshold", min_value=0.5, max_value=1.0, value=0.7, step=0.05)
            
            if st.button("Analyze"):
                X = self.data.drop(columns=[target_column])
                y = self.data[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                self.univariate_roc_auc(X_train, y_train, X_test, y_test, threshold)
        
        elif option == "Univariate MSE Feature Selection":
            threshold = st.slider("MSE Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            
            if st.button("Analyze"):
                X = self.data.drop(columns=[target_column])
                y = self.data[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                self.univariate_mse(X_train, y_train, X_test, y_test, threshold)