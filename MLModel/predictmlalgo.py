import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import r2_score,mean_absolute_error
import streamlit as st

class PredictAlgo:
    def __init__(self, df) -> None:
        self.data = df
        self.df = self.convert_to_float(self.data).select_dtypes(include='number')

        
    def convert_to_float(self, data):
        """
        Convert integer and numeric object columns to float.
        """
        # Create a copy to avoid modifying the original data
        data = self.data.copy()
        
        # Convert integer columns to float
        int_cols = data.select_dtypes(include=['int']).columns
        data[int_cols] = data[int_cols].astype('float')
        
        # Convert object columns to float
        obj_cols = data.select_dtypes(include=['object']).columns
        for col in obj_cols:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except ValueError as e:
                print(f"Error converting column '{col}': {e}")
        
        return data
    
    def linear_regression(self):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)
        
        
        if not features:
            st.warning("Please select at least one feature column.")
            return
    
        if not target:
            st.warning("Please select a target variable.")
            return
            

        X_train,X_test,y_train,y_test = train_test_split(self.df[features], self.df[target], test_size=0.2, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score', r2_score(y_test,y_pred))
        st.write('MAE', mean_absolute_error(y_test,y_pred))
        
        # Writing the weights and bias
        coef = pipe.named_steps['regressor'].coef_
        intercept = pipe.named_steps['regressor'].intercept_
        st.write('Weights:', coef)
        st.write('Bias:', intercept)

    def logistic_regression(self):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        
        X_train,X_test,y_train,y_test = train_test_split(self.data[features], self.data[target], test_size=0.2, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write('Classification Report:\n', classification_report(y_test, y_pred))
        
        # Writing the coefficients and intercept
        coef = pipe.named_steps['classifier'].coef_
        intercept = pipe.named_steps['classifier'].intercept_
        st.write('Coefficients:', coef)
        st.write('Intercept:', intercept)

    def ridge_regression(self, alpha=1.0):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)
        
        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train,X_test,y_train,y_test = train_test_split(self.df[features], self.df[target], test_size=0.2, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=alpha))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score', r2_score(y_test, y_pred))
        st.write('MAE', mean_absolute_error(y_test, y_pred))
        
        # Writing the coefficients and intercept
        coef = pipe.named_steps['regressor'].coef_
        intercept = pipe.named_steps['regressor'].intercept_
        st.write('Weights:', coef)
        st.write('Bias:', intercept)
    
    def lesso_regression(self, alpha=0.001):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train,X_test,y_train,y_test = train_test_split(self.df[features], self.df[target], test_size=0.2, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Lasso(alpha=alpha))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score', r2_score(y_test, y_pred))
        st.write('MAE', mean_absolute_error(y_test, y_pred))
        
        # Writing the coefficients and intercept
        coef = pipe.named_steps['regressor'].coef_
        intercept = pipe.named_steps['regressor'].intercept_
        st.write('Weights:', coef)
        st.write('Bias:', intercept)
    
    def KNN_Regression(self):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=0.2, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', KNeighborsRegressor(n_neighbors=3))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))
    
    def Decision_Tree_Regression(self):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=0.2, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', DecisionTreeRegressor(max_depth=8))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))
    
    def SVR_Regression(self):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=0.2, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SVR(kernel='rbf', C=10000, epsilon=0.1))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))
    
    def RandomForest_Regression(self, n_estimators=100, max_depth=15, max_features=0.75, max_samples=0.5):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=0.2, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=n_estimators,
                                                random_state=3,
                                                max_samples=max_samples,
                                                max_features=max_features,
                                                max_depth=max_depth))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))

    
    def ExtraTrees_Regression(self, n_estimators=100, max_depth=15, max_features=0.75, max_samples=None):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=0.2, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ExtraTreesRegressor(n_estimators=n_estimators,
                                            random_state=3,
                                            max_samples=max_samples,
                                            max_features=max_features,
                                            max_depth=max_depth))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))

    
    def AdaBoost_Regression(self, n_estimators, learning_rate):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=0.2, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))
    
    def GradientBoosting_Regression(self, n_estimators, max_features):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=0.2, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(n_estimators=n_estimators,
                                                    max_features=max_features))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))
    
    def XGBRegressor_Regression(self, n_estimators, max_depth, learning_rate):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=0.2, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))
    
    def StackingRegressor_Regression(self):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=0.2, random_state=42)

        estimators = [
            ('rf', RandomForestRegressor(n_estimators=350, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15)),
            ('gbdt', GradientBoostingRegressor(n_estimators=100, max_features=0.5)),
            ('xgb', XGBRegressor(n_estimators=25, learning_rate=0.3, max_depth=5))
        ]

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100)))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))
    
    
    def algo(self):
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Predictive Algorithms</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>", unsafe_allow_html=True)
        st.dataframe(self.data, width=800)
        
        algorithm_option = st.sidebar.selectbox("Select Algorithm", ["Linear Regression", "Logistic Regression", "Ridge Regression", "Lasso Regression",
                                                            "KNN Regression", "Decision Tree Regression", "SVR Regression", 
                                                            "Random Forest Regression", "Extra Trees Regression", 
                                                            "AdaBoost Regression", "Gradient Boosting Regression", 
                                                            "XGBRegressor Regression", "Stacking Regressor"])

        if algorithm_option == "Linear Regression":
            self.linear_regression()
        elif algorithm_option == "Logistic Regression":
            self.logistic_regression()
        elif algorithm_option == "Ridge Regression":
            alpha = st.number_input("Enter alpha value", min_value=0.001, max_value=1000.0, value=1.0)
            self.ridge_regression(alpha=alpha)  
        elif algorithm_option == "Lasso Regression":
            alpha = st.number_input("Enter alpha value", min_value=0.001, max_value=1000.0, value=0.001)
            self.lesso_regression(alpha=alpha)  
        elif algorithm_option == "KNN Regression":
            self.KNN_Regression()
        elif algorithm_option == "Decision Tree Regression":
            self.Decision_Tree_Regression()
        elif algorithm_option == "SVR Regression":
            self.SVR_Regression()
        elif algorithm_option == "Random Forest Regression":
            n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
            max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=15)
            max_features = st.number_input("Enter max features", min_value=0.1, max_value=1.0, value=0.75)
            max_samples = st.number_input("Enter max samples", min_value=0.1, max_value=1.0, value=0.5)
            self.RandomForest_Regression(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, max_samples=max_samples)
        elif algorithm_option == "Extra Trees Regression":
            n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
            max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=15)
            max_features = st.number_input("Enter max features", min_value=0.1, max_value=1.0, value=0.75)
            self.ExtraTrees_Regression(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
        elif algorithm_option == "AdaBoost Regression":
            n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
            learning_rate = st.number_input("Enter learning rate", min_value=0.01, max_value=1.0, value=1.0)
            self.AdaBoost_Regression(n_estimators=n_estimators, learning_rate=learning_rate)
        elif algorithm_option == "Gradient Boosting Regression":
            n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
            max_features = st.number_input("Enter max features", min_value=0.1, max_value=1.0, value=0.75)
            self.GradientBoosting_Regression(n_estimators=n_estimators, max_features=max_features)
        elif algorithm_option == "XGBRegressor Regression":
            n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
            max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=3)
            learning_rate = st.number_input("Enter learning rate", min_value=0.01, max_value=1.0, value=0.1)
            self.XGBRegressor_Regression(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
        elif algorithm_option == "Stacking Regressor":
            self.StackingRegressor_Regression()

