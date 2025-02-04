import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import r2_score,mean_absolute_error
import streamlit as st
import joblib

class RegressionModel:
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
                st.info(f"Error converting column '{col}': {e}")
        
        return data

    def get_scaler_instance(self, scaler_name):

        if scaler_name == 'StandardScaler':
            return StandardScaler()
        elif scaler_name == 'MinMaxScaler':
            return MinMaxScaler()
        elif scaler_name == 'RobustScaler':
            return RobustScaler()
        elif scaler_name == 'Normalizer':
            return Normalizer()
        else:
            raise ValueError(f"Unsupported scaler method: {scaler_name}")

    def linear_regression(self):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)       
        
        if not features:
            st.warning("Please select at least one feature column.")
            return
    
        if not target:
            st.warning("Please select a target variable.")
            return

        X_train,X_test,y_train,y_test = train_test_split(self.df[features], self.df[target], test_size=test_size, random_state=random_state)
        
        pipe = Pipeline([
            ('scaler', scaler),
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

        # Save the trained model to a file
        model_file = "trained_model_linear.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )

    def ridge_regression(self, alpha=1.0):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)
        
        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train,X_test,y_train,y_test = train_test_split(self.df[features], self.df[target], test_size=test_size, random_state=random_state)
        
        pipe = Pipeline([
            ('scaler', scaler),
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

        # Save the trained model to a file
        model_file = "trained_mode_ridge.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )
    
    def lesso_regression(self, alpha=0.001):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train,X_test,y_train,y_test = train_test_split(self.df[features], self.df[target], test_size=test_size, random_state=random_state)
        
        pipe = Pipeline([
            ('scaler', scaler),
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

        # Save the trained model to a file
        model_file = "trained_mode_lesso.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )
    
    def knn_regression(self):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=test_size, random_state=random_state)
       
        n_neighbors = st.number_input("N Neighbors", value=3)
        
        pipe = Pipeline([
            ('scaler', scaler),
            ('regressor', KNeighborsRegressor(n_neighbors=n_neighbors))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))

        # Save the trained model to a file
        model_file = "trained_mode_knn.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )
    
    def decision_tree_regression(self):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=8, step=1)
        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
        min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=20, value=1, step=1)


        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=test_size, random_state=random_state)
        
        pipe = Pipeline([
            ('scaler', scaler),
            ('regressor', DecisionTreeRegressor(max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))

        # Save the trained model to a file
        model_file = "trained_mode_dicision.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )
    
    def svr_regression(self):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=test_size, random_state=random_state)
        
        C = st.slider("C (Regularization Parameter)", min_value=1, max_value=10000, value=1000, step=100)
        epsilon = st.slider("Epsilon (Tube Radius)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        kernel = st.selectbox("Kernel", ['linear', 'poly', 'rbf', 'sigmoid'], index=2)  # Default: 'rbf'

        pipe = Pipeline([
            ('scaler', scaler),
            ('regressor', SVR(kernel=kernel, C=C, epsilon=epsilon))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))

        # Save the trained model to a file
        model_file = "trained_mode_svr.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )
    
    def randomforest_regression(self, n_estimators=100, max_depth=15, max_features=0.75, max_samples=0.5):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=test_size, random_state=random_state)
        
        pipe = Pipeline([
            ('scaler', scaler),
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

        # Save the trained model to a file
        model_file = "trained_mode_rf_regression.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )
    
    def extratrees_regression(self, n_estimators=100, max_depth=15, max_features=0.75, max_samples=None):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=test_size, random_state=random_state)
        
        pipe = Pipeline([
            ('scaler', scaler),
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

        # Save the trained model to a file
        model_file = "trained_mode_extratree.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )
    
    def adaboost_regression(self, n_estimators, learning_rate):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=test_size, random_state=random_state)
        
        pipe = Pipeline([
            ('scaler', scaler),
            ('regressor', AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))

        # Save the trained model to a file
        model_file = "trained_mode_adaboost.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )
    
    def gradientboosting_regression(self, n_estimators, max_features):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=test_size, random_state=random_state)
        
        pipe = Pipeline([
            ('scaler', scaler),
            ('regressor', GradientBoostingRegressor(n_estimators=n_estimators,
                                                    max_features=max_features))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))

        # Save the trained model to a file
        model_file = "trained_mode_gradientboost.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )
    
    def xgbregressor_regression(self, n_estimators, max_depth, learning_rate):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=test_size, random_state=random_state)
        
        pipe = Pipeline([
            ('scaler', scaler),
            ('regressor', XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))

        # Save the trained model to a file
        model_file = "trained_mode_xgb.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )
    
    def stackingregressor_regression(self):
        features = st.multiselect("Select Feature Columns", self.df.columns)
        target = st.selectbox("Select Target Variable", self.df.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(self.df[features], self.df[target], test_size=test_size, random_state=random_state)

        estimators = [
            ('rf', RandomForestRegressor(n_estimators=350, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15)),
            ('gbdt', GradientBoostingRegressor(n_estimators=100, max_features=0.5)),
            ('xgb', XGBRegressor(n_estimators=25, learning_rate=0.3, max_depth=5))
        ]

        pipe = Pipeline([
            ('scaler',scaler),
            ('regressor', StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100)))
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        st.write('R2 score:', r2_score(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))

        # Save the trained model to a file
        model_file = "trained_mode_stack.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )