import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from imblearn.over_sampling import SMOTE
from sklearn.metrics import r2_score,mean_absolute_error
import streamlit as st
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


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
        
    def confusion_matrix(self, y_true, y_pred, labels=None):
        """
        Display a confusion matrix based on true and predicted labels.
        """
        if labels is None:
            labels = np.unique(np.concatenate((y_true, y_pred)))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        st.write("Confusion Matrix:")
        st.table(cm_df)

        # Create figure and axes objects explicitly
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Plot heatmap on the specific axes
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        
        # Set labels and title using the axes object
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        # Display the figure using st.pyplot with the figure object
        st.pyplot(fig)

    def logistic_regression(self, C, max_iter):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

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
        
        use_smote = st.checkbox("Use SMOTE")
        
        X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
        
        if use_smote:
            # Apply SMOTE to balance the class distribution
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        pipe = Pipeline([
            ('scaler', scaler),
            ('classifier', LogisticRegression(C=C, max_iter=max_iter))
        ])

        pipe.fit(X_train_resampled, y_train_resampled)

        y_pred = pipe.predict(X_test)

        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write('Classification Report:\n', classification_report(y_test, y_pred))
        
        # Display confusion matrix
        self.confusion_matrix(y_test, y_pred)

        # Writing the coefficients and intercept
        coef = pipe.named_steps['classifier'].coef_
        intercept = pipe.named_steps['classifier'].intercept_
        st.write('Coefficients:', coef)
        st.write('Intercept:', intercept)

        # Save the trained model to a file
        model_file = "trained_model_logistic.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )

    def random_forest_classifier(self, n_estimators, max_depth, max_features):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

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
        
        use_smote = st.checkbox("Use SMOTE")
        
        X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
        
        if use_smote:
            # Apply SMOTE to balance the class distribution
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        pipe = Pipeline([
            ('scaler', scaler),
            ('classifier', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features))
        ])

        pipe.fit(X_train_resampled, y_train_resampled)

        y_pred = pipe.predict(X_test)

        # Display evaluation metrics
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write('Classification Report:\n', classification_report(y_test, y_pred))
        
        # Display confusion matrix
        self.confusion_matrix(y_test, y_pred)

        # Save the trained model to a file
        model_file = "trained_mode_random_class.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
            )

    def decision_tree_classifier(self, max_depth, min_samples_split):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

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
            
        use_smote = st.checkbox("Use SMOTE")
        
        X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
        
        if use_smote:
            # Apply SMOTE to balance the class distribution
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        pipe = Pipeline([
            ('scaler', scaler),
            ('classifier', DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split))
        ])

        pipe.fit(X_train_resampled, y_train_resampled)

        y_pred = pipe.predict(X_test)

        # Display evaluation metrics
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write('Classification Report:\n', classification_report(y_test, y_pred))
        
        # Display confusion matrix
        self.confusion_matrix(y_test, y_pred)
            
        # Save the trained model to a file
        model_file = "trained_model_decision_tree_classifier.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
        )

    def knn_classifier(self, n_neighbors, algorithm):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

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
            
        use_smote = st.checkbox("Use SMOTE")
        
        X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
        
        if use_smote:
            # Apply SMOTE to balance the class distribution
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        pipe = Pipeline([
            ('scaler', scaler),
            ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm))
        ])

        pipe.fit(X_train_resampled, y_train_resampled)

        y_pred = pipe.predict(X_test)
        
        # Display evaluation metrics
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write('Classification Report:\n', classification_report(y_test, y_pred))
        
        # Display confusion matrix
        self.confusion_matrix(y_test, y_pred)
            
        # Save the trained model to a file
        model_file = "trained_model_knn_classifier.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
        )

    def support_vector_classifier(self, C, kernel):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

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
            
        use_smote = st.checkbox("Use SMOTE")
        
        X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
        
        if use_smote:
            # Apply SMOTE to balance the class distribution
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        pipe = Pipeline([
            ('scaler', scaler),
            ('classifier', SVC(C=C, kernel=kernel))
        ])

        pipe.fit(X_train_resampled, y_train_resampled)

        y_pred = pipe.predict(X_test)

        # Display evaluation metrics
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write('Classification Report:\n', classification_report(y_test, y_pred))
        
        # Display confusion matrix
        self.confusion_matrix(y_test, y_pred)
            
        # Save the trained model to a file
        model_file = "trained_model_support_vector_classifier.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
        )

    def gradient_boosting_classifier(self, n_estimators, learning_rate, max_depth):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

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
            
        use_smote = st.checkbox("Use SMOTE")
        
        X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
        
        if use_smote:
            # Apply SMOTE to balance the class distribution
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        pipe = Pipeline([
            ('scaler', scaler),
            ('classifier', GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth))
        ])

        pipe.fit(X_train_resampled, y_train_resampled)

        y_pred = pipe.predict(X_test)

        # Display evaluation metrics
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write('Classification Report:\n', classification_report(y_test, y_pred))
        
        # Display confusion matrix
        self.confusion_matrix(y_test, y_pred)
            
        # Save the trained model to a file
        model_file = "trained_model_gradient_boosting_classifier.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
        )

    def adaboost_classifier(self, n_estimators, learning_rate):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

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
            
        use_smote = st.checkbox("Use SMOTE")
        
        X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
        
        if use_smote:
            # Apply SMOTE to balance the class distribution
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        pipe = Pipeline([
            ('scaler', scaler),
            ('classifier', AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate))
        ])

        pipe.fit(X_train_resampled, y_train_resampled)

        y_pred = pipe.predict(X_test)

        # Display evaluation metrics
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write('Classification Report:\n', classification_report(y_test, y_pred))
        
        # Display confusion matrix
        self.confusion_matrix(y_test, y_pred)
            
        # Save the trained model to a file
        model_file = "trained_model_adaboost_classifier.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
        )

    def xgboost_classifier(self, n_estimators, max_depth, learning_rate):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

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
            
        use_smote = st.checkbox("Use SMOTE")
        
        X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
        
        if use_smote:
            # Apply SMOTE to balance the class distribution
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        pipe = Pipeline([
            ('scaler', scaler),
            ('classifier', XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate))
        ])

        pipe.fit(X_train_resampled, y_train_resampled)

        y_pred = pipe.predict(X_test)

        # Display evaluation metrics
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write('Classification Report:\n', classification_report(y_test, y_pred))
        
        # Display confusion matrix
        self.confusion_matrix(y_test, y_pred)
            
        # Save the trained model to a file
        model_file = "trained_model_xgboost_classifier.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
        )

    def stacking_classifier(self):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

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
        
        use_smote = st.checkbox("Use SMOTE")
        
        X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
        
        if use_smote:
            # Apply SMOTE to balance the class distribution
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
                
        # Define base estimators
        base_estimators = [
            ('decision_tree', DecisionTreeClassifier()),
            ('random_forest', RandomForestClassifier())
            # Add more base estimators as needed
        ]
        
        # Initialize StackingClassifier
        stacking_classifier = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression()  # You can replace LogisticRegression() with any other final estimator
        )

        pipe = Pipeline([
            ('scaler', scaler),
            ('classifier', stacking_classifier)
        ])

        pipe.fit(X_train_resampled, y_train_resampled)

        y_pred = pipe.predict(X_test)

        # Display evaluation metrics
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write('Classification Report:\n', classification_report(y_test, y_pred))
        
        # Display confusion matrix
        self.confusion_matrix(y_test, y_pred)
            
        # Save the trained model to a file
        model_file = "trained_model_stacking_classifier.pkl"
        joblib.dump(pipe, model_file)

        # Provide a download button to the user
        st.download_button(
            label="Download Trained Model",
            data=open(model_file, 'rb').read(),
            file_name=model_file,
            mime="application/octet-stream"
        )

    
    def algo(self):
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Select Model Type</h1>", unsafe_allow_html=True)
        model_type = st.sidebar.radio("Select Model Type", ["Predictive", "Classification"])

        if model_type == "Predictive":
            st.markdown("<h1 style='text-align: center; font-size: 30px;'>Predictive Algorithms</h1>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>", unsafe_allow_html=True)
            st.dataframe(self.data, width=800)
            
            algorithm_option = st.sidebar.selectbox("Select Algorithm", ["Linear Regression", 
                                                                        "Ridge Regression", 
                                                                        "Lasso Regression",
                                                                        "KNN Regression", 
                                                                        "Decision Tree Regression", 
                                                                        "SVR Regression", 
                                                                        "Random Forest Regression", 
                                                                        "Extra Trees Regression", 
                                                                        "AdaBoost Regression", 
                                                                        "Gradient Boosting Regression", 
                                                                        "XGBRegressor Regression", 
                                                                        "Stacking Regressor"])

            if algorithm_option == "Linear Regression":
                self.linear_regression()
            elif algorithm_option == "Ridge Regression":
                alpha = st.number_input("Enter alpha value", min_value=0.001, max_value=1000.0, value=1.0)
                self.ridge_regression(alpha=alpha)  
            elif algorithm_option == "Lasso Regression":
                alpha = st.number_input("Enter alpha value", min_value=0.001, max_value=1000.0, value=0.001)
                self.lasso_regression(alpha=alpha)  
            elif algorithm_option == "KNN Regression":
                self.knn_regression()
            elif algorithm_option == "Decision Tree Regression":
                self.decision_tree_regression()
            elif algorithm_option == "SVR Regression":
                self.svr_regression()
            elif algorithm_option == "Random Forest Regression":
                n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=15)
                max_features = st.number_input("Enter max features", min_value=0.1, max_value=1.0, value=0.75)
                max_samples = st.number_input("Enter max samples", min_value=0.1, max_value=1.0, value=0.5)
                self.randomforest_regression(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, max_samples=max_samples)
            elif algorithm_option == "Extra Trees Regression":
                n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=15)
                max_features = st.number_input("Enter max features", min_value=0.1, max_value=1.0, value=0.75)
                self.extratrees_regression(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
            elif algorithm_option == "AdaBoost Regression":
                n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
                learning_rate = st.number_input("Enter learning rate", min_value=0.01, max_value=1.0, value=1.0)
                self.adaboost_regression(n_estimators=n_estimators, learning_rate=learning_rate)
            elif algorithm_option == "Gradient Boosting Regression":
                n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
                max_features = st.number_input("Enter max features", min_value=0.1, max_value=1.0, value=0.75)
                self.gradientboosting_regression(n_estimators=n_estimators, max_features=max_features)
            elif algorithm_option == "XGBRegressor Regression":
                n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=3)
                learning_rate = st.number_input("Enter learning rate", min_value=0.01, max_value=1.0, value=0.1)
                self.xgbregressor_regression(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
            elif algorithm_option == "Stacking Regressor":
                self.stackingregressor_regression()
                
        elif model_type == "Classification":
            st.markdown("<h1 style='text-align: center; font-size: 30px;'>Classification Algorithms</h1>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>", unsafe_allow_html=True)
            st.dataframe(self.data, width=800)

            algorithm_option = st.sidebar.selectbox("Select Algorithm", ["Logistic Regression",
                                                                        "Random Forest Classifier",
                                                                        "Decision Tree Classifier",
                                                                        "KNN Classifier",
                                                                        "Support Vector Classifier",
                                                                        "Gradient Boosting Classifier",
                                                                        "AdaBoost Classifier",
                                                                        "XGBoost Classifier",
                                                                        "Stacking Classifier"])

            if algorithm_option == "Logistic Regression":
                C = st.number_input("Enter regularization strength (C)", min_value=0.01, max_value=100.0, value=1.0)
                max_iter = st.number_input("Enter max iterations", min_value=100, max_value=1000, value=100)
                self.logistic_regression(C=C, max_iter=max_iter)
            elif algorithm_option == "Random Forest Classifier":
                n_estimators = st.number_input("Enter number of estimators", min_value=10, max_value=1000, value=100)
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=15)
                max_features = st.number_input("Enter max features", min_value=0.1, max_value=1.0, value=0.75)
                self.random_forest_classifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
            elif algorithm_option == "Decision Tree Classifier":
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=15)
                min_samples_split = st.number_input("Enter min samples split", min_value=2, max_value=10, value=2)
                self.decision_tree_classifier(max_depth=max_depth, min_samples_split=min_samples_split)
            elif algorithm_option == "KNN Classifier":
                n_neighbors = st.number_input("Enter number of neighbors", min_value=1, max_value=50, value=5)
                algorithm = st.selectbox("Select algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
                self.knn_classifier(n_neighbors=n_neighbors, algorithm=algorithm)
            elif algorithm_option == "Support Vector Classifier":
                C = st.number_input("Enter regularization parameter (C)", min_value=0.01, max_value=100.0, value=1.0)
                kernel = st.selectbox("Select kernel", ["linear", "poly", "rbf", "sigmoid"])
                self.support_vector_classifier(C=C, kernel=kernel)
            elif algorithm_option == "Gradient Boosting Classifier":
                n_estimators = st.number_input("Enter number of estimators", min_value=10, max_value=1000, value=100)
                learning_rate = st.number_input("Enter learning rate", min_value=0.01, max_value=1.0, value=0.1)
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=3)
                self.gradient_boosting_classifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            elif algorithm_option == "AdaBoost Classifier":
                n_estimators = st.number_input("Enter number of estimators", min_value=10, max_value=1000, value=50)
                learning_rate = st.number_input("Enter learning rate", min_value=0.01, max_value=1.0, value=1.0)
                self.adaboost_classifier(n_estimators=n_estimators, learning_rate=learning_rate)
            elif algorithm_option == "XGBoost Classifier":
                n_estimators = st.number_input("Enter number of estimators", min_value=10, max_value=1000, value=100)
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=3)
                learning_rate = st.number_input("Enter learning rate", min_value=0.01, max_value=1.0, value=0.1)
                self.xgboost_classifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
            elif algorithm_option == "Stacking Classifier":
                self.stacking_classifier()
