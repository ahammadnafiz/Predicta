import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class ClassificationModel:
    def __init__(self, df):
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
        
        # Convert target to numeric if it's not already
        le = LabelEncoder()
        y = le.fit_transform(self.data[target].astype(str))
        
        # Ensure features are numeric
        X = self.data[features].astype(float)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        if use_smote:
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Create and fit scaler
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)

        # Create and fit XGBoost classifier
        xgb_clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,  # Add this parameter
            eval_metric='mlogloss'    # Add this parameter
        )
        
        # Fit the model
        xgb_clf.fit(X_train_scaled, y_train_resampled)
        
        # Make predictions
        y_pred = xgb_clf.predict(X_test_scaled)
        
        # Transform predictions back to original labels
        y_test_original = le.inverse_transform(y_test)
        y_pred_original = le.inverse_transform(y_pred)

        # Display evaluation metrics
        st.write('Accuracy:', accuracy_score(y_test_original, y_pred_original))
        st.write('Classification Report:\n', classification_report(y_test_original, y_pred_original))
        
        # Display confusion matrix
        self.confusion_matrix(y_test_original, y_pred_original)
            
        # Save the model and preprocessing objects
        model_data = {
            'model': xgb_clf,
            'scaler': scaler,
            'label_encoder': le,
            'feature_names': features
        }
        
        model_file = "trained_model_xgboost_classifier.pkl"
        joblib.dump(model_data, model_file)

        # Provide download button
        with open(model_file, 'rb') as f:
            model_bytes = f.read()
            
        st.download_button(
            label="Download Trained Model",
            data=model_bytes,
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
