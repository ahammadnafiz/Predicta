import streamlit as st
from MLModel.regression_models import RegressionModel
from MLModel.classification_models import ClassificationModel

class PredictAlgo:
    def __init__(self, df) -> None:
        self.data = df
        self.reg_model = RegressionModel(self.data)
        self.cls_model = ClassificationModel(self.data)
    
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
                self.reg_model.linear_regression()
            elif algorithm_option == "Ridge Regression":
                alpha = st.number_input("Enter alpha value", min_value=0.001, max_value=1000.0, value=1.0)
                self.reg_model.ridge_regression(alpha=alpha)  
            elif algorithm_option == "Lasso Regression":
                alpha = st.number_input("Enter alpha value", min_value=0.001, max_value=1000.0, value=0.001)
                self.reg_model.lasso_regression(alpha=alpha)  
            elif algorithm_option == "KNN Regression":
               self.reg_model.knn_regression()
            elif algorithm_option == "Decision Tree Regression":
                self.reg_model.decision_tree_regression()
            elif algorithm_option == "SVR Regression":
                self.reg_model.svr_regression()
            elif algorithm_option == "Random Forest Regression":
                n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=15)
                max_features = st.number_input("Enter max features", min_value=0.1, max_value=1.0, value=0.75)
                max_samples = st.number_input("Enter max samples", min_value=0.1, max_value=1.0, value=0.5)
                self.reg_model.randomforest_regression(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, max_samples=max_samples)
            elif algorithm_option == "Extra Trees Regression":
                n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=15)
                max_features = st.number_input("Enter max features", min_value=0.1, max_value=1.0, value=0.75)
                self.reg_model.extratrees_regression(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
            elif algorithm_option == "AdaBoost Regression":
                n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
                learning_rate = st.number_input("Enter learning rate", min_value=0.01, max_value=1.0, value=1.0)
                self.reg_model.adaboost_regression(n_estimators=n_estimators, learning_rate=learning_rate)
            elif algorithm_option == "Gradient Boosting Regression":
                n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
                max_features = st.number_input("Enter max features", min_value=0.1, max_value=1.0, value=0.75)
                self.reg_model.gradientboosting_regression(n_estimators=n_estimators, max_features=max_features)
            elif algorithm_option == "XGBRegressor Regression":
                n_estimators = st.number_input("Enter number of estimators", min_value=1, max_value=1000, value=100)
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=3)
                learning_rate = st.number_input("Enter learning rate", min_value=0.01, max_value=1.0, value=0.1)
                self.reg_model.xgbregressor_regression(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
            elif algorithm_option == "Stacking Regressor":
                self.reg_model.stackingregressor_regression()
                
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
                self.cls_model.logistic_regression(C=C, max_iter=max_iter)
            elif algorithm_option == "Random Forest Classifier":
                n_estimators = st.number_input("Enter number of estimators", min_value=10, max_value=1000, value=100)
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=15)
                max_features = st.number_input("Enter max features", min_value=0.1, max_value=1.0, value=0.75)
                self.cls_model.random_forest_classifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
            elif algorithm_option == "Decision Tree Classifier":
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=15)
                min_samples_split = st.number_input("Enter min samples split", min_value=2, max_value=10, value=2)
                self.cls_model.decision_tree_classifier(max_depth=max_depth, min_samples_split=min_samples_split)
            elif algorithm_option == "KNN Classifier":
                n_neighbors = st.number_input("Enter number of neighbors", min_value=1, max_value=50, value=5)
                algorithm = st.selectbox("Select algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
                self.cls_model.knn_classifier(n_neighbors=n_neighbors, algorithm=algorithm)
            elif algorithm_option == "Support Vector Classifier":
                C = st.number_input("Enter regularization parameter (C)", min_value=0.01, max_value=100.0, value=1.0)
                kernel = st.selectbox("Select kernel", ["linear", "poly", "rbf", "sigmoid"])
                self.cls_model.support_vector_classifier(C=C, kernel=kernel)
            elif algorithm_option == "Gradient Boosting Classifier":
                n_estimators = st.number_input("Enter number of estimators", min_value=10, max_value=1000, value=100)
                learning_rate = st.number_input("Enter learning rate", min_value=0.01, max_value=1.0, value=0.1)
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=3)
                self.cls_model.gradient_boosting_classifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            elif algorithm_option == "AdaBoost Classifier":
                n_estimators = st.number_input("Enter number of estimators", min_value=10, max_value=1000, value=50)
                learning_rate = st.number_input("Enter learning rate", min_value=0.01, max_value=1.0, value=1.0)
                self.cls_model.adaboost_classifier(n_estimators=n_estimators, learning_rate=learning_rate)
            elif algorithm_option == "XGBoost Classifier":
                n_estimators = st.number_input("Enter number of estimators", min_value=10, max_value=1000, value=100)
                max_depth = st.number_input("Enter max depth", min_value=1, max_value=100, value=3)
                learning_rate = st.number_input("Enter learning rate", min_value=0.01, max_value=1.0, value=0.1)
                self.cls_model.xgboost_classifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
            elif algorithm_option == "Stacking Classifier":
                self.cls_model.stacking_classifier()