import streamlit as st
import optuna
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from show_code import ShowCode

class BestParam:
    def __init__(self, data):
        self.data = data
        self.view_code = ShowCode()
        self.view_code.set_target_class(BestParam)
        self.n_trials = 30  # Default number of trials
    
    def optuna_search(self, X_train, y_train, create_model, param_distributions, scoring='accuracy', cv=5, n_trials=10):
        """
        Use Optuna to find the best hyperparameters
        
        Parameters:
        -----------
        X_train : features training data
        y_train : target training data
        create_model : function that creates and returns a model instance with given parameters
        param_distributions : dictionary of parameter names and their possible distributions
        scoring : metric to evaluate models
        cv : number of cross-validation folds
        n_trials : number of optimization trials
        
        Returns:
        --------
        best_model : model instance with the best found parameters
        """
        direction = "maximize" if scoring in ['accuracy', 'r2', 'f1'] else "minimize"
        
        def objective(trial):
            # Create a set of parameters to evaluate
            params = {}
            for param_name, param_config in param_distributions.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'loguniform':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
            
            # Create and evaluate model with these parameters
            model = create_model(**params)
            
            if scoring == 'neg_mean_squared_error':
                scores = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
                return np.mean(scores)  # Negative MSE, higher is better
            else:
                scores = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
                return np.mean(scores)
        
        # Create and run the study
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        
        # Get the best parameters and create the best model
        best_params = study.best_params
        best_model = create_model(**best_params)
        best_model.fit(X_train, y_train)
        
        # Display the results
        st.write("Best Parameters:", best_params)
        st.write("Best Score:", study.best_value)
        
        # Store the study in session state for visualization
        if 'current_study' not in st.session_state:
            st.session_state.current_study = study
        
        # Store the parameter distributions in session state
        st.session_state.param_distributions = param_distributions
        
        # Visualization using Plotly
        st.subheader("Parameter Importance")
        try:
            # Get parameter importances
            importances = optuna.importance.get_param_importances(study)
            
            # Create a plotly figure for parameter importance
            param_names = list(importances.keys())
            importance_values = list(importances.values())
            
            fig_importance = go.Figure(data=[
                go.Bar(x=param_names, y=importance_values)
            ])
            
            fig_importance.update_layout(
                title="Parameter Importance",
                xaxis_title="Parameter",
                yaxis_title="Importance",
                height=500
            )
            
            st.plotly_chart(fig_importance)
        except Exception as e:
            st.write(f"Could not plot parameter importance: {str(e)}")
        
        # Plot optimization history
        st.subheader("Optimization History")
        try:
            # Get optimization history
            values = [trial.value for trial in study.trials]
            best_values = [study.best_value for _ in range(len(study.trials))]
            
            fig_history = go.Figure()
            
            # Add value trace
            fig_history.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode='markers',
                    name='Trial Value',
                    marker=dict(color='blue', size=8)
                )
            )
            
            # Add best value trace
            fig_history.add_trace(
                go.Scatter(
                    x=list(range(len(best_values))),
                    y=best_values,
                    mode='lines',
                    name='Best Value',
                    line=dict(color='red', width=2)
                )
            )
            
            fig_history.update_layout(
                title="Optimization History",
                xaxis_title="Trial",
                yaxis_title="Objective Value",
                height=500
            )
            
            st.plotly_chart(fig_history)
        except Exception as e:
            st.write(f"Could not plot optimization history: {str(e)}")
        
        return best_model

    def linear_regression(self, X_train, y_train):
        try:
            def create_model(fit_intercept):
                return LinearRegression(fit_intercept=fit_intercept)
            
            param_distributions = {
                'fit_intercept': {'type': 'categorical', 'values': [True, False]}
            }
            
            best_model = self.optuna_search(
                X_train, 
                y_train, 
                create_model, 
                param_distributions, 
                scoring='neg_mean_squared_error',
                n_trials=self.n_trials
            )
            return best_model
        except Exception as e:
            st.error(f"Error occurred during linear regression parameter tuning: {str(e)}")
            return None

    def random_forest_regression(self, X_train, y_train, param_distributions=None, scoring='neg_mean_squared_error'):
        try:
            if param_distributions is None:
                param_distributions = {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                    'max_depth': {'type': 'int', 'low': 5, 'high': 30},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                    'max_features': {'type': 'categorical', 'values': ['sqrt', 'log2', None]}
                }
            
            def create_model(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
                return RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=0
                )
            
            best_model = self.optuna_search(
                X_train, 
                y_train, 
                create_model, 
                param_distributions, 
                scoring=scoring,
                n_trials=self.n_trials
            )
            return best_model
        except Exception as e:
            st.error(f"Error occurred during random forest regression parameter tuning: {str(e)}")
            return None

    def random_forest_classifier(self, X_train, y_train, param_distributions=None, scoring='accuracy'):
        try:
            if param_distributions is None:
                param_distributions = {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                    'max_depth': {'type': 'int', 'low': 5, 'high': 30},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                    'max_features': {'type': 'categorical', 'values': ['sqrt', 'log2', None]},
                    'class_weight': {'type': 'categorical', 'values': ['balanced', None]}
                }
            
            def create_model(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, class_weight):
                return RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    class_weight=class_weight,
                    random_state=0
                )
            
            best_model = self.optuna_search(
                X_train, 
                y_train, 
                create_model, 
                param_distributions, 
                scoring=scoring,
                n_trials=self.n_trials
            )
            return best_model
        except Exception as e:
            st.error(f"Error occurred during random forest classifier parameter tuning: {str(e)}")
            return None

    def gradient_boosting_classifier(self, X_train, y_train, param_distributions=None, scoring='accuracy'):
        try:
            if param_distributions is None:
                param_distributions = {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                    'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
                }
            
            def create_model(n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample):
                return GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    subsample=subsample,
                    random_state=0
                )
            
            best_model = self.optuna_search(
                X_train, 
                y_train, 
                create_model, 
                param_distributions, 
                scoring=scoring,
                n_trials=self.n_trials
            )
            return best_model
        except Exception as e:
            st.error(f"Error occurred during gradient boosting classifier parameter tuning: {str(e)}")
            return None

    def select_hyper(self):
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Select Best Parameters (Optuna)</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>", unsafe_allow_html=True)
        st.dataframe(self.data, width=800)

        target_column = st.sidebar.selectbox("Select Target Column", self.data.columns)
        feature_column = self.data.drop(columns=[target_column])
        
        options = [
            "Linear Regression",
            "Random Forest Regression",
            "Random Forest Classifier",
            "Gradient Boosting Classifier"
        ]
        
        option = st.sidebar.selectbox("Select a Model for Parameter Tuning", options)
        num_trials = st.sidebar.slider("Number of Optuna Trials", min_value=10, max_value=100, value=30)
        
        if 'tuned_model' not in st.session_state:
            st.session_state.tuned_model = None
        
        if 'selected_option' not in st.session_state:
            st.session_state.selected_option = None

        if 'show_code' not in st.session_state:
            st.session_state.show_code = False
            
        if 'is_tuned' not in st.session_state:
            st.session_state.is_tuned = False

        tune_button = st.button("Find Best Parameters")

        # Only run the tuning when the button is pressed (not when show_code changes)
        if tune_button:
            # Update the session state to indicate we are tuning a new model
            st.session_state.is_tuned = True
            self.n_trials = num_trials
            
            if option == "Linear Regression":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Linear Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                st.session_state.tuned_model = self.linear_regression(feature_column, self.data[target_column])
            elif option == "Random Forest Regression":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Random Forest Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                st.session_state.tuned_model = self.random_forest_regression(feature_column, self.data[target_column])
            elif option == "Random Forest Classifier":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Random Forest Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                st.session_state.tuned_model = self.random_forest_classifier(feature_column, self.data[target_column])
            elif option == "Gradient Boosting Classifier":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Gradient Boosting Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                st.session_state.tuned_model = self.gradient_boosting_classifier(feature_column, self.data[target_column])
            
            st.session_state.selected_option = option
        
        # Display the existing results if we've already tuned a model
        elif st.session_state.is_tuned and st.session_state.tuned_model is not None:
            if st.session_state.selected_option == "Linear Regression":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Linear Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
            elif st.session_state.selected_option == "Random Forest Regression":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Random Forest Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
            elif st.session_state.selected_option == "Random Forest Classifier":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Random Forest Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
            elif st.session_state.selected_option == "Gradient Boosting Classifier":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Gradient Boosting Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
            
            # Display visualizations if a study exists in session state
            if 'current_study' in st.session_state:
                # Create the visualizations again using the stored study
                study = st.session_state.current_study
                
                st.subheader("Parameter Importance")
                try:
                    # Get parameter importances
                    importances = optuna.importance.get_param_importances(study)
                    
                    # Create a plotly figure for parameter importance
                    param_names = list(importances.keys())
                    importance_values = list(importances.values())
                    
                    fig_importance = go.Figure(data=[
                        go.Bar(x=param_names, y=importance_values)
                    ])
                    
                    fig_importance.update_layout(
                        title="Parameter Importance",
                        xaxis_title="Parameter",
                        yaxis_title="Importance",
                        height=500
                    )
                    
                    st.plotly_chart(fig_importance)
                except Exception as e:
                    st.write(f"Could not plot parameter importance: {str(e)}")
                
                st.subheader("Optimization History")
                try:
                    # Get optimization history
                    values = [trial.value for trial in study.trials]
                    best_values = [study.best_value for _ in range(len(study.trials))]
                    
                    fig_history = go.Figure()
                    
                    # Add value trace
                    fig_history.add_trace(
                        go.Scatter(
                            x=list(range(len(values))),
                            y=values,
                            mode='markers',
                            name='Trial Value',
                            marker=dict(color='blue', size=8)
                        )
                    )
                    
                    # Add best value trace
                    fig_history.add_trace(
                        go.Scatter(
                            x=list(range(len(best_values))),
                            y=best_values,
                            mode='lines',
                            name='Best Value',
                            line=dict(color='red', width=2)
                        )
                    )
                    
                    fig_history.update_layout(
                        title="Optimization History",
                        xaxis_title="Trial",
                        yaxis_title="Objective Value",
                        height=500
                    )
                    
                    st.plotly_chart(fig_history)
                except Exception as e:
                    st.write(f"Could not plot optimization history: {str(e)}")

        # Show the results and code toggle only if a model has been tuned
        if st.session_state.is_tuned and st.session_state.tuned_model is not None:
            # Create a container for code display to maintain state between show/hide
            code_container = st.container()
            
            # Toggle for showing code - won't trigger retuning
            show_code = st.checkbox('Show Code', value=st.session_state.show_code)
            st.session_state.show_code = show_code
            
            if show_code:
                with code_container:
                    # Store the current parameters before displaying code to preserve them
                    if 'param_distributions' in st.session_state:
                        param_distributions = st.session_state.param_distributions
                        
                    if st.session_state.selected_option == "Linear Regression":
                        self.view_code._display_code('linear_regression')
                    elif st.session_state.selected_option == "Random Forest Regression":
                        self.view_code._display_code('random_forest_regression')
                    elif st.session_state.selected_option == "Random Forest Classifier":
                        self.view_code._display_code('random_forest_classifier')
                    elif st.session_state.selected_option == "Gradient Boosting Classifier":
                        self.view_code._display_code('gradient_boosting_classifier')