import streamlit as st
import optuna
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from show_code import ShowCode
import pandas as pd
from optuna.visualization import plot_param_importances, plot_optimization_history, plot_contour, plot_parallel_coordinate

class BestParam:
    def __init__(self, data):
        self.data = data
        self.view_code = ShowCode()
        self.view_code.set_target_class(BestParam)
        self.n_trials = 30  # Default number of trials
        
        # Initialize custom parameter dictionaries for each model
        if 'custom_params' not in st.session_state:
            st.session_state.custom_params = {
                "Linear Regression": {},
                "Random Forest Regression": {},
                "Random Forest Classifier": {},
                "Gradient Boosting Classifier": {},
                "Logistic Regression": {},
                "AdaBoost Classifier": {},
                "Decision Tree Classifier": {},
                "KNN Classifier": {},
                "Stacking Classifier": {},
                "KNN Regression": {},
                "Decision Tree Regression": {},
                "Gradient Boosting Regression": {},
                "AdaBoost Regression": {},
                "Extra Trees Regression": {},
                "SVR Regression": {},
                "XGBoost Regression": {},
                "Stacking Regression": {}
            }

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
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'], step=param_config.get('step', 1))
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], step=param_config.get('step', None))
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
        study_name = f"study_{scoring}_{n_trials}"
        study = optuna.create_study(direction=direction, study_name=study_name)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
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
        else:
            st.session_state.current_study = study
        
        # Store the parameter distributions in session state
        st.session_state.param_distributions = param_distributions
        
        # Advanced visualizations with tabs
        st.subheader("Optuna Visualizations")
        tabs = st.tabs(["Parameter Importance", "Optimization History", "Parallel Coordinate", "Contour Plot", "Trial Data"])
        
        with tabs[0]:  # Parameter Importance
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
                
                st.plotly_chart(fig_importance, use_container_width=True)
            except Exception as e:
                st.write(f"Could not plot parameter importance: {str(e)}")
        
        with tabs[1]:  # Optimization History
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
                
                st.plotly_chart(fig_history, use_container_width=True)
            except Exception as e:
                st.write(f"Could not plot optimization history: {str(e)}")
                
        with tabs[2]:  # Parallel Coordinate Plot
            try:
                # Create data for parallel coordinate plot
                trial_data = []
                for trial in study.trials:
                    if trial.state == optuna.trial.TrialState.COMPLETE:
                        params = trial.params.copy()
                        params['value'] = trial.value
                        trial_data.append(params)
                
                if trial_data:
                    df = pd.DataFrame(trial_data)
                    
                    fig = go.Figure(data=
                        go.Parcoords(
                            line=dict(
                                color=df['value'],
                                colorscale='Viridis',
                                showscale=True
                            ),
                            dimensions=[
                                dict(
                                    range=[df[col].min(), df[col].max()],
                                    label=col,
                                    values=df[col]
                                ) for col in df.columns
                            ]
                        )
                    )
                    
                    fig.update_layout(
                        title="Parallel Coordinate Plot",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Not enough complete trials for parallel coordinate plot")
            except Exception as e:
                st.write(f"Could not create parallel coordinate plot: {str(e)}")
                
        with tabs[3]:  # Contour Plot
            try:
                if len(param_distributions) >= 2:
                    # Get the two most important parameters
                    importances = optuna.importance.get_param_importances(study)
                    top_params = list(importances.keys())[:2] if len(importances) >= 2 else list(param_distributions.keys())[:2]
                    
                    # Create data for contour plot
                    param_x = top_params[0]
                    param_y = top_params[1]
                    
                    x_values = []
                    y_values = []
                    objective_values = []
                    
                    for trial in study.trials:
                        if trial.state == optuna.trial.TrialState.COMPLETE and param_x in trial.params and param_y in trial.params:
                            x_values.append(trial.params[param_x])
                            y_values.append(trial.params[param_y])
                            objective_values.append(trial.value)
                    
                    if x_values and y_values and objective_values:
                        fig = go.Figure(data=
                            go.Contour(
                                x=x_values,
                                y=y_values,
                                z=objective_values,
                                contours=dict(
                                    coloring='heatmap',
                                    showlabels=True
                                ),
                                colorscale='Viridis'
                            )
                        )
                        
                        fig.update_layout(
                            title=f"Contour Plot: {param_x} vs {param_y}",
                            xaxis_title=param_x,
                            yaxis_title=param_y,
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Not enough data for contour plot")
                else:
                    st.write("Need at least 2 parameters for contour plot")
            except Exception as e:
                st.write(f"Could not create contour plot: {str(e)}")
                
        with tabs[4]:  # Trial Data Table
            try:
                trials_data = []
                for i, trial in enumerate(study.trials):
                    if trial.state == optuna.trial.TrialState.COMPLETE:
                        trial_info = {"Trial": i, "Value": trial.value}
                        trial_info.update(trial.params)
                        trials_data.append(trial_info)
                
                if trials_data:
                    df = pd.DataFrame(trials_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.write("No completed trials available")
            except Exception as e:
                st.write(f"Could not display trial data: {str(e)}")
        
        return best_model

    def customize_param_distributions(self, model_type):
        """
        Allow users to customize hyperparameter search spaces
        """
        st.subheader("Customize Hyperparameter Search Space")
        
        default_params = {}
        if model_type == "Linear Regression":
            default_params = {
                'fit_intercept': {'type': 'categorical', 'values': [True, False]}
            }
        elif model_type == "Random Forest Regression":
            default_params = {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 5, 'high': 30},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'values': ['sqrt', 'log2', None]}
            }
        elif model_type == "Random Forest Classifier":
            default_params = {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 5, 'high': 30},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'values': ['sqrt', 'log2', None]},
                'class_weight': {'type': 'categorical', 'values': ['balanced', None]}
            }
        elif model_type == "Gradient Boosting Classifier":
            default_params = {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
            }
        elif model_type == "Logistic Regression":
            default_params = {
                'C': {'type': 'loguniform', 'low': 0.01, 'high': 10.0},
                'penalty': {'type': 'categorical', 'values': ['l1', 'l2', 'elasticnet', None]},
                'solver': {'type': 'categorical', 'values': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
                'max_iter': {'type': 'int', 'low': 100, 'high': 1000},
                'l1_ratio': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'class_weight': {'type': 'categorical', 'values': ['balanced', None]}
            }
        elif model_type == "Decision Tree Classifier":
            default_params = {
                'max_depth': {'type': 'int', 'low': 3, 'high': 30},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 20},
                'max_features': {'type': 'categorical', 'values': ['sqrt', 'log2', None]},
                'criterion': {'type': 'categorical', 'values': ['gini', 'entropy']},
                'class_weight': {'type': 'categorical', 'values': ['balanced', None]}
            }
        elif model_type == "AdaBoost Classifier":
            default_params = {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 2.0},
                'algorithm': {'type': 'categorical', 'values': ['SAMME']}
            }
        elif model_type == "KNN Classifier":
            default_params = {
                'n_neighbors': {'type': 'int', 'low': 1, 'high': 30},
                'weights': {'type': 'categorical', 'values': ['uniform', 'distance']},
                'algorithm': {'type': 'categorical', 'values': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                'leaf_size': {'type': 'int', 'low': 10, 'high': 100},
                'p': {'type': 'int', 'low': 1, 'high': 2}
            }
        elif model_type == "Stacking Classifier":
            default_params = {
                'rf_n_estimators': {'type': 'int', 'low': 50, 'high': 200},
                'rf_max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'gb_n_estimators': {'type': 'int', 'low': 50, 'high': 200},
                'gb_learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                'lr_C': {'type': 'loguniform', 'low': 0.01, 'high': 10.0},
                'final_estimator': {'type': 'categorical', 'values': ['lr', 'rf']}
            }
        
        # Get custom params from session state or use defaults
        if model_type not in st.session_state.custom_params or not st.session_state.custom_params[model_type]:
            st.session_state.custom_params[model_type] = default_params.copy()
        
        custom_params = st.session_state.custom_params[model_type]
        
        use_custom = st.checkbox("Customize Parameters", value=bool(custom_params != default_params))
        
        if use_custom:
            param_tabs = st.tabs(list(custom_params.keys()))
            
            for i, (param_name, param_config) in enumerate(custom_params.items()):
                with param_tabs[i]:
                    param_type = st.selectbox(
                        f"Parameter Type for {param_name}", 
                        ['categorical', 'int', 'float', 'loguniform'],
                        index=['categorical', 'int', 'float', 'loguniform'].index(param_config['type']),
                        key=f"{model_type}_{param_name}_type"
                    )
                    
                    # Update config based on selected type
                    custom_params[param_name]['type'] = param_type
                    
                    if param_type == 'categorical':
                        # Handle categorical values as a comma-separated string
                        if 'values' in param_config:
                            values_str = ", ".join([str(v) for v in param_config['values']])
                        else:
                            values_str = "True, False"
                            
                        values_input = st.text_input(
                            f"Values for {param_name} (comma-separated)", 
                            value=values_str,
                            key=f"{model_type}_{param_name}_values"
                        )
                        
                        # Parse the input string back to a list
                        values = []
                        for v in values_input.split(','):
                            v = v.strip()
                            if v.lower() == 'true':
                                values.append(True)
                            elif v.lower() == 'false':
                                values.append(False)
                            elif v.lower() == 'none':
                                values.append(None)
                            elif v.isdigit():
                                values.append(int(v))
                            elif v.replace('.', '', 1).isdigit():
                                values.append(float(v))
                            else:
                                values.append(v)
                                
                        custom_params[param_name]['values'] = values
                        
                    else:  # numeric types (int, float, loguniform)
                        # Set default low/high values if not present
                        low = param_config.get('low', 0)
                        high = param_config.get('high', 100)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            low = st.number_input(
                                f"Minimum value for {param_name}", 
                                value=float(low),
                                key=f"{model_type}_{param_name}_low"
                            )
                        with col2:
                            high = st.number_input(
                                f"Maximum value for {param_name}", 
                                value=float(high),
                                key=f"{model_type}_{param_name}_high"
                            )
                            
                        custom_params[param_name]['low'] = low
                        custom_params[param_name]['high'] = high
                        
                        # Add step for int and float
                        if param_type in ['int', 'float']:
                            # Set default step value based on parameter type
                            default_step = 1 if param_type == 'int' else 0.01
                            step = param_config.get('step', default_step)
                            
                            # Ensure all numeric values have consistent types
                            if param_type == 'int':
                                # For integer parameters, convert all values to int
                                step = int(step)
                                step_min = 1
                            else:
                                # For float parameters, convert all values to float
                                step = float(step)
                                step_min = 0.0001
                                
                            step = st.number_input(
                                f"Step size for {param_name}", 
                                value=step,
                                min_value=step_min,
                                key=f"{model_type}_{param_name}_step"
                            )
                            custom_params[param_name]['step'] = step
            
            # Option to add a new parameter
            new_param = st.text_input("Add new parameter (name):")
            if new_param and new_param not in custom_params:
                custom_params[new_param] = {'type': 'int', 'low': 1, 'high': 10}
                st.success(f"Added {new_param}. Please customize it in the tabs above.")
                st.experimental_rerun()
            
            # Option to remove a parameter
            params_to_remove = st.multiselect("Remove parameters:", list(custom_params.keys()))
            if params_to_remove:
                for param in params_to_remove:
                    if param in custom_params:
                        del custom_params[param]
                st.success(f"Removed {', '.join(params_to_remove)}.")
                st.experimental_rerun()
            
            # Save custom parameters to session state
            st.session_state.custom_params[model_type] = custom_params
            
            st.info("Your custom parameters have been saved and will be used for hyperparameter tuning.")
            
        return st.session_state.custom_params[model_type] if use_custom else default_params

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
            "Decision Tree Regression",
            "KNN Regression",
            "Gradient Boosting Regression",
            "AdaBoost Regression",
            "Extra Trees Regression",
            "SVR Regression",
            "XGBoost Regression",
            "Stacking Regression",
            "Random Forest Classifier",
            "Gradient Boosting Classifier",
            "Logistic Regression",
            "Decision Tree Classifier",
            "AdaBoost Classifier",
            "KNN Classifier",
            "Stacking Classifier"
        ]
        
        option = st.sidebar.selectbox("Select a Model for Parameter Tuning", options)
        num_trials = st.sidebar.slider("Number of Optuna Trials", min_value=10, max_value=100, value=30)
        
        # Define scoring metrics based on task type
        if option in ["Linear Regression", "Random Forest Regression"]:
            scoring_options = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
            default_idx = 0
        else:  # Classification models
            scoring_options = ["accuracy", "f1", "precision", "recall", "roc_auc"]
            default_idx = 0
            
        scoring_metric = st.sidebar.selectbox(
            "Select Scoring Metric", 
            scoring_options, 
            index=default_idx
        )
        
        cross_val_folds = st.sidebar.slider("Cross-Validation Folds", min_value=2, max_value=10, value=5)
        
        # Get customized parameters
        custom_params = self.customize_param_distributions(option)
        
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
            
            with st.spinner(f"Tuning {option} with Optuna - Running {num_trials} trials..."):
                if option == "Linear Regression":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>Linear Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.linear_regression(feature_column, self.data[target_column])
                    
                elif option == "Random Forest Regression":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>Random Forest Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.random_forest_regression(
                        feature_column, 
                        self.data[target_column], 
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                
                elif option == "Decision Tree Regression":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>Decision Tree Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.decision_tree_regressor(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                
                elif option == "KNN Regression":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>KNN Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.knn_regressor(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                
                elif option == "Gradient Boosting Regression":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>Gradient Boosting Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.gradient_boosting_regressor(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                
                elif option == "AdaBoost Regression":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>AdaBoost Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.adaboost_regressor(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                
                elif option == "Extra Trees Regression":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>Extra Trees Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.extra_trees_regressor(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                
                elif option == "SVR Regression":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>SVR Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.svr_regressor(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                
                elif option == "XGBoost Regression":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>XGBoost Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.xgb_regressor(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                
                elif option == "Stacking Regression":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>Stacking Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.stacking_regressor(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                    
                elif option == "Random Forest Classifier":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>Random Forest Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.random_forest_classifier(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                    
                elif option == "Gradient Boosting Classifier":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>Gradient Boosting Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.gradient_boosting_classifier(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                    
                elif option == "Logistic Regression":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>Logistic Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.logistic_regression_classifier(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                    
                elif option == "Decision Tree Classifier":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>Decision Tree Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.decision_tree_classifier(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                    
                elif option == "AdaBoost Classifier":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>AdaBoost Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.adaboost_classifier(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                    
                elif option == "KNN Classifier":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>KNN Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.knn_classifier(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
                    
                elif option == "Stacking Classifier":
                    st.markdown("<h2 style='text-align: center; font-size: 25px;'>Stacking Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
                    st.session_state.tuned_model = self.stacking_classifier(
                        feature_column, 
                        self.data[target_column],
                        param_distributions=custom_params,
                        scoring=scoring_metric
                    )
            
            st.session_state.selected_option = option
            st.success(f"Completed hyperparameter tuning for {option}!")
        
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
            elif st.session_state.selected_option == "Logistic Regression":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Logistic Regression HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
            elif st.session_state.selected_option == "Decision Tree Classifier":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Decision Tree Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
            elif st.session_state.selected_option == "AdaBoost Classifier":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>AdaBoost Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
            elif st.session_state.selected_option == "KNN Classifier":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>KNN Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
            elif st.session_state.selected_option == "Stacking Classifier":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Stacking Classifier HyperParameter Tuning with Optuna</h2>", unsafe_allow_html=True)
            
            # Display visualizations if a study exists in session state
            if 'current_study' in st.session_state:
                # Create the visualizations again using the stored study
                study = st.session_state.current_study
                
                # We'll display advanced visualizations with tabs
                st.subheader("Optuna Visualizations")
                tabs = st.tabs(["Parameter Importance", "Optimization History", "Parallel Coordinate", "Contour Plot", "Trial Data"])
                
                with tabs[0]:  # Parameter Importance
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
                        
                        st.plotly_chart(fig_importance, use_container_width=True)
                    except Exception as e:
                        st.write(f"Could not plot parameter importance: {str(e)}")
                
                with tabs[1]:  # Optimization History
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
                        
                        st.plotly_chart(fig_history, use_container_width=True)
                    except Exception as e:
                        st.write(f"Could not plot optimization history: {str(e)}")
                
                with tabs[2]:  # Parallel Coordinate Plot
                    try:
                        # Create data for parallel coordinate plot
                        trial_data = []
                        for trial in study.trials:
                            if trial.state == optuna.trial.TrialState.COMPLETE:
                                params = trial.params.copy()
                                params['value'] = trial.value
                                trial_data.append(params)
                        
                        if trial_data:
                            df = pd.DataFrame(trial_data)
                            
                            fig = go.Figure(data=
                                go.Parcoords(
                                    line=dict(
                                        color=df['value'],
                                        colorscale='Viridis',
                                        showscale=True
                                    ),
                                    dimensions=[
                                        dict(
                                            range=[df[col].min(), df[col].max()],
                                            label=col,
                                            values=df[col]
                                        ) for col in df.columns
                                    ]
                                )
                            )
                            
                            fig.update_layout(
                                title="Parallel Coordinate Plot",
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("Not enough complete trials for parallel coordinate plot")
                    except Exception as e:
                        st.write(f"Could not create parallel coordinate plot: {str(e)}")
                
                with tabs[3]:  # Contour Plot
                    try:
                        if 'param_distributions' in st.session_state and len(st.session_state.param_distributions) >= 2:
                            # Get the two most important parameters
                            importances = optuna.importance.get_param_importances(study)
                            top_params = list(importances.keys())[:2] if len(importances) >= 2 else list(st.session_state.param_distributions.keys())[:2]
                            
                            # Create data for contour plot
                            param_x = top_params[0]
                            param_y = top_params[1]
                            
                            x_values = []
                            y_values = []
                            objective_values = []
                            
                            for trial in study.trials:
                                if trial.state == optuna.trial.TrialState.COMPLETE and param_x in trial.params and param_y in trial.params:
                                    x_values.append(trial.params[param_x])
                                    y_values.append(trial.params[param_y])
                                    objective_values.append(trial.value)
                            
                            if x_values and y_values and objective_values:
                                fig = go.Figure(data=
                                    go.Contour(
                                        x=x_values,
                                        y=y_values,
                                        z=objective_values,
                                        contours=dict(
                                            coloring='heatmap',
                                            showlabels=True
                                        ),
                                        colorscale='Viridis'
                                    )
                                )
                                
                                fig.update_layout(
                                    title=f"Contour Plot: {param_x} vs {param_y}",
                                    xaxis_title=param_x,
                                    yaxis_title=param_y,
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.write("Not enough data for contour plot")
                        else:
                            st.write("Need at least 2 parameters for contour plot")
                    except Exception as e:
                        st.write(f"Could not create contour plot: {str(e)}")
                
                with tabs[4]:  # Trial Data Table
                    try:
                        trials_data = []
                        for i, trial in enumerate(study.trials):
                            if trial.state == optuna.trial.TrialState.COMPLETE:
                                trial_info = {"Trial": i, "Value": trial.value}
                                trial_info.update(trial.params)
                                trials_data.append(trial_info)
                        
                        if trials_data:
                            df = pd.DataFrame(trials_data)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.write("No completed trials available")
                    except Exception as e:
                        st.write(f"Could not display trial data: {str(e)}")

        # Show the results and code toggle only if a model has been tuned
        if st.session_state.is_tuned and st.session_state.tuned_model is not None:
            # Create a container for code display to maintain state between show/hide
            code_container = st.container()
            
            # Export model button
            if st.button("Export Tuned Model"):
                import pickle
                import io
                from datetime import datetime
                
                model_pickle = pickle.dumps(st.session_state.tuned_model)
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"{st.session_state.selected_option.replace(' ', '_').lower()}_{now}.pkl"
                
                st.download_button(
                    label="Download Model",
                    data=model_pickle,
                    file_name=model_name,
                    mime="application/octet-stream"
                )
            
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
                    elif st.session_state.selected_option == "Logistic Regression":
                        self.view_code._display_code('logistic_regression_classifier')
                    elif st.session_state.selected_option == "Decision Tree Classifier":
                        self.view_code._display_code('decision_tree_classifier')
                    elif st.session_state.selected_option == "AdaBoost Classifier":
                        self.view_code._display_code('adaboost_classifier')
                    elif st.session_state.selected_option == "KNN Classifier":
                        self.view_code._display_code('knn_classifier')
                    elif st.session_state.selected_option == "Stacking Classifier":
                        self.view_code._display_code('stacking_classifier')

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
        """
        Perform hyperparameter tuning for a Random Forest Classifier using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'accuracy')
            
        Returns:
        --------
        best_model : trained RandomForestClassifier with best parameters
        """
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
        """
        Perform hyperparameter tuning for a Gradient Boosting Classifier using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'accuracy')
            
        Returns:
        --------
        best_model : trained GradientBoostingClassifier with best parameters
        """
        try:
            # Use default parameter distributions if none provided
            if param_distributions is None:
                param_distributions = {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                    'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
                }
            
            # Define model creation function that matches the parameters in param_distributions
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
            
            # Use optuna_search to find the best model
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

    def logistic_regression_classifier(self, X_train, y_train, param_distributions=None, scoring='accuracy'):
        """
        Perform hyperparameter tuning for a Logistic Regression Classifier using Optuna

        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'accuracy')

        Returns:
        --------
        best_model : trained LogisticRegression with best parameters
        """
        try:
            if param_distributions is None:
                param_distributions = {
                    'C': {'type': 'loguniform', 'low': 0.01, 'high': 10.0},
                    'penalty': {'type': 'categorical', 'values': ['l1', 'l2', 'elasticnet', None]},
                    'solver': {'type': 'categorical', 'values': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
                    'max_iter': {'type': 'int', 'low': 100, 'high': 1000},
                    'l1_ratio': {'type': 'float', 'low': 0.0, 'high': 1.0}  # Include l1_ratio for elasticnet
                }

            def create_model(C, penalty, solver, max_iter, l1_ratio=None):
                if penalty == 'elasticnet' and solver != 'saga':
                    raise ValueError("Elasticnet penalty requires 'saga' solver.")
                return LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter, l1_ratio=l1_ratio)

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
            st.error(f"Error occurred during logistic regression parameter tuning: {str(e)}")
            return None

    def decision_tree_classifier(self, X_train, y_train, param_distributions=None, scoring='accuracy'):
        """
        Perform hyperparameter tuning for a Decision Tree Classifier using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'accuracy')
            
        Returns:
        --------
        best_model : trained DecisionTreeClassifier with best parameters
        """
        try:
            if param_distributions is None:
                param_distributions = {
                    'max_depth': {'type': 'int', 'low': 3, 'high': 30},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 20},
                    'max_features': {'type': 'categorical', 'values': ['sqrt', 'log2', None]},
                    'criterion': {'type': 'categorical', 'values': ['gini', 'entropy']},
                    'class_weight': {'type': 'categorical', 'values': ['balanced', None]}
                }
            
            def create_model(max_depth, min_samples_split, min_samples_leaf, max_features, criterion, class_weight):
                return DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    criterion=criterion,
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
            st.error(f"Error occurred during decision tree classifier parameter tuning: {str(e)}")
            return None
            
    def adaboost_classifier(self, X_train, y_train, param_distributions=None, scoring='accuracy'):
        """
        Perform hyperparameter tuning for an AdaBoost Classifier using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'accuracy')
            
        Returns:
        --------
        best_model : trained AdaBoostClassifier with best parameters
        """
        try:
            if param_distributions is None:
                param_distributions = {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                    'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 2.0},
                    'algorithm': {'type': 'categorical', 'values': ['SAMME']}
                }
            
            def create_model(n_estimators, learning_rate, algorithm):
                return AdaBoostClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    algorithm=algorithm,
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
            st.error(f"Error occurred during AdaBoost classifier parameter tuning: {str(e)}")
            return None

    def knn_classifier(self, X_train, y_train, param_distributions=None, scoring='accuracy'):
        """
        Perform hyperparameter tuning for a K-Nearest Neighbors Classifier using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'accuracy')
            
        Returns:
        --------
        best_model : trained KNeighborsClassifier with best parameters
        """
        try:
            if param_distributions is None:
                param_distributions = {
                    'n_neighbors': {'type': 'int', 'low': 1, 'high': 30},
                    'weights': {'type': 'categorical', 'values': ['uniform', 'distance']},
                    'algorithm': {'type': 'categorical', 'values': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                    'leaf_size': {'type': 'int', 'low': 10, 'high': 100},
                    'p': {'type': 'int', 'low': 1, 'high': 2}  # 1 for Manhattan, 2 for Euclidean
                }
            
            def create_model(n_neighbors, weights, algorithm, leaf_size, p):
                return KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    algorithm=algorithm,
                    leaf_size=leaf_size,
                    p=p
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
            st.error(f"Error occurred during KNN classifier parameter tuning: {str(e)}")
            return None
            
    def stacking_classifier(self, X_train, y_train, param_distributions=None, scoring='accuracy'):
        """
        Perform hyperparameter tuning for a Stacking Classifier using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'accuracy')
            
        Returns:
        --------
        best_model : trained StackingClassifier with best parameters
        """
        try:
            if param_distributions is None:
                param_distributions = {
                    'rf_n_estimators': {'type': 'int', 'low': 50, 'high': 200},
                    'rf_max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'gb_n_estimators': {'type': 'int', 'low': 50, 'high': 200},
                    'gb_learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                    'lr_C': {'type': 'loguniform', 'low': 0.01, 'high': 10.0},
                    'final_estimator': {'type': 'categorical', 'values': ['lr', 'rf']}
                }
            
            def create_model(rf_n_estimators, rf_max_depth, gb_n_estimators, gb_learning_rate, lr_C, final_estimator):
                # Create base estimators
                estimators = [
                    ('rf', RandomForestClassifier(
                        n_estimators=rf_n_estimators,
                        max_depth=rf_max_depth,
                        random_state=0
                    )),
                    ('gb', GradientBoostingClassifier(
                        n_estimators=gb_n_estimators,
                        learning_rate=gb_learning_rate,
                        random_state=0
                    )),
                    ('lr', LogisticRegression(C=lr_C, max_iter=500, random_state=0))
                ]
                
                # Create final estimator
                if final_estimator == 'lr':
                    final_est = LogisticRegression(random_state=0)
                else:  # rf
                    final_est = RandomForestClassifier(random_state=0)
                
                return StackingClassifier(
                    estimators=estimators,
                    final_estimator=final_est,
                    cv=5,
                    stack_method='auto',
                    n_jobs=-1
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
            st.error(f"Error occurred during Stacking classifier parameter tuning: {str(e)}")
            return None

    def decision_tree_regressor(self, X_train, y_train, param_distributions=None, scoring='neg_mean_squared_error'):
        """
        Perform hyperparameter tuning for a Decision Tree Regressor using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'neg_mean_squared_error')
            
        Returns:
        --------
        best_model : trained DecisionTreeRegressor with best parameters
        """
        try:
            if param_distributions is None:
                param_distributions = {
                    'max_depth': {'type': 'int', 'low': 3, 'high': 30},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 20},
                    'max_features': {'type': 'categorical', 'values': ['sqrt', 'log2', None]},
                    'criterion': {'type': 'categorical', 'values': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']}
                }
            
            def create_model(max_depth, min_samples_split, min_samples_leaf, max_features, criterion):
                return DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    criterion=criterion,
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
            st.error(f"Error occurred during decision tree regressor parameter tuning: {str(e)}")
            return None
            
    def knn_regressor(self, X_train, y_train, param_distributions=None, scoring='neg_mean_squared_error'):
        """
        Perform hyperparameter tuning for a K-Nearest Neighbors Regressor using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'neg_mean_squared_error')
            
        Returns:
        --------
        best_model : trained KNeighborsRegressor with best parameters
        """
        try:
            if param_distributions is None:
                param_distributions = {
                    'n_neighbors': {'type': 'int', 'low': 1, 'high': 30},
                    'weights': {'type': 'categorical', 'values': ['uniform', 'distance']},
                    'algorithm': {'type': 'categorical', 'values': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                    'leaf_size': {'type': 'int', 'low': 10, 'high': 100},
                    'p': {'type': 'int', 'low': 1, 'high': 2}  # 1 for Manhattan, 2 for Euclidean
                }
            
            def create_model(n_neighbors, weights, algorithm, leaf_size, p):
                return KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    algorithm=algorithm,
                    leaf_size=leaf_size,
                    p=p
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
            st.error(f"Error occurred during KNN regressor parameter tuning: {str(e)}")
            return None

    def gradient_boosting_regressor(self, X_train, y_train, param_distributions=None, scoring='neg_mean_squared_error'):
        """
        Perform hyperparameter tuning for a Gradient Boosting Regressor using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'neg_mean_squared_error')
            
        Returns:
        --------
        best_model : trained GradientBoostingRegressor with best parameters
        """
        try:
            if param_distributions is None:
                param_distributions = {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                    'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                    'max_features': {'type': 'categorical', 'values': ['sqrt', 'log2', None]}
                }
            
            def create_model(n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, max_features):
                return GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    subsample=subsample,
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
            st.error(f"Error occurred during gradient boosting regressor parameter tuning: {str(e)}")
            return None
            
    def adaboost_regressor(self, X_train, y_train, param_distributions=None, scoring='neg_mean_squared_error'):
        """
        Perform hyperparameter tuning for an AdaBoost Regressor using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'neg_mean_squared_error')
            
        Returns:
        --------
        best_model : trained AdaBoostRegressor with best parameters
        """
        try:
            if param_distributions is None:
                param_distributions = {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                    'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 2.0},
                    'loss': {'type': 'categorical', 'values': ['linear', 'square', 'exponential']}
                }
            
            def create_model(n_estimators, learning_rate, loss):
                return AdaBoostRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    loss=loss,
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
            st.error(f"Error occurred during AdaBoost regressor parameter tuning: {str(e)}")
            return None

    def extra_trees_regressor(self, X_train, y_train, param_distributions=None, scoring='neg_mean_squared_error'):
        """
        Perform hyperparameter tuning for an Extra Trees Regressor using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'neg_mean_squared_error')
            
        Returns:
        --------
        best_model : trained ExtraTreesRegressor with best parameters
        """
        try:
            if param_distributions is None:
                param_distributions = {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                    'max_depth': {'type': 'int', 'low': 5, 'high': 30},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                    'max_features': {'type': 'categorical', 'values': ['sqrt', 'log2', None]},
                    'bootstrap': {'type': 'categorical', 'values': [True, False]}
                }
            
            def create_model(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap):
                return ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=bootstrap,
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
            st.error(f"Error occurred during Extra Trees regressor parameter tuning: {str(e)}")
            return None
            
    def svr_regressor(self, X_train, y_train, param_distributions=None, scoring='neg_mean_squared_error'):
        """
        Perform hyperparameter tuning for a Support Vector Regressor using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'neg_mean_squared_error')
            
        Returns:
        --------
        best_model : trained SVR with best parameters
        """
        try:
            if param_distributions is None:
                param_distributions = {
                    'C': {'type': 'loguniform', 'low': 0.01, 'high': 100.0},
                    'epsilon': {'type': 'loguniform', 'low': 0.001, 'high': 1.0},
                    'kernel': {'type': 'categorical', 'values': ['linear', 'poly', 'rbf', 'sigmoid']},
                    'gamma': {'type': 'categorical', 'values': ['scale', 'auto']}
                }
            
            def create_model(C, epsilon, kernel, gamma):
                return SVR(
                    C=C,
                    epsilon=epsilon,
                    kernel=kernel,
                    gamma=gamma
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
            st.error(f"Error occurred during SVR parameter tuning: {str(e)}")
            return None

    def xgb_regressor(self, X_train, y_train, param_distributions=None, scoring='neg_mean_squared_error'):
        """
        Perform hyperparameter tuning for an XGBoost Regressor using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'neg_mean_squared_error')
            
        Returns:
        --------
        best_model : trained XGBRegressor with best parameters
        """
        try:
            if param_distributions is None:
                param_distributions = {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                    'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 15},
                    'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
                    'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
                    'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0},
                    'gamma': {'type': 'float', 'low': 0, 'high': 5},
                    'reg_alpha': {'type': 'loguniform', 'low': 0.001, 'high': 10.0},
                    'reg_lambda': {'type': 'loguniform', 'low': 0.001, 'high': 10.0}
                }
            
            def create_model(n_estimators, learning_rate, max_depth, min_child_weight, 
                             subsample, colsample_bytree, gamma, reg_alpha, reg_lambda):
                return XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    gamma=gamma,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    objective='reg:squarederror',
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
            st.error(f"Error occurred during XGBoost regressor parameter tuning: {str(e)}")
            return None
            
    def stacking_regressor(self, X_train, y_train, param_distributions=None, scoring='neg_mean_squared_error'):
        """
        Perform hyperparameter tuning for a Stacking Regressor using Optuna
        
        Parameters:
        -----------
        X_train : DataFrame
            Features training data
        y_train : Series
            Target training data
        param_distributions : dict, optional
            Dictionary of parameter names and their possible distributions for Optuna
        scoring : str, optional
            Metric to evaluate models (default: 'neg_mean_squared_error')
            
        Returns:
        --------
        best_model : trained StackingRegressor with best parameters
        """
        try:
            if param_distributions is None:
                param_distributions = {
                    'rf_n_estimators': {'type': 'int', 'low': 50, 'high': 200},
                    'rf_max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'gb_n_estimators': {'type': 'int', 'low': 50, 'high': 200},
                    'gb_learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                    'xgb_n_estimators': {'type': 'int', 'low': 50, 'high': 200},
                    'xgb_learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                    'final_estimator': {'type': 'categorical', 'values': ['linear', 'rf']}
                }
            
            def create_model(rf_n_estimators, rf_max_depth, gb_n_estimators, gb_learning_rate, 
                             xgb_n_estimators, xgb_learning_rate, final_estimator):
                # Create base estimators
                estimators = [
                    ('rf', RandomForestRegressor(
                        n_estimators=rf_n_estimators,
                        max_depth=rf_max_depth,
                        random_state=0
                    )),
                    ('gb', GradientBoostingRegressor(
                        n_estimators=gb_n_estimators,
                        learning_rate=gb_learning_rate,
                        random_state=0
                    )),
                    ('xgb', XGBRegressor(
                        n_estimators=xgb_n_estimators,
                        learning_rate=xgb_learning_rate,
                        random_state=0
                    ))
                ]
                
                # Create final estimator
                if final_estimator == 'linear':
                    final_est = LinearRegression()
                else:  # rf
                    final_est = RandomForestRegressor(random_state=0)
                
                return StackingRegressor(
                    estimators=estimators,
                    final_estimator=final_est,
                    cv=5,
                    n_jobs=-1
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
            st.error(f"Error occurred during Stacking regressor parameter tuning: {str(e)}")
            return None