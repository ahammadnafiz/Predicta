import streamlit as st
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from show_code import ShowCode

class BestParam:
    def __init__(self, data):
        self.data = data
        self.view_code = ShowCode()
        self.view_code.set_target_class(BestParam)
    
    def grid_search(self, X_train, y_train, model, param_grid, scoring='accuracy', cv=5):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        st.write("Best Parameters:", grid_search.best_params_)
        st.write("Best Score:", grid_search.best_score_)
        return best_model

    def linear_regression(self, X_train, y_train):
        try:
            model = LinearRegression()
            param_grid = {'fit_intercept': [True, False]}
            best_model = self.grid_search(X_train, y_train, model, param_grid, scoring='neg_mean_squared_error')
            return best_model
        except ValueError:
            st.error("Error occurred during linear regression parameter tuning.")
            return None    

    def random_forest_regression(self, X_train, y_train, param_grid=None, scoring='neg_mean_squared_error'):
        try:
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [10, 15, 20]
                }
            model = RandomForestRegressor(random_state=0)
            best_model = self.grid_search(X_train, y_train, model, param_grid, scoring=scoring)
            return best_model
        except ValueError:
            st.error("Error occurred during random forest regression parameter tuning.")
            return None

    def random_forest_classifier(self, X_train, y_train, param_grid=None, scoring='accuracy'):
        try:
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [10, 15, 20],
                    'class_weight': ['balanced', None]
                }
            model = RandomForestClassifier(random_state=0)
            best_model = self.grid_search(X_train, y_train, model, param_grid, scoring=scoring)
            return best_model
        except ValueError:
            st.error("Error occurred during random forest classifier parameter tuning.")
            return None

    def gradient_boosting_classifier(self, X_train, y_train, param_grid=None, scoring='accuracy'):
        try:
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [10, 15, 20]
                }
            model = GradientBoostingClassifier(random_state=0)
            best_model = self.grid_search(X_train, y_train, model, param_grid, scoring=scoring)
            return best_model
        except ValueError:
            st.error("Error occurred during gradient boosting classifier parameter tuning.")
            return None

    def select_hyper(self):
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Select Best Parameters</h1>", unsafe_allow_html=True)
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
        
        if 'tuned_model' not in st.session_state:
            st.session_state.tuned_model = None
        
        if 'selected_option' not in st.session_state:
            st.session_state.selected_option = None

        if 'show_code' not in st.session_state:
            st.session_state.show_code = False

        tune_button = st.button("Best Param")

        if tune_button or (st.session_state.tuned_model is not None and option == st.session_state.selected_option):
            if option == "Linear Regression":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Linear Regression HyperParameter Tuning</h2>", unsafe_allow_html=True)
                st.session_state.tuned_model = self.linear_regression(feature_column, self.data[target_column])
            elif option == "Random Forest Regression":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Random Forest Regression HyperParameter Tuning</h2>", unsafe_allow_html=True)
                st.session_state.tuned_model = self.random_forest_regression(feature_column, self.data[target_column])
            elif option == "Random Forest Classifier":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Random Forest Classifier HyperParameter Tuning</h2>", unsafe_allow_html=True)
                st.session_state.tuned_model = self.random_forest_classifier(feature_column, self.data[target_column])
            elif option == "Gradient Boosting Classifier":
                st.markdown("<h2 style='text-align: center; font-size: 25px;'>Gradient Boosting Classifier HyperParameter Tuning</h2>", unsafe_allow_html=True)
                st.session_state.tuned_model = self.gradient_boosting_classifier(feature_column, self.data[target_column])
            
            st.session_state.selected_option = option

        if st.session_state.tuned_model is not None:
            st.session_state.show_code = st.checkbox('Show Code', value=st.session_state.show_code)
            
            if st.session_state.show_code:
                if st.session_state.selected_option == "Linear Regression":
                    self.view_code._display_code('linear_regression')
                elif st.session_state.selected_option == "Random Forest Regression":
                    self.view_code._display_code('random_forest_regression')
                elif st.session_state.selected_option == "Random Forest Classifier":
                    self.view_code._display_code('random_forest_classifier')
                elif st.session_state.selected_option == "Gradient Boosting Classifier":
                    self.view_code._display_code('gradient_boosting_classifier')
