import pandas as pd
import numpy as np
import logging
import streamlit as st
from show_code import ShowCode


class DataImputer:
    def __init__(self, data):
        super().__init__()
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data
        self.view_code = ShowCode()
        self.view_code.set_target_class(DataImputer)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

    def _drop_columns(self, columns_to_drop):
        """
        Drops the specified columns from the DataFrame.
        """
        try:
            # Check if the columns exist in the DataFrame
            non_existent_cols = [col for col in columns_to_drop if col not in self.data.columns]
            if non_existent_cols:
                st.error("The following columns do not exist in the DataFrame and will be ignored: %s" % ", ".join(non_existent_cols))
                columns_to_drop = [col for col in columns_to_drop if col in self.data.columns]

            # Drop the specified columns
            self.data = self.data.drop(columns_to_drop, axis=1)
            st.info("Dropped the following columns: %s" % ", ".join(columns_to_drop))
            return self.data

        except Exception as e:
            st.error("An error occurred while dropping columns: %s" % str(e))
    
    def check_missing(self, output_path=None):
        try:
            result = pd.concat([self.data.isnull().sum(), self.data.isnull().mean()], axis=1)
            result = result.rename(index=str, columns={0: 'total missing', 1: 'proportion'})
            
            if output_path is not None:
                result.to_csv(output_path + 'missing.csv')
                self.logger.info('Result saved at %smissing.csv', output_path)
            return result
        except Exception as e:
            self.logger.error("An error occurred while checking missing values: %s", str(e))
            raise

    def drop_missing(self, axis=0):
        try:
            original_shape = self.data.shape
            self.data = self.data.dropna(axis=axis)
            if self.data.shape == original_shape:
                return None  
            else:
                return self.data
        except Exception as e:
            self.logger.error("An error occurred while dropping missing values: %s", str(e))
            raise
            
    def add_var_denote_NA(self, NA_col=[]):
        try:
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    self.data[i] = np.where(self.data[i].isnull(), 1, 0)
                    return self.data
                else:
                    self.logger.warning("Column %s has no missing cases", i)
        except Exception as e:
            self.logger.error("An error occurred while adding variable to denote NA: %s", str(e))
            raise

    def impute_NA_with_arbitrary(self, impute_value, NA_col=[]):
        try:
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    self.data[i].fillna(impute_value, inplace=True)
                else:
                    self.logger.warning("Column %s has no missing cases", i)
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with arbitrary value: %s", str(e))
            raise

    def impute_NA_with_avg(self, strategy='mean', NA_col=[]):
        try:
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    if strategy == 'mean':
                        self.data[i].fillna(self.data[i].mean(), inplace=True)
                    elif strategy == 'median':
                        self.data[i].fillna(self.data[i].median(), inplace=True)
                    elif strategy == 'mode':
                        self.data[i].fillna(self.data[i].mode()[0], inplace=True)
                else:
                    self.logger.warning("Column %s has no missing", i)
            return self.data
        except Exception as e:
            error_msg = "An error occurred while imputing NA with average: %s" % str(e)
            st.error(error_msg)

    def impute_NA_with_end_of_distribution(self, NA_col=[]):
        try:
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    self.data[i].fillna(self.data[i].mean() + 3 * self.data[i].std(), inplace=True)
                    return self.data
                else:
                    self.logger.warning("Column %s has no missing", i)
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with end of distribution: %s", str(e))
            raise

    def impute_NA_with_random(self, NA_col=[], random_state=0):
        try:
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    random_sample = self.data[i].dropna().sample(self.data[i].isnull().sum(), random_state=random_state)
                    random_sample.index = self.data[self.data[i].isnull()].index
                    self.data.loc[self.data[i].isnull(), i] = random_sample
                    return self.data
                
                else:
                    self.logger.warning("Column %s has no missing", i)
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with random sampling: %s", str(e))
            raise

    def impute_NA_with_interpolation(self, method='linear', limit=None, limit_direction='forward', NA_col=[]):
        try:
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    self.data[i] = self.data[i].interpolate(method=method, limit=limit, limit_direction=limit_direction)
                    return self.data
                else:
                    self.logger.warning("Column %s has no missing cases", i)
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with interpolation: %s", str(e))
            raise

    def impute_NA_with_knn(self, NA_col=[], n_neighbors=5):
        try:
            from sklearn.impute import KNNImputer
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    imputed_values = knn_imputer.fit_transform(self.data[i].values.reshape(-1, 1))
                    self.data[i] = imputed_values.ravel()
                    return self.data
                else:
                    self.logger.warning("Column %s has no missing cases", i)
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with KNN: %s", str(e))
            raise

    def impute_NA_with_simple_imputer(self, NA_col=[], strategy='mean'):
        try:
            from sklearn.impute import SimpleImputer
            
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    imputer = SimpleImputer(strategy=strategy)
                    imputed_data = imputer.fit_transform(self.data[i].values.reshape(-1, 1))
                    self.data[i] = imputed_data.ravel()
                else:
                    self.logger.warning("Column %s has no missing cases", i)
            return self.data
        
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with SimpleImputer: %s", str(e))
            raise

    def imputer(self):
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Impute Missing Values</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h1>", unsafe_allow_html=True)
        st.dataframe(self.data, width=800)

        options = [
            "Check Missing Values",
            "Drop Columns",
            "Drop Missing Values",
            "Add Variable to Denote NA",
            "Impute NA with Arbitrary Value",
            "Impute NA with Interpolation",
            "Impute NA with KNN",
            "Impute NA with SimpleImputer",
            "Impute NA with Average",
            "Impute NA with End of Distribution",
            "Impute NA with Random Sampling"
        ]

        option = st.sidebar.selectbox("Select an Imputation Method", options)

        # Initialize session state for each option
        for opt in options:
            if f'{opt}_clicked' not in st.session_state:
                st.session_state[f'{opt}_clicked'] = False
            if f'{opt}_show_code' not in st.session_state:
                st.session_state[f'{opt}_show_code'] = False

        if option == "Check Missing Values":
            self._handle_check_missing_values()
        elif option == "Drop Columns":
            self._handle_drop_columns()
        elif option == "Drop Missing Values":
            self._handle_drop_missing_values()
        elif option == "Add Variable to Denote NA":
            self._handle_add_var_denote_na()
        elif option == "Impute NA with Arbitrary Value":
            self._handle_impute_arbitrary()
        elif option == "Impute NA with Interpolation":
            self._handle_impute_interpolation()
        elif option == "Impute NA with KNN":
            self._handle_impute_knn()
        elif option == "Impute NA with SimpleImputer":
            self._handle_impute_simple_imputer()
        elif option == "Impute NA with Average":
            self._handle_impute_average()
        elif option == "Impute NA with End of Distribution":
            self._handle_impute_end_distribution()
        elif option == "Impute NA with Random Sampling":
            self._handle_impute_random()

        return self.data

    def _handle_option(self, option, function, **kwargs):
        st.markdown(f"<h1 style='text-align: center; font-size: 25px;'>{option}</h1>", unsafe_allow_html=True)
        
        if st.button("Execute"):
            st.session_state[f'{option}_clicked'] = True

        if st.session_state[f'{option}_clicked']:
            result = function(**kwargs)
            st.write(result)
            
            st.session_state[f'{option}_show_code'] = st.checkbox('Show Code', value=st.session_state[f'{option}_show_code'])
            
            if st.session_state[f'{option}_show_code']:
                self.view_code._display_code(function.__name__)

    def _handle_check_missing_values(self):
        self._handle_option("Check Missing Values", self.check_missing)

    def _handle_drop_columns(self):
        columns_to_drop = st.multiselect("Select Columns to Drop", self.data.columns)
        self._handle_option("Drop Columns", self._drop_columns, columns_to_drop=columns_to_drop)

    def _handle_drop_missing_values(self):
        axis = st.radio("Drop rows or columns?", ["Rows", "Columns"])
        axis = 0 if axis == "Rows" else 1
        self._handle_option("Drop Missing Values", self.drop_missing, axis=axis)

    def _handle_add_var_denote_na(self):
        selected_columns = st.multiselect("Select columns to impute", options=self.data.columns)
        self._handle_option("Add Variable to Denote NA", self.add_var_denote_NA, NA_col=selected_columns)

    def _handle_impute_arbitrary(self):
        impute_value = st.text_input("Enter Arbitrary Value")
        na_cols = st.multiselect("Select Columns", self.data.columns)
        self._handle_option("Impute NA with Arbitrary Value", self.impute_NA_with_arbitrary, impute_value=float(impute_value), NA_col=na_cols)

    def _handle_impute_interpolation(self):
        na_cols = st.multiselect("Select Columns", self.data.columns)
        interp_method = st.selectbox("Interpolation Method", ['linear', 'quadratic', 'cubic'])
        interp_limit = st.text_input("Limit", None)
        interp_limit_direction = st.selectbox("Limit Direction", ['forward', 'backward', 'both'])
        self._handle_option("Impute NA with Interpolation", self.impute_NA_with_interpolation, method=interp_method, limit=interp_limit, limit_direction=interp_limit_direction, NA_col=na_cols)

    def _handle_impute_knn(self):
        n_neighbors = st.number_input("Number of Neighbors", min_value=1, value=5)
        selected_columns = st.multiselect("Select columns to impute", options=self.data.columns)
        self._handle_option("Impute NA with KNN", self.impute_NA_with_knn, NA_col=selected_columns, n_neighbors=n_neighbors)

    def _handle_impute_simple_imputer(self):
        na_cols = st.multiselect("Select Columns", self.data.columns)
        impute_strategy = st.selectbox("Imputation Strategy", ["mean", "median", "most_frequent", "constant"])
        self._handle_option("Impute NA with SimpleImputer", self.impute_NA_with_simple_imputer, NA_col=na_cols, strategy=impute_strategy)

    def _handle_impute_average(self):
        na_cols = st.multiselect("Select Columns", self.data.columns)
        strategy = st.selectbox("Imputation Strategy", ['mean', 'median', 'mode'])
        self._handle_option("Impute NA with Average", self.impute_NA_with_avg, strategy=strategy, NA_col=na_cols)

    def _handle_impute_end_distribution(self):
        na_cols = st.multiselect("Select Columns", self.data.columns)
        self._handle_option("Impute NA with End of Distribution", self.impute_NA_with_end_of_distribution, NA_col=na_cols)

    def _handle_impute_random(self):
        na_cols = st.multiselect("Select Columns", self.data.columns)
        random_state = st.number_input("Random State", min_value=0, value=0)
        self._handle_option("Impute NA with Random Sampling", self.impute_NA_with_random, NA_col=na_cols, random_state=random_state)