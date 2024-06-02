import pandas as pd
import numpy as np
import logging
import streamlit as st


class DataImputer:
    def __init__(self, data):
        super().__init__()
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data
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
        
        st.markdown(
    "<h1 style='text-align: center; font-size: 30px;'>Impute Missing Values</h1>", 
    unsafe_allow_html=True
)
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h1>", unsafe_allow_html=True)
        st.dataframe(self.data, width=800)

        
        option = st.sidebar.selectbox("Select an Imputation Method", [
            "Drop Columns",
            "Check Missing Values",
            "Drop Missing Values",
            "Add Variable to Denote NA",
            "Impute NA with Arbitrary Value",
            "Impute NA with Interpolation",
            "Impute NA with KNN",
            "Impute NA with SimpleImputer",
            "Impute NA with Average",
            "Impute NA with End of Distribution",
            "Impute NA with Random Sampling"
        ])

        if option == "Check Missing Values":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Check Missing Values</h1>", unsafe_allow_html=True)
            if st.button("Check"):
                self.check_missing()
                st.write(self.check_missing())
                
        elif option == "Drop Columns":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Drop Columns</h1>", unsafe_allow_html=True)
            columns_to_drop = st.multiselect("Select Columns to Drop", self.data.columns)
            if st.button("Drop Columns"):
                try:
                    self.data = self._drop_columns(columns_to_drop)
                    st.dataframe(self.data)
                except Exception as e:
                    st.error(f"An error occurred while dropping columns: {str(e)}")

        elif option == "Drop Missing Values":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Drop Missing Values</h1>", unsafe_allow_html=True)
            axis = st.radio("Drop rows or columns?", ["Rows", "Columns"])
            axis = 0 if axis == "Rows" else 1
            if st.button("Drop"):
                self.drop_missing(axis=axis)
                if self.data is not None:
                    st.dataframe(self.data)
                else:
                    st.warning("No missing values found in the data.")

        elif option == "Add Variable to Denote NA":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Add Variable to Denote NA</h1>", unsafe_allow_html=True)
            selected_columns = st.multiselect("Select columns to impute", options=self.data.columns)
            if st.button("Add"):
                if selected_columns:
                    data_add_var = self.add_var_denote_NA(NA_col=selected_columns)
                    st.write(data_add_var)
                else:
                    st.warning("Please select at least one column to impute")

        elif option == "Impute NA with Arbitrary Value":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Impute NA with Arbitrary Value</h1>", unsafe_allow_html=True)
            impute_value = st.text_input("Enter Arbitrary Value")
            na_cols = st.multiselect("Select Columns", self.data.columns)
            if st.button("Impute Arbitrary Value"):
                self.impute_NA_with_arbitrary(impute_value=float(impute_value), NA_col=na_cols)
                st.success(f"{na_cols} columns imputed successfully")

        elif option == "Impute NA with Interpolation":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Impute NA with Interpolation</h1>", unsafe_allow_html=True)
            na_cols = st.multiselect("Select Columns", self.data.columns)
            interp_method = st.selectbox("Interpolation Method", ['linear', 'quadratic', 'cubic'])
            interp_limit = st.text_input("Limit", None)
            interp_limit_direction = st.selectbox("Limit Direction", ['forward', 'backward', 'both'])
            if st.button("Impute Interpolation"):
                data_interp = self.impute_NA_with_interpolation(method=interp_method, limit=interp_limit, limit_direction=interp_limit_direction, NA_col=na_cols)
                st.write(data_interp)

        elif option == "Impute NA with KNN":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Impute NA with KNN</h1>", unsafe_allow_html=True)
            n_neighbors = st.number_input("Number of Neighbors", min_value=1, value=5)
            selected_columns = st.multiselect("Select columns to impute", options=self.data.columns)
            if st.button("Impute KNN"):
                if selected_columns:  # Check if at least one column is selected
                    data_knn = self.impute_NA_with_knn(NA_col=selected_columns, n_neighbors=n_neighbors)
                    st.write(data_knn)
                else:
                    st.warning("Please select at least one column to impute")

        elif option == "Impute NA with SimpleImputer":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Impute NA with SimpleImputer</h1>", unsafe_allow_html=True)
            na_cols = st.multiselect("Select Columns", self.data.columns)
            impute_strategy = st.selectbox("Imputation Strategy", ["mean", "median", "most_frequent", "constant"])

            if st.button("Impute SimpleImputer"):
                try:
                    data_simple_imputer = self.impute_NA_with_simple_imputer(NA_col=na_cols, strategy=impute_strategy)
                    st.write(data_simple_imputer)
                except Exception as e:
                    st.error(f"An error occurred while imputing NA with SimpleImputer: {str(e)}")

        elif option == "Impute NA with Average":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Impute NA with Average</h1>", unsafe_allow_html=True)
            na_cols = st.multiselect("Select Columns", self.data.columns)
            strategy = st.selectbox("Imputation Strategy", ['mean', 'median', 'mode'])
            if st.button("Impute Average"):
                data_avg = self.impute_NA_with_avg(strategy=strategy, NA_col=na_cols)
                st.write(data_avg)

        elif option == "Impute NA with End of Distribution":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Impute NA with End of Distribution</h1>", unsafe_allow_html=True)
            na_cols = st.multiselect("Select Columns", self.data.columns)
            if st.button("Impute End of Distribution"):
                data_end_dist = self.impute_NA_with_end_of_distribution(NA_col=na_cols)
                st.write(data_end_dist)

        elif option == "Impute NA with Random Sampling":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Impute NA with Random Sampling</h1>", unsafe_allow_html=True)
            na_cols = st.multiselect("Select Columns", self.data.columns)
            random_state = st.number_input("Random State", min_value=0, value=0)
            if st.button("Impute Random"):
                data_random = self.impute_NA_with_random(NA_col=na_cols, random_state=random_state)
                st.write(data_random)
        
        return self.data