import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import os


class DataAnalyzer:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        self.data = data
        self.numeric_data = self.data.select_dtypes(include='number')

    
    def drop_columns(self, columns_to_drop):
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
            st.error("Dropped the following columns: %s" % ", ".join(columns_to_drop))
            return self.data

        except Exception as e:
            st.error("An error occurred while dropping columns: %s" % str(e))
            return self.data

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
    
    def _display_info(self):
        """
        Displays summary information about the dataset.
        """
        st.markdown("<h1 style='text-align: center; font-size: 25px;'>Dataset Information</h1>", unsafe_allow_html=True)
        st.write(f"Rows: {self.data.shape[0]}")
        st.write(f"Columns: {self.data.shape[1]}")
        st.write("Info:")
        
        # Get the summary information of the DataFrame
        summary_info = self.data.dtypes.reset_index()
        summary_info.columns = ['Column', 'Dtype']
        summary_info['Non-Null Count'] = self.data.count().values
        
        # Create DataFrame from the summary information
        summary = pd.DataFrame(summary_info)
        st.write(summary)

    def _describe(self):
        """Generates descriptive statistics for numerical columns."""
        #data = self.convert_to_float(self.data)

        # Generate descriptive statistics
        profile = self.data.describe(include='all')
        return profile

    def _discrete_var_barplot(self, x, y, output_path=None):
        """
        Creates a bar plot for discrete variables.
        """
        
        fig = go.Figure(data=[go.Bar(x=self.data[x], y=self.data[y], marker_color="#f95738")])
        
        if output_path:
            output = os.path.join(output_path, f'Barplot_{x}.html')
            fig.write_html(output)
            st.write('Figure saved at:', output)
    
        st.plotly_chart(fig, use_container_width=True)

    def _discrete_var_countplot(self, x, output_path=None):
        """
        Creates a count plot for a discrete variable.
        """
        if x not in self.data.columns:
            st.error(f"'{x}' column does not exist in the dataset.")
            return

        counts = self.data[x].value_counts()

        if counts.empty:
            st.error(f"No data available for '{x}'.")
            return

        bar_trace = go.Bar(x=counts.index, y=counts.values, marker_color="#f95738")
        layout = go.Layout(title=f'Count Plot: {x}')
        
        # Create figure
        fig = go.Figure(data=[bar_trace], layout=layout)
        
        if output_path:
            output = os.path.join(output_path, f'Countplot_{x}.html')
            fig.write_html(output)
            st.write('Figure saved at:', output)
        
        st.plotly_chart(fig)

    def _discrete_var_boxplot(self, x, y, output_path=None):
        """
        Creates a box plot for a discrete variable against a continuous variable.
        """
        # Create box trace
        box_trace = go.Box(x=self.data[x], y=self.data[y], marker_color="#f95738")
        
        # Create layout
        layout = go.Layout(title=f'Box Plot: {y} vs {x}')
        
        # Create figure
        fig = go.Figure(data=[box_trace], layout=layout)
        
        if output_path:
            output = os.path.join(output_path, f'Boxplot_{x}_{y}.html')
            fig.write_html(output)
            st.write('Figure saved at:', output)
        
        st.plotly_chart(fig)

    def _continuous_var_distplot(self, x, output_path=None, bins=None):
        """
        Creates a distribution plot for a continuous variable.
        """
        fig = px.histogram(self.data, x=x, nbins=bins, title=f'Distribution Plot: {x}', histnorm='probability density', marginal='box')

        x_data = self.data[x].values
        kde_fig = ff.create_distplot([x_data], ['KDE'], curve_type='kde')

        # Add KDE traces to the histogram
        for trace in kde_fig['data']:
            fig.add_trace(trace)

        if output_path:
            output = os.path.join(output_path, f'Distplot_{x}.html')
            fig.write_html(output)
            st.write('Figure saved at:', output)
        
        st.plotly_chart(fig)

    def _scatter_plot(self, x, y, output_path=None):
        """
        Creates a scatter plot for two continuous variables.
        """
        # Create scatter trace
        scatter_trace = go.Scatter(x=self.data[x], y=self.data[y], mode='markers', marker_color="#f95738")
        
        # Create layout
        layout = go.Layout(title=f'Scatter Plot: {y} vs {x}')
        
        # Create figure
        fig = go.Figure(data=[scatter_trace], layout=layout)
        
        if output_path:
            output = os.path.join(output_path, f'Scatter_plot_{x}_{y}.html')
            fig.write_html(output)
            st.write('Figure saved at:', output)
        
        st.plotly_chart(fig)

    def _correlation_plot(self, output_path=None):
        """
        Creates a correlation plot for numerical columns.
        """
        corr_data = self.numeric_data.corr()

        # Create heatmap trace
        heatmap_trace = go.Heatmap(
                                x=corr_data.columns,
                                y=corr_data.index,
                                z=corr_data.values,
                                colorscale='Viridis'
                                )

        # Create layout
        layout = go.Layout(title='Correlation Plot')

        # Create figure
        fig = go.Figure(data=[heatmap_trace], layout=layout)

        # Add annotations with correlation coefficients
        for i in range(len(corr_data)):
            for j in range(len(corr_data)):
                fig.add_annotation(x=corr_data.columns[i], y=corr_data.index[j],
                                text=str(round(corr_data.iloc[j, i], 2)),
                                showarrow=False)

        if output_path:
            output = os.path.join(output_path, 'Corr_plot.html')
            fig.write_html(output)
            st.write('Figure saved at:', output)

        st.plotly_chart(fig)


    def _interactive_data_table(self):
        """Displays an interactive data table with Excel-like functionality."""
        if not hasattr(self, 'data') or not isinstance(self.data, pd.DataFrame):
            st.error("Invalid data. Please ensure that 'self.data' is a valid DataFrame.")
            return

        st.write("#### Interactive Data Table")

        # Filter columns
        if not self.data.columns.tolist():
            st.error("No columns found in the data.")
            return

        selected_columns = st.multiselect("Select Columns", self.data.columns, default=self.data.columns.tolist())

        # Global search functionality
        search_term = st.text_input("Search", placeholder="Search in table...")
        if search_term:
            filtered_data = self.data[self.data.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
        else:
            filtered_data = self.data[selected_columns]

        # Column-specific filtering
        column_filters = {}
        for column in selected_columns:
            filter_operator = st.selectbox(f"{column} filter", ["None"] + [ "==", "!=", ">", "<", ">=", "<="], key=f"{column}_operator")
            if filter_operator:
                filter_value = st.text_input(f"{column} filter value", key=f"{column}_value")
                if filter_value:
                    try:
                        filtered_data = filtered_data[eval(f"filtered_data['{column}'] {filter_operator} {filter_value}")]
                        column_filters[column] = (filter_operator, filter_value)
                    except Exception as e:
                        st.warning(f"Invalid filter expression for column '{column}': {e}")

        # Sorting
        sorted_column = st.selectbox("Sort by", ["None"] + selected_columns, index=0)
        if sorted_column != "None":
            ascending = st.checkbox("Ascending", value=True)
            filtered_data = filtered_data.sort_values(by=sorted_column, ascending=ascending)

        # Display editable table
        edited_data = filtered_data.copy()
        st.dataframe(edited_data, use_container_width=True)

        # Optionally highlight rows based on condition
        highlight_column = st.selectbox("Highlight rows where:", ["None"] + selected_columns, index=0)
        if highlight_column != "None":
            condition_value = st.text_input("Condition", "")
            if condition_value:
                try:
                    highlighted = filtered_data[eval(f"filtered_data['{highlight_column}'] {condition_value}")].index
                    st.dataframe(filtered_data.style.apply(lambda x: ['background: lightgreen' if x.name in highlighted else '' for i in x], axis=1), use_container_width=True)
                except Exception as e:
                    st.warning(f"Invalid condition for highlighting: {e}")

        # Visualization based on selected column
        feature_to_visualize = st.selectbox("Select Feature to Visualize", selected_columns)
        if feature_to_visualize:
            visualization_type = st.selectbox("Select Visualization Type", ["Count Plot", "Box Plot", "Distribution Plot", "Scatter Plot", "Correlation Plot"])
            if visualization_type == "Count Plot":
                self._discrete_var_countplot(feature_to_visualize, output_path=None)
            elif visualization_type == "Box Plot":
                y = st.selectbox("Select Y-axis", self.data.columns)
                self._discrete_var_boxplot(feature_to_visualize, y, output_path=None)
            elif visualization_type == "Distribution Plot":
                self._continuous_var_distplot(feature_to_visualize, output_path=None)
            elif visualization_type == "Scatter Plot":
                y = st.selectbox("Select Y-axis", self.data.columns)
                self._scatter_plot(feature_to_visualize, y, output_path=None)
            elif visualization_type == "Correlation Plot":
                self._correlation_plot(output_path=None)

        

    def _time_series_plot(self, time_column, value_column, aggregation_function='mean', time_interval='D', smoothing_technique=None, output_path=None):
        """
        Creates a time series plot based on the specified time and value columns.
        """
        try:
            # Ensure time column is datetime type
            self.data[time_column] = pd.to_datetime(self.data[time_column])
        
        except Exception as e:
            st.warning(f"An error occurred: {e}")
            return
        
        try:
            # Aggregate data based on time intervals
            aggregated_data = self.data.resample(time_interval, on=time_column).agg({value_column: aggregation_function}).reset_index()
        except ValueError as e:
            st.warning(f"An error occurred: {e}")
            return
        
        # Reset the index
        aggregated_data.reset_index(drop=True, inplace=True)
        
        # Plot time series
        time_series_fig = px.line(aggregated_data, x=time_column, y=value_column, title="Time Series Plot", labels={time_column: "Time", value_column: value_column})
        

        if smoothing_technique:
            # Apply smoothing technique
            aggregated_data[value_column] = aggregated_data[value_column].rolling(window=7).mean()
        
        # Display the plot
        st.plotly_chart(time_series_fig)
        
        # Save plot as HTML if output path is provided
        if output_path:
            ts_plot_path = os.path.join(output_path, 'time_series_plot.html')
            time_series_fig.write_html(ts_plot_path)
            st.write('Time series plot saved at:', ts_plot_path)

    def _distribution_comparison_plot(self, columns, output_path=None):
        """
        Creates a plot to compare the distributions of multiple columns.
        """
        if not columns:
            st.warning("Please select at least one column for comparison.")
            return
        
        fig = make_subplots(rows=1, cols=len(columns), subplot_titles=[f"Distribution of {column}" for column in columns])
        
        # Plot distribution for each column
        for i, column in enumerate(columns):
            fig.add_trace(go.Histogram(x=self.data[column], name=column, histnorm='probability density'), row=1, col=i+1)
        fig.update_layout(title="Distribution Comparison Plot", showlegend=False)
        
        st.plotly_chart(fig)
        if output_path:
            dist_comp_plot_path = os.path.join(output_path, 'distribution_comparison_plot.html')
            fig.write_html(dist_comp_plot_path)
            st.write('Distribution comparison plot saved at:', dist_comp_plot_path)

    def analyzer(self):
        """
        Main function to display the data analysis options and perform selected analysis.
        """
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Data Explore</h1>", unsafe_allow_html=True)       
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>", unsafe_allow_html=True)
        st.dataframe(self.data, width=800)
        
        analysis_option = st.sidebar.selectbox("Select Exploration Option", ["Dataset Information", "Describe", "Drop Columns", "Discrete Variable Barplot", "Discrete Variable Countplot",
                                                            "Discrete Variable Boxplot", "Continuous Variable Distplot", 
                                                            "Scatter Plot", "Correlation Plot", "Time Series Plot",
                                                            "Distribution Comparison Plot", "Interactive Data Table"])
        
        if analysis_option == "Dataset Information":
            self._display_info()
        elif analysis_option == "Describe":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Descriptive Statistics</h1>", unsafe_allow_html=True)
            st.dataframe(self._describe(), width=800)
        elif analysis_option == "Drop Columns":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Drop Columns</h1>", unsafe_allow_html=True)
            columns_to_drop = st.multiselect("Select Columns to Drop", self.data.columns)
            if st.button("Drop Columns"):
                try:
                    self.data = self.drop_columns(columns_to_drop)
                    st.dataframe(self.data)
                except Exception as e:
                    st.error(f"An error occurred while dropping columns: {str(e)}")
        elif analysis_option == "Discrete Variable Barplot":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Discrete Variable Barplot</h1>", unsafe_allow_html=True)
            x = st.selectbox("Select X axis", self.data.columns)
            y = st.selectbox("Select Y axis", self.data.columns)
            self._discrete_var_barplot(x, y)
        elif analysis_option == "Discrete Variable Countplot":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Discrete Variable Countplot</h1>", unsafe_allow_html=True)
            x = st.selectbox("Select X axis", self.data.columns)
            self._discrete_var_countplot(x)
        elif analysis_option == "Discrete Variable Boxplot":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Discrete Variable Boxplot</h1>", unsafe_allow_html=True)
            x = st.selectbox("Select X axis", self.data.columns)
            y = st.selectbox("Select Y axis", self.data.columns)
            self._discrete_var_boxplot(x, y)
        elif analysis_option == "Continuous Variable Distplot":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Continuous Variable Distplot</h1>", unsafe_allow_html=True)
            x = st.selectbox("Select X axis", self.data.columns)
            bins = st.number_input("Enter number of bins", min_value=1, max_value=100, value=10)
            self._continuous_var_distplot(x, bins=bins)
        elif analysis_option == "Scatter Plot":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Scatter Plot</h1>", unsafe_allow_html=True)
            x = st.selectbox("Select X axis", self.data.columns)
            y = st.selectbox("Select Y axis", self.data.columns)
            self._scatter_plot(x, y)
        elif analysis_option == "Correlation Plot":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Correlation Plot</h1>", unsafe_allow_html=True)
            self._correlation_plot()
        elif analysis_option == "Time Series Plot":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Time Series Plot</h1>", unsafe_allow_html=True)
            time_column = st.selectbox("Select Time Column", self.data.columns)
            value_column = st.selectbox("Select Value Column", self.data.columns)
            aggregation_function = st.selectbox("Select Aggregation Function", ["mean", "sum", "count", "min", "max"])
            time_interval = st.selectbox("Select Time Interval", ["D", "W", "M", "Q", "Y"])
            self._time_series_plot(time_column, value_column, aggregation_function, time_interval)
        elif analysis_option == "Distribution Comparison Plot":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Distribution Comparison Plot</h1>", unsafe_allow_html=True)
            columns = st.multiselect("Select Columns for Comparison", self.data.columns)
            self._distribution_comparison_plot(columns)
        elif analysis_option == "Interactive Data Table":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Interactive Data Table</h1>", unsafe_allow_html=True)
            self._interactive_data_table()
        
        return self.data
