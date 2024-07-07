import os
from dateutil import parser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st
from show_code import ShowCode


class DataAnalyzer:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        self.data = data
        self.view_code = ShowCode()
        self.view_code.set_target_class(DataAnalyzer)
        self.numeric_data = self.data.select_dtypes(include='number')
            
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
                st.error(f"Error converting column '{col}': {e}")
        
        return data

    def _line_plot(self, x, y_list):
        """Creates a line plot for multiple y-axis variables against a single x-axis variable. """
        
        # Use Plotly Express to create the line plot
        fig = px.line(self.data, x=x, y=y_list, markers=True)

        # Update layout and display the plot using Streamlit
        fig.update_layout(title=f'Line Plot: {", ".join(y_list)} vs {x}', xaxis_title=x, yaxis_title="Value")
        st.plotly_chart(fig)
        self.view_code._display_code('_line_plot', x, y_list)
    
    def _boxplot_with_outliers(self, x, y, output_path=None):
        """
        Creates a box plot with optional outlier analysis.
        """
        fig = go.Figure()
        fig.add_trace(go.Box(x=self.data[x], y=self.data[y], name=y, boxpoints='outliers'))
        
        fig.update_layout(title=f'Box Plot: {y} vs {x}')
        
        if output_path:
            output = os.path.join(output_path, f'Boxplot_with_outliers_{x}_{y}.html')
            fig.write_html(output)
            st.write('Figure saved at:', output)
        
        st.plotly_chart(fig)
        self.view_code._display_code('_boxplot_with_outliers', x, y)
   
    def _pie_chart(self, category_columns, target_column):
        """
        Creates dynamic pie charts to visualize the distribution of multiple categorical variables.

        Args:
            category_columns (list of str): List of column names for the categorical variables.

        """
       # Iterate over each category column and create a pie chart for each category
        for category_column in category_columns:
            # Get unique categories in the current category column
            categories = self.data[category_column].unique()

            # Define a color palette for the pie chart slices
            color_palette = px.colors.qualitative.Pastel

            # Iterate over each category and create a pie chart
            for category in categories:
                category_data = self.data[self.data[category_column] == category]
                target_counts = category_data[target_column].value_counts()

                # Use Plotly Express to create the pie chart
                fig = px.pie(names=target_counts.index, values=target_counts.values,
                             title=f'Distribution of {target_column} in {category_column} {category}',
                             color_discrete_sequence=color_palette)

                # Customize pie chart layout
                fig.update_traces(textposition='inside', textinfo='percent+label', pull=0.05,
                                  marker=dict(line=dict(color='white', width=1)))

                # Display the pie chart using Streamlit's st.plotly_chart
                st.plotly_chart(fig)
        self.view_code._display_code('_pie_chart', category_columns, target_column)
                
    def _pairwise_scatter_matrix(self, variables, output_path=None):
        """
        Creates a pairwise scatter plot matrix for multiple variables.
        """
        scatter_matrix_fig = px.scatter_matrix(self.data[variables], title='Pairwise Scatter Plot Matrix')
        
        if output_path:
            output = os.path.join(output_path, 'pairwise_scatter_matrix.html')
            scatter_matrix_fig.write_html(output)
            st.write('Pairwise scatter plot matrix saved at:', output)
        
        st.plotly_chart(scatter_matrix_fig)
        self.view_code._display_code('_pairwise_scatter_matrix', variables)
    
    def _categorical_heatmap(self, x, y, output_path=None):
        """
        Creates a heatmap for visualizing relationships between two categorical variables.
        """
        pivot_table = self.data.pivot_table(index=x, columns=y, aggfunc='size', fill_value=0)
        
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns.tolist(),
            y=pivot_table.index.tolist(),
            colorscale='Viridis'
        ))
        
        heatmap_fig.update_layout(title=f'Heatmap: {x} vs {y}')
        
        if output_path:
            output = os.path.join(output_path, f'heatmap_{x}_vs_{y}.html')
            heatmap_fig.write_html(output)
            st.write('Heatmap saved at:', output)
        
        st.plotly_chart(heatmap_fig)
        self.view_code._display_code('_categorical_heatmap', x, y)

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
        self.view_code._display_code('_discrete_var_barplot', x, y)

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
        self.view_code._display_code('_discrete_var_countplot', x)

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
        self.view_code._display_code('_discrete_var_boxplot', x, y)

    def _continuous_var_distplot(self, x, output_path=None, bins=None):
        """
        Creates a distribution plot for a continuous variable.
        """
        try:
            fig = px.histogram(self.data, x=x, nbins=bins, title=f'Distribution Plot: {x}', histnorm='probability density', marginal='box')

            x_data = self.data[x].values

            # Convert x_data to numeric type if needed (e.g., from string)
            try:
                x_data = pd.to_numeric(x_data)
            except ValueError:
                # Handle non-convertible values or NaNs if necessary
                st.warning(f"Failed to convert '{x}' column to numeric type. Plotting distribution without KDE.")
            
            kde_fig = ff.create_distplot([x_data], ['KDE'], curve_type='kde')

            # Add KDE traces to the histogram
            for trace in kde_fig['data']:
                fig.add_trace(trace)

            if output_path:
                output = os.path.join(output_path, f'Distplot_{x}.html')
                fig.write_html(output)
                st.write('Figure saved at:', output)
            
            # Show the Plotly figure using Streamlit
            st.plotly_chart(fig)
            self.view_code._display_code('_continuous_var_distplot', x, bins)

        except Exception as e:
            st.error(f"An error occurred while generating the distribution plot: {e}")

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
        self.view_code._display_code('_scatter_plot', x, y)

    def _scatter_3d_plot(self, x, y, z, output_path=None):
        """
        Creates a 3D scatter plot for three variables.
        Args:
            x (str): Name of the x-axis variable.
            y (str): Name of the y-axis variable.
            z (str): Name of the z-axis variable.
            output_path (str): Optional path to save the plot as an HTML file.
        """
        # Check if z-axis data is categorical (string values)
        if pd.api.types.is_string_dtype(self.data[z]):
            st.info("The selected z-axis variable contains categorical data (string values). "
                     "Please choose a numeric variable for the z-axis to create a 3D scatter plot.")
            return

        # Convert z data to numeric if needed
        self.data[z] = pd.to_numeric(self.data[z], errors='coerce')

        # Create 3D scatter trace
        scatter_3d_trace = go.Scatter3d(
            x=self.data[x],
            y=self.data[y],
            z=self.data[z],
            mode='markers',
            marker=dict(
                size=8,
                color=self.data[z],  # Use z variable for color
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title=z)  # Optional colorbar title
            )
        )

        # Create layout for the 3D scatter plot
        layout = go.Layout(
            title=f'3D Scatter Plot: {x} vs {y} vs {z}',
            scene=dict(
                xaxis=dict(title=x),
                yaxis=dict(title=y),
                zaxis=dict(title=z)
            )
        )

        # Create figure
        fig = go.Figure(data=[scatter_3d_trace], layout=layout)

        # Display the 3D scatter plot using Streamlit
        st.plotly_chart(fig)
        self.view_code._display_code('_scatter_3d_plot', x, y, z)

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
        self.view_code._display_code('_correlation_plot')

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
            self.data[time_column] = self.data[time_column].apply(lambda x: parser.parse(str(x)))
        
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
        self.view_code._display_code('_time_series_plot', time_column, value_column, aggregation_function, time_interval, smoothing_technique)
        
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
        self.view_code._display_code('_distribution_comparison_plot', columns)
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
        
        analysis_option = st.sidebar.selectbox("Select Exploration Option", [
                                                                             "Line Plot",
                                                                             "Pie Charts",
                                                                             "Discrete Variable Barplot", 
                                                                             "Discrete Variable Countplot",
                                                                             "Discrete Variable Boxplot", 
                                                                             "Continuous Variable Distplot",
                                                                             "Box Plot with Outliers",
                                                                             "Pairwise Scatter Plot Matrix",
                                                                             "Scatter Plot",
                                                                             "3D Scatter Plot",
                                                                             "Categorical Heatmap",
                                                                             "Correlation Plot", 
                                                                             "Time Series Plot",
                                                                             "Distribution Comparison Plot", 
                                                                             "Interactive Data Table"])
        

        if analysis_option == "Line Plot":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Line Plot</h1>", unsafe_allow_html=True)
            x = st.selectbox("Select X axis", self.data.columns)
            y_list = st.multiselect("Select Y axis", self.data.columns)
            
            # Ensure at least one y variable is selected
            if y_list:
                self._line_plot(x, y_list)
            else:
                st.warning("Please select at least one Y axis variable.")
        elif analysis_option == "Pie Charts":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Pie Chart</h1>", unsafe_allow_html=True)
               # Get user input for the categorical columns and the target column
            category_columns = st.multiselect("Select Categorical Columns", self.data.columns)
            target_column = st.selectbox("Select Target Column", self.data.columns)

            if category_columns and target_column:
                # Display distribution by category for the selected columns
                self._pie_chart(category_columns, target_column)
            else:
                st.warning("Please select at least one categorical column and one target column.")

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
        elif analysis_option == "3D Scatter Plot":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>3D Scatter Plot</h1>", unsafe_allow_html=True)
            x = st.selectbox("Select X axis", self.data.columns)
            y = st.selectbox("Select Y axis", self.data.columns)
            z = st.selectbox("Select Z axis", self.data.columns)
            self._scatter_3d_plot(x, y, z)

        elif analysis_option == "Box Plot with Outliers":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Box Plot with Outliers</h1>", unsafe_allow_html=True)
            x = st.selectbox("Select X axis", self.data.columns)
            y = st.selectbox("Select Y axis", self.data.columns)
            self._boxplot_with_outliers(x, y)
        elif analysis_option == "Pairwise Scatter Plot Matrix":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Pairwise Scatter Plot Matrix</h1>", unsafe_allow_html=True)
            variables = st.multiselect("Select Variables", self.data.columns)
            self._pairwise_scatter_matrix(variables)
        elif analysis_option == "Categorical Heatmap":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Categorical Heatmap</h1>", unsafe_allow_html=True)
            x = st.selectbox("Select X axis", self.data.columns)
            y = st.selectbox("Select Y axis", self.data.columns)
            self._categorical_heatmap(x, y)
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