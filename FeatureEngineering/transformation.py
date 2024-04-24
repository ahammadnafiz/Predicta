import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import plotly.express as px
import scipy.stats as stats

def diagnostic_plots(data, variable):
    # Function to plot a histogram and a Q-Q plot side by side for a certain variable
    
    qq_plot = px.scatter(x=stats.probplot(data.columns[variable], dist="norm")[0],
                         y=stats.probplot(data.columns[variable], dist="norm")[1],
                         trendline="ols")
    qq_plot.show()


def log_transform(data, cols=[]):
    """
    Logarithmic transformation
    """
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    log_transformed_data = log_transformer.fit_transform(data[cols])
    log_transformed_data = pd.DataFrame(log_transformed_data, columns=[col+'_log' for col in cols])
    for col in log_transformed_data.columns:
        print('Variable ' + col[:-4] +' Q-Q plot')
        diagnostic_plots(log_transformed_data, col)       
    return log_transformed_data 


def reciprocal_transform(data, cols=[]):
    """
    Reciprocal transformation
    """
    reciprocal_transformer = FunctionTransformer(lambda x: 1/x, validate=True)
    reciprocal_transformed_data = reciprocal_transformer.fit_transform(data[cols])
    reciprocal_transformed_data = pd.DataFrame(reciprocal_transformed_data, columns=[col+'_reciprocal' for col in cols])
    for col in reciprocal_transformed_data.columns:
        print('Variable ' + col[:-11] +' Q-Q plot')
        diagnostic_plots(reciprocal_transformed_data, col)       
    return reciprocal_transformed_data 


def square_root_transform(data, cols=[]):
    """
    Square root transformation
    """
    sqrt_transformer = FunctionTransformer(np.sqrt, validate=True)
    sqrt_transformed_data = sqrt_transformer.fit_transform(data[cols])
    sqrt_transformed_data = pd.DataFrame(sqrt_transformed_data, columns=[col+'_square_root' for col in cols])
    for col in sqrt_transformed_data.columns:
        print('Variable ' + col[:-12] +' Q-Q plot')
        diagnostic_plots(sqrt_transformed_data, col)        
    return sqrt_transformed_data 


def exp_transform(data, coef, cols=[]):
    """
    Exponential transformation
    """
    exp_transformer = FunctionTransformer(lambda x: np.power(x, coef), validate=True)
    exp_transformed_data = exp_transformer.fit_transform(data[cols])
    exp_transformed_data = pd.DataFrame(exp_transformed_data, columns=[col+'_exp' for col in cols])
    for col in exp_transformed_data.columns:
        print('Variable ' + col[:-4] +' Q-Q plot')
        diagnostic_plots(exp_transformed_data, col)         
    return exp_transformed_data 


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(0)
    data = pd.DataFrame({
        'variable1': np.random.normal(loc=100, scale=20, size=1000),
        'variable2': np.random.uniform(low=1, high=100, size=1000)
    })

    # Apply transformations
    transformed_data = log_transform(data, ['variable1'])
    transformed_data = reciprocal_transform(data, ['variable2'])
    transformed_data = square_root_transform(data, ['variable1'])
    transformed_data = exp_transform(data, 0.5, ['variable2'])
