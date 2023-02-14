import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
import random
# random.seed(42)
# warnings.filterwarnings('ignore')


def run_sequence_plot (x, y, title ,  mean_line = False, xlabel="time", ylabel="series"):
    """
    A function that plots a line plot for the given values
    parameters:
        x : Values to be plot in x a-xis
        y : values to plot on y-axis( must be same length to parameter x )
        title : Title  of the plot
        xlabel : Xlabel of the plot
        ylabel: Ylabel of the plot
    returns: y
        Line plot
    """
    plt.figure(figsize =(10 ,7))
    plt.plot(x, y, 'k-', label = ylabel)
    if(mean_line  == True):
        plt.axhline(y=np.nanmean(y), color = "red", label = f'mean(= {np.nanmean(y):.2f} )')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.xticks(rotation=45 , rotation_mode='anchor', ha ='right' )
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show();

def eval_metrics(actual_val, predicted_val):
    """
    Returns the MSE, MAE and RMSE Error for the given values
    Parameters:
    * actual_val    : The actual value of the data
    * predicted_val : The predicted value of the data
    returns:
    errors<Tuple> : A tuple of MSE MAE and RMSE score for the data (MSE, MAE, RMSE)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import warnings
    import random
    mse     = (np.square(actual_val - predicted_val)).mean()
    mae     = (np.abs(actual_val - predicted_val)).mean()
    rmse    = np.sqrt(mse)
    errors  = (mse,mae,rmse)
    return errors

def print_eval_metrics(actual_val, predicted_val):
    mse,mae,rmse = eval_metrics(actual_val, predicted_val)
    print(f"\n The Mean Squared Error : {mse}")
    print(f"\nThe Mean Absolute Error : {mae}")
    print(f"\nThe Root Mean Squared Error : {mse}")

def auto_model():
    model_autoARIMA = auto_arima(first_store_open["Sales"], start_p=0, start_q=0,
    test='adf',
    # use adftest to find osales_combined.Sales - sales_combined.predptimal 'd'
    max_p=3, max_q=3, # maximum p and q
    m=1,
    # frequency of series
    d=None,
    # let model determine 'd'
    seasonal=False,
    # No Seasonality
    start_P=0,
    D=0,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True)
    print(model_autoARIMA.summary())
    model_autoARIMA.plot_diagnostics(figsize=(15,8))
    plt.show()