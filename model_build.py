import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from numpy.linalg import LinAlgError

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import warnings
warnings.filterwarnings('ignore')


class Model():
    def __init__(self, df:pd.DataFrame, sid, pn:str):
        self.df = df
        self.ar_model = None
        self.train_data, self.test_data = self.prepare_train_test()
        self.sid, self.pn = sid, pn
        self.fig, self.axs = plt.subplots(8, 1, figsize=(25, 25))
        self.axs = self.axs.flatten()


    # Splitting the data into train and test data.
    def prepare_train_test(self):
        train_data = self.df['Revenue'][:len(self.df)-24]
        test_data = self.df['Revenue'][len(self.df)-24:]
        print(f"Train and Test data prepared.")
        return train_data, test_data
    
    def check_stationarity(self, s:pd.Series) -> bool:
        print("Checking Stationarity")
        result = adfuller(s)

        if result<=0.05:
            return True
        else:
            return False


    def differencing(self, s:pd.Series):
        return s.diff().dropnna
        # return df

    def get_pacf_orders(self, s, max_order=10, alpha=0.05) -> int:
        pacf_, confint = pacf(s, nlags=max_order, alpha=alpha)

        for i, (lb, ub) in enumerate(confint):
            if pacf_[i] < lb or pacf_[i] > ub:
                continue
            return i
        return max_order


    def fit_predict(self, train_data):
        pacf_order = self.get_pacf_orders(train_data)
        print("Started Fitting")
        self.ar_model = AutoReg(train_data, lags=pacf_order).fit()
        pred = self.ar_model.predict(start=len(train_data), end=len(train_data) + len(self.test_data) - 1, dynamic=True)
        pred = np.where(pred<0, 0, pred)

        return pred

    def plot_feature(self, ax:np.ndarray, feature:str, plot_type=None) -> None:
        if plot_type=='bar':
            self.df[feature].plot(kind='bar', ax=ax)
        else:

            ax.plot(self.df[feature])
        ax.set_xlabel('Date')
        ax.set_ylabel(feature)
        ax.set_title(f"{feature}.")
        # ax.set_xticklabels(df.date.values, rotation=10, ha='right')

    def evaluate(self, pred:np.ndarray):
        residuals = self.test_data - pred
        rmse = np.sqrt(np.mean(residuals**2))
        mape = round(np.mean(abs(residuals/self.test_data)), 4)
        return rmse, mape

    def forecast(self):
        future_predictions = self.ar_model.forecast(24)
        future_predictions = np.where(future_predictions<0, 0, future_predictions)
        return future_predictions
    
    def create_plots(self, preds, forecast_vals):
        print("Started Plotting")
        #plot features and acf
        # fig, axs = plt.subplots(8, 1, figsize=(25, 25))
        # axs = axs.flatten()
        self.plot_feature(self.axs[0], 'Delivered')
        self.plot_feature(self.axs[1], "Clicked")
        self.plot_feature(self.axs[2], 'Converted')
        self.plot_feature(self.axs[3], 'Revenue')
        plot_acf(self.train_data, ax=self.axs[4], lags=10)
        self.axs[5].plot(self.test_data.index, self.test_data, label="Actual Data",)
        self.axs[5].plot(self.test_data.index, preds, label="Predictions",)
        self.axs[5].set_xlabel("Date")
        self.axs[5].set_ylabel("Actual vs Predicted")
        # self.axs[5].legend()

        residuals = self.test_data - preds
        # plot residulas
        self.axs[6].plot(residuals.index, residuals)
        self.axs[6].set_xlabel("Date")
        self.axs[6].set_ylabel("Residuals")
        # self.axs[6].legend()

        prognose = np.concatenate((self.df['Revenue'],forecast_vals))
        self.axs[7].plot(range(1,len(prognose)+1), prognose, 'g')
        self.axs[7].plot(range(1,len(self.df)+1), self.df['Revenue'], 'b')
        self.axs[7].set_xlabel("Date")
        self.axs[7].set_ylabel("Forecast")
        plt.tight_layout()
        plt.close()

        print("Saved")
        print("Plotting Completed.")

    def save_plots(self, path:str):

        self.fig.savefig(f"{path}/image/{self.sid}_{self.pn}")
        print("Saved")

    def train(self):
        # while not self.check_stationarity(self.train_data):
        #     self.train_data = self.differencing(self.train_data)
        #     if self.check_stationarity(self.train_data):
        #         break
        print("Data is Stationarity")
        #fit and predict
        preds = self.fit_predict(self.train_data)
        rmse, mape = self.evaluate(preds)

        #forecast 24 hours
        forecast_vals = self.forecast()
        
        #creating plots
        self.create_plots(preds, forecast_vals)
        self.save_plots('ar_model1')
        return rmse, mape, forecast_vals




