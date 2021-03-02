import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from time import gmtime, strftime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


class suppress_stdout_stderr(object):
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))
  
    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class KPIForecaster():

    def __init__(self, conf):
		# store the configuration object
        self.conf = conf
        
    def makeDir(self, path):
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
            
            
    def getTrainingData(self, df_kpi, cell, KPI = 'DL_USER_THROUGHPUT_MBPS'):  
        
        # create blank dataframe
        df = pd.DataFrame()

        # Get cell specific info
        cell_df = df_kpi[df_kpi["CELL_NAME"] == cell].copy()

        # Convert to pandas datatime format
        df['ds'] = pd.to_datetime(cell_df['START_TIME'])
        #print(df.info())
        # Extract KPI
        df['y'] = cell_df[KPI]

        # Sort by date
        df = df.sort_values("ds")
        # Edit datatime format so we can remove timestamp format and filter by date (YY-MM-DD)
        df['Date'] = df['ds'].dt.strftime('%d/%m/%y')
        try:
            if self.conf["filter_training_period"] == "Yes":
                print("Filtering by Dates")
                df['Date'] = pd.to_datetime(df['Date'])
                start_date = self.conf["training_data_start_date"] 
                end_date = self.conf["training_data_end_date"] 
                mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
                df = df.loc[mask]
                df['y'].replace(0, np.nan, inplace=True)
                df['y'].fillna((df['y'].mean()), inplace=True)
        except:
            raise Exception("Invalid Date Format, please use YYYY-MM-DD")
        return df

    def getForecast(self, df):

        prophet = Prophet(changepoint_prior_scale = self.conf["changepoint_prior_scale"],
                        seasonality_mode= self.conf["seasonality_mode"]
                        )
                            
        future = pd.DataFrame()
        df['cap'] = 200
        df['floor'] = 0
        future['cap'] = 200
        future['floor'] = 0
        with suppress_stdout_stderr():
            m = prophet.fit(df)
        future = prophet.make_future_dataframe(periods=24*self.conf["forecast_days"], freq='H')
        forecast = prophet.predict(future)
        fig = prophet.plot(forecast)
        return prophet, forecast, m
    
    
    def saveModel(self, prophet, m, forecast, cell, KPI = "DL_USER_THROUGHPUT_MBPS"):
        import pickle
        #folder_name = strftime("%Y_%m_%d", gmtime())
        folder_name = "new"
        pkl_path = "./models/" + KPI +"/" + folder_name + "/" + str(cell ) + ".pkl"
        self.makeDir("./models/" + KPI +"/" + folder_name)
        with open(pkl_path, "wb") as f:
            # Pickle the 'Prophet' model using the highest protocol available.
            pickle.dump(m, f)

        # save the dataframe
        pkl_fore_cast_path = "./models/" + KPI +"/" + folder_name + "/" + str(cell ) + "_forecast.pkl"
        forecast.to_pickle(pkl_fore_cast_path)
        fig_file_name = "./models/" + KPI +"/" + folder_name + "/" + str(cell ) + "_plot.jpg"
        fig = prophet.plot(forecast)
        fig.savefig(fig_file_name, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
    
    def analyzeData(self, forecast, df_last_day, last_day, cell):
        #cell = "TNTAA405_L02A"
        forecast['Date'] = forecast['ds'].dt.strftime('%d/%m/%y')
        forecast['pred_upper_15'] = forecast['yhat_upper'] *(1+self.conf["threshold_margin"])
        forecast['pred_lower_15'] = forecast['yhat_lower'] * (1-self.conf["threshold_margin"])
        forecast['CELL_NAME'] = cell

        # Get last 24 hours
        forecast_last_day = forecast.loc[forecast['Date'] == last_day]
        forecast_last_day = forecast_last_day[['CELL_NAME','ds', 'Date','pred_upper_15','pred_lower_15','yhat']]

        result = pd.merge(forecast_last_day.reset_index(), df_last_day.reset_index(), on=['ds'], how='inner')
        foreLD = result[['CELL_NAME','ds', 'Date_x','pred_upper_15','pred_lower_15','yhat','y']]
        foreLD.columns = ['CELL_NAME','ds', 'Date','pred_upper_15','pred_lower_15','Expected_Value','Actual_Value']

        pd.options.mode.chained_assignment = None

        foreLD['Exceeds_Thresh'] = foreLD['Actual_Value'] >= foreLD['pred_upper_15']
        foreLD['Under_Thresh'] = foreLD['Actual_Value'] <= foreLD['pred_lower_15']
        foreLD.loc[(foreLD['Exceeds_Thresh'] == True) | (foreLD['Under_Thresh'] == True), 'Investigate_Cell'] = True 
        foreLD.loc[(foreLD['Under_Thresh'] != True) & (foreLD['Under_Thresh'] != True), 'Investigate_Cell'] = False 
        return foreLD
    
    def getForecastData(self, cell, KPI):
        mypath = "./models/" + KPI
        subfolder = [f.path for f in os.scandir(mypath) if f.is_dir()][0]

        file_names = [f for f in listdir(subfolder) if isfile(join(subfolder, f))]
        file_names.sort(key=lambda x: os.stat(os.path.join(subfolder, x)).st_mtime)
        file_name = cell + "_forecast.pkl"
        path = subfolder + "/" + file_name
        try:
            unpickled_df = pd.read_pickle(path)
            return unpickled_df
        except:
            raise Exception("Models not found")
    
    def getLastDay(self, df_kpi, KPI = 'DL_USER_THROUGHPUT_MBPS', cell = ''):  
        
           # create blank dataframe
        df = pd.DataFrame()

        # Get cell specific info
        cell_df = df_kpi[df_kpi["CELL_NAME"] == cell].copy()

        # Convert to pandas datatime format
        df['ds'] = pd.to_datetime(cell_df['START_TIME'])
        #print(df.info())
        # Extract KPI
        df['y'] = cell_df[KPI]

        # Sort by date
        df = df.sort_values("ds")
        
        
        # create blank dataframe
        #df = pd.DataFrame()
        # Get cell specific info
        #cell_df = df_kpi[df_kpi["CELL_NAME"] == cell]   
        # Convert to pandas datatime format
        #df['ds'] = pd.to_datetime(cell_df['START_TIME'])  
        # Extract KPI
        #df['y'] = cell_df[KPI]    
        # Sort by date
        df.sort_values("ds")    
        # Edit datatime format so we can remove timestamp format and filter by date (YY-MM-DD)
        df['Date'] = df['ds'].dt.strftime('%d/%m/%y')
        # Get last date
        last = df.tail(1)
        last_day = last.iloc[0]['Date']    
        # Get last 24 hours, we may need to change this
        df_last_day = df.loc[df['Date'] == last_day]
        #df_model = df.loc[df['Date'] != last_day]
        
        return df_last_day, last_day
    
    def baseLineNetworkModel(self, df_train):
        grouped = df_train[df_train['DL_USER_THROUGHPUT_MBPS'] != 0]
        grouped = grouped.groupby('START_TIME')
        grouped = grouped['DL_USER_THROUGHPUT_MBPS'].agg(np.mean)
        grouped = grouped.reset_index()

        # create blank dataframe
        df = pd.DataFrame()
        # Convert to pandas datatime format
        df['ds'] = pd.to_datetime(grouped['START_TIME'])
        # Extract KPI
        df['y'] = grouped['DL_USER_THROUGHPUT_MBPS']
        # Sort by date
        df.sort_values("ds")
        # Edit datatime format so we can remove timestamp format and filter by date (YY-MM-DD)
        df['Date'] = df['ds'].dt.strftime('%d/%m/%y')

        prophet = Prophet(changepoint_prior_scale = self.conf["changepoint_prior_scale"],
                seasonality_mode= self.conf["seasonality_mode"]
                )
        future = pd.DataFrame()

        df['cap'] = 200
        df['floor'] = 0
        future['cap'] = 200
        future['floor'] = 0
        with suppress_stdout_stderr():
            m = prophet.fit(df)
        future = prophet.make_future_dataframe(periods=24*self.conf["forecast_days"], freq='H')
        forecast = prophet.predict(future)
        fig = prophet.plot(forecast)
        
        cell = "NETWORK_BASELINE_MODEL"          
        KPI = "DL_USER_THROUGHPUT_MBPS"
        #folder_name = strftime("%Y_%m_%d", gmtime())
        folder_name = "new"
        self.makeDir("./models/" + KPI +"/" + folder_name)
        fig_file_name = pkl_path = "./models/" + KPI +"/" + folder_name + "/" + str(cell) + "_plot.jpg"
        fig.savefig(fig_file_name, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        self.saveModel(prophet, m, forecast, cell = cell, KPI = "DL_USER_THROUGHPUT_MBPS")