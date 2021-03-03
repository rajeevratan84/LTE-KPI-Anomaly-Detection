from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from time import gmtime, strftime
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
import xgboost as xgb
import os.path
import time
import shutil
import sys
import os.path

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

    def __init__(self, conf, crontab = False):
		# store the configuration object
        self.conf = conf
        self.path = sys.argv[0].rsplit("/", 1)[0]
        self.crontab = crontab
        print(self.path)
        
        
    def makeDir(self, path):
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
            
    def filterDates(self, df, filter_from_config = False):  
        #df = df['START_TIME'] = df['START_TIME'].str.replace('\(UTC-04:00\)', '')
        df['START_TIME'] = df['START_TIME'].str.replace('\(UTC-04:00\)', '')
        # Convert to pandas datatime format
        df['ds'] = pd.to_datetime(df['START_TIME'])
        
        # Sort by date
        df = df.sort_values("ds")
        # Edit datatime format so we can remove timestamp format and filter by date (YY-MM-DD)
        df['Date'] = df['ds'].dt.strftime('%d/%m/%y')
        
        if filter_from_config:
            try:
                if self.conf["filter_training_period"].upper() == "YES":
                    
                    start = self.conf["training_data_start_date"]  + ' 00:00:00(UTC-04:00)'
                    end = self.conf["training_data_end_date"]  + ' 23:00:00(UTC-04:00)'
                    print(f"[INFO] Training on data from {start} to {end}...")
                    
                    df['Date'] = pd.to_datetime(df['Date'])
                    start_date = self.conf["training_data_start_date"] 
                    end_date = self.conf["training_data_end_date"] 
                    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date).copy()
                    df = df.loc[mask]
            except:
                raise Exception("Invalid Date Format, please use YYYY-MM-DD")
        return df
    
            
    def getTrainingData(self, df_kpi, cell, KPI = 'DL_USER_THROUGHPUT_MBPS'):  
        
        # create blank dataframe
        df = pd.DataFrame()
        # Get cell specific info
        cell_df = df_kpi[df_kpi["CELL_NAME"] == cell].copy()
        # Convert to pandas datatime format
        df['ds'] = pd.to_datetime(cell_df['START_TIME'])
        # Extract KPI
        df['y'] = cell_df[KPI]
        # Sort by date
        df = df.sort_values("ds")
        # Edit datatime format so we can remove timestamp format and filter by date (YY-MM-DD)
        df['Date'] = df['ds'].dt.strftime('%d/%m/%y')
        df['y'].replace(0, np.nan, inplace=True)
        df['y'].fillna((df['y'].mean()), inplace=True)
        return df

    def getForecast(self, df):

        prophet = Prophet(changepoint_prior_scale = self.conf["changepoint_prior_scale"],
                          seasonality_mode = self.conf["seasonality_mode"])
                            
        future = pd.DataFrame()
        df['cap'] = 300
        df['floor'] = 0
        future['cap'] = 300
        future['floor'] = 0
        
        with suppress_stdout_stderr():
            m = prophet.fit(df)
            
        future = prophet.make_future_dataframe(periods=24*self.conf["forecast_days"], freq='H')
        forecast = prophet.predict(future)
        #fig = prophet.plot(forecast)
        return prophet, forecast, m
    
    
    def saveModel(self, prophet, m, forecast, cell, KPI = "DL_USER_THROUGHPUT_MBPS"):
        import pickle
        #folder_name = strftime("%Y_%m_%d", gmtime())
        folder_name = "NEWEST"
        
        if self.crontab :
            pkl_path = os.path.join(self.path,"./models/" + KPI +"/" + folder_name + "/" + str(cell) + "_model.pkl")
            plot_path = os.path.join(self.path,"./models/" + KPI +"/plots/" + str(cell) + ".pkl")
            self.makeDir(os.path.join(self.path,"./models/" + KPI +"/" + folder_name))
            self.makeDir(os.path.join(self.path,"./models/" + KPI +"/plots/" + folder_name))
        else:    
            pkl_path =  "./models/" + KPI +"/" + folder_name + "/" + str(cell) + "_model.pkl"
            plot_path = "./models/" + KPI +"/plots/" + str(cell) + ".pkl"
            self.makeDir("./models/" + KPI +"/" + folder_name)
            self.makeDir("./models/" + KPI +"/plots/" + folder_name)
            
        with open(pkl_path, "wb") as f:
            # Pickle the 'Prophet' model using the highest protocol available.
            pickle.dump(m, f)

        # save the dataframe
        if self.crontab :
            pkl_fore_cast_path = os.path.join(self.path,"./models/" + KPI +"/" + folder_name + "/" + str(cell ) + "_forecast.pkl")
        else:
            pkl_fore_cast_path = "./models/" + KPI +"/" + folder_name + "/" + str(cell ) + "_forecast.pkl"
            
        forecast.to_pickle(pkl_fore_cast_path)
        if self.conf["save_images"].upper() == "YES":
            fig_file_name = os.path.join(self.path,"./models/" + KPI +"/plots/" + str(cell ) + "_plot.jpg")
            fig = prophet.plot(forecast)
            fig.savefig(fig_file_name, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
    
    def analyzeData(self, forecast, df_last_day, last_day, cell):
        #cell = "TNTAA405_L02A"
        forecast['Date'] = forecast['ds'].dt.strftime('%d/%m/%y')
        forecast['pred_upper_15'] = forecast['yhat_upper'] *(1+self.conf["threshold_margin"])
        forecast['pred_lower_15'] = forecast['yhat_lower'] * (1-self.conf["threshold_margin"])
        forecast['pred_lower_15'].values[forecast['pred_lower_15'].values < 0] = 0

        forecast['CELL_NAME'] = cell

        # Get last 24 hours
        forecast_last_day = forecast.loc[forecast['Date'] == last_day]
        forecast_last_day = forecast_last_day[['CELL_NAME','ds', 'Date','pred_upper_15','pred_lower_15','yhat']]

        result = pd.merge(forecast_last_day.reset_index(), df_last_day.reset_index(), on=['ds'], how='inner')
        foreLD = result[['CELL_NAME','ds', 'Date_x','pred_upper_15','pred_lower_15','yhat','y']]
        foreLD.columns = ['CELL_NAME','ds', 'Date','pred_upper_15','pred_lower_15','Expected_Value','Actual_Value']

        pd.options.mode.chained_assignment = None
        foreLD['Expected_Value'].values[foreLD['Expected_Value'].values < 0] = 0
        foreLD['Exceeds_Thresh'] = foreLD['Actual_Value'] >= foreLD['pred_upper_15']
        foreLD['Under_Thresh'] = foreLD['Actual_Value'] <= foreLD['pred_lower_15']
        
        foreLD.loc[(foreLD['Exceeds_Thresh'] == True) | (foreLD['Under_Thresh'] == True), 'Investigate_Cell'] = True
        foreLD['Investigate_Cell'].fillna('False', inplace=True)
        foreLD['Delta'] = foreLD['Actual_Value'] / foreLD['Expected_Value']
        foreLD['Delta'].replace(np.inf, 0, inplace=True)
        

        foreLD['Exceeds_Thresh'] = foreLD['Exceeds_Thresh'].apply(lambda x: 1 if x == True else 0)
        foreLD['Under_Thresh'] = foreLD['Under_Thresh'].apply(lambda x: 1 if x == True else 0)
        foreLD['Investigate_Cell'] = foreLD['Investigate_Cell'].apply(lambda x: 1 if x == True else 0)
        
        foreLD.loc[foreLD['Under_Thresh'] == True, 'Delta_from_Bound'] = (foreLD['pred_lower_15'] - foreLD['Actual_Value'])/foreLD['pred_lower_15']
        foreLD.loc[foreLD['Exceeds_Thresh'] == True, 'Delta_from_Bound'] = (foreLD['pred_upper_15'] - foreLD['Actual_Value'])/foreLD['pred_upper_15']
        foreLD['Delta_from_Bound'] = foreLD['Delta_from_Bound'].abs() 
        foreLD['Delta_from_Bound'].replace(np.inf, 0, inplace=True)
        return foreLD, forecast
    
    def getForecastData(self, cell, KPI):
        mypath = "models/" + KPI + "/NEWEST"

        file_name = cell + "_forecast.pkl"
        fpath = mypath + "/" + file_name
        if self.crontab :
            final_path = os.path.join(self.path,fpath)
        else:
            final_path = fpath
            pass
        #print(final_path)
        try:
            
            unpickled_df = pd.read_pickle(final_path)
            return True, unpickled_df
        except:
            #raise Exception("Models not found")
            print(f'This model {file_name} not found')
            return False, None
            
            
    def getLastDay(self, df_kpi, KPI = 'DL_USER_THROUGHPUT_MBPS', cell = ''):  
        
        # create blank dataframe
        df = pd.DataFrame()
        # Get cell specific info
        cell_df = df_kpi[df_kpi["CELL_NAME"] == cell].copy()
        cell_df['START_TIME'] = cell_df['START_TIME'].str.replace('\(UTC-04:00\)', '')
        # Convert to pandas datatime format
        df['ds'] = pd.to_datetime(cell_df['START_TIME'])
        #print(df.info())
        # Extract KPI
        df['y'] = cell_df[KPI]
        # Sort by date
        df = df.sort_values("ds")
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
        import pickle
        grouped = df_train[df_train['DL_USER_THROUGHPUT_MBPS'] != 0]
        grouped = grouped.groupby('START_TIME')
        grouped = grouped['DL_USER_THROUGHPUT_MBPS'].agg(np.mean)
        grouped = grouped.reset_index()

        # create blank dataframe
        df = pd.DataFrame()
        # Convert to pandas datatime format
        df['ds'] = pd.to_datetime(grouped['START_TIME']).copy()
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

        if self.crontab:
            self.makeDir(os.path.join(self.path, "./models/NetworkBaseline/" + KPI +"/"))
            fig_file_name = os.path.join(self.path, "./models/NetworkBaseline/" + KPI + "/" + str(cell) + "_plot.jpg")
        else:
            self.makeDir("./models/NetworkBaseline/" + KPI +"/")
            fig_file_name = "./models/NetworkBaseline/" + KPI + "/" + str(cell) + "_plot.jpg"
        
        fig.savefig(fig_file_name, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        if self.crontab :
            pkl_path = os.path.join(self.path, "./models/NetworkBaseline/" + KPI +"/"  + "/" + str(cell) + "_model.pkl")
        else:
            pkl_path = "./models/NetworkBaseline/" + KPI +"/"  + "/" + str(cell) + "_model.pkl"
            
        with open(pkl_path, "wb") as f:
            # Pickle the 'Prophet' model using the highest protocol available.
            pickle.dump(m, f)

        # save the dataframe
        if self.crontab :
            pkl_fore_cast_path = os.path.join(self.path, "./models/NetworkBaseline/" + KPI + "/" + str(cell ) + "_forecast.pkl")
        else:
            pkl_fore_cast_path = "./models/NetworkBaseline/" + KPI + "/" + str(cell ) + "_forecast.pkl"
        forecast.to_pickle(pkl_fore_cast_path)
        #self.saveModel(prophet, m, forecast, cell = cell, KPI = "DL_USER_THROUGHPUT_MBPS")
        
        
    def getPredictions(self, df):
        df.columns = map(str.upper, df.columns)
        '''Takes SQL dump of KPIs as inputs and outputs the predictions for DL Throughput'''
        df['START_TIME'] = df['START_TIME'].str.replace('\(UTC-04:00\)', '')
        df['KEY'] = df['CELL_NAME'] + df['START_TIME']

        print(df.head())
        # extract relavent fields 
        input_data = df[['MAX_PHYS_DL_THROUGHPUT_MBPS',
                    'DL_64QAM_MODULATION_UTIL',
                    'GOOD_CQI_10_15',
                    'DL_PRB_UTILIZATION',
                    'BAD_CQI_0_6',
                    'UL_PRB_UTILIZATION',
                    'TOTAL_DATA_VOLUME_GB',
                    'DL_DATA_VOLUME_GB',
                    'LATENCY_MS',
                    'INTERFERENCE_DBM']].copy()

        model = xgb.Booster()
        if self.crontab:
            model.load_model(os.path.join(self.path,"my_model_reduced_features.model"))
        else:
            model.load_model("my_model_reduced_features.model")
            
        xgb_input = xgb.DMatrix(input_data)
        preds = model.predict(xgb_input)
        preds = pd.DataFrame(preds)
        keys = df[['KEY']].copy()
        
        predictions = pd.concat([keys, preds], axis = 1)
        return predictions
    
    def getYesterdaysReport(self, df):
        input_data = df[['START_TIME',
            'CELL_NAME',
            'DL_USER_THROUGHPUT_MBPS']].copy()
        return input_data
            
                
    def findDegradation(self, df, weeks = 3):
        df_prev = df.shift(1)['DEGRADED']
        df_next = df.shift(-1)['DEGRADED']
        df_next2 = df.shift(-2)['DEGRADED']
        df_next3 = df.shift(-3)['DEGRADED']
        df_next4 = df.shift(-4)['DEGRADED']
        
        if weeks == 3:
            df.loc[(df_prev != 1) & (df['DEGRADED'] == 1) & (df_next == 1) & (df_next2 == 1), 'FLAG'] = 1
            #df.loc[(df['Degrade'] != 0) & (df_next == 0) & (df_next3 == 0), 'end'] = 1
        else:
            df.loc[(df_prev != 1) & (df['DEGRADED'] == 1) & (df_next == 1) & (df_next2 == 1) & (df_next3 == 1), 'FLAG'] = 1
            #df.loc[(df['Degrade'] != 0) & (df_next == 0) & (df_next4 == 0), 'end'] = 1

        df.fillna(0, inplace=True)
        df['FLAG'] = df['FLAG'].astype(int)
        #df['end'] = df['end'].astype(int)
        return df

    def getConsecutiveSequences(self, df):
        i = 0
        ind = []

        for index, row in df.iterrows():
            if row['FLAG'] == 1:
                ind.append(i)
            i += 1

        for i in ind:
            for j in range(1,7):
                s = i+j
                if df.iloc[i+j,4] == 1:
                    df.iloc[s,5] = 1
                else:
                    break
        return df
    
    def getSummaryReport(self, df):
        dates = df['DATE'].unique()
        dates = pd.to_datetime(dates)
        dates = dates.sort_values()
        dates = dates[-4:]
        dates = pd.DataFrame(dates)
        dates[0] = dates[0].dt.strftime('%Y-%m-%d')
        recent = list(dates[0])
        flagged_only_df = df[df['FLAG'] == 1]
        recent_df = flagged_only_df[flagged_only_df['DATE'].isin(recent)]
        recent_df = recent_df.groupby(['CELL_NAME']).mean().reset_index()
        recent_df = recent_df[['CELL_NAME', 'DL_USER_THROUGHPUT_MBPS_AVERAGE',
                                     'DL_USER_THROUGHPUT_MBPS_PCT_CHANGE','FLAG']]
        recent_df.loc[recent_df.FLAG > 0, 'FLAG'] = 1
        return recent_df