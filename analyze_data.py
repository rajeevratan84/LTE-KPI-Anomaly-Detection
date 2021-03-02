#!/usr/bin/env python
from KPIForecaster.forecaster import KPIForecaster
from configuration.settings import Conf
from database.sql_connect import SQLDatabase
from datetime import datetime
import pandas as pd
import sys
import os.path
import time

path = sys.argv[0].rsplit("/", 1)[0]

# Create configuration and Database connection and our KPI Forecaster Object
#conf = Conf(os.path.join(path,"config.json"))
try:
    conf = Conf(os.path.join(path,"config.json"))
except:
    conf = Conf("config.json")
    
sql = SQLDatabase(conf)
KPIForecaster = KPIForecaster(conf, crontab=True)
StoreForecast = False

#input_report = pd.read_csv("DAILY_ANOMALY_REPORT_DL_USER_THROUGHPUT_MBPS_2020_12_13.csv")
#del input_report['Unnamed: 0']
input_df = sql.getDailyKPIData()
input_report = KPIForecaster.getYesterdaysReport(input_df)
#input_report = sql.getYesterdaysReport()

# Ensure all columns are uppercase
input_report.columns = [x.upper() for x in input_report.columns]

# Get unique cell IDs
cell_names = input_report.CELL_NAME.unique()

print(f'[INFO] Analysing {len(cell_names)} Models')

T_START = time.time()
appended_data = []
full_forecast = []

KPI = 'DL_USER_THROUGHPUT_MBPS'

# Iterate through each cell, creating a model, forecast and plot for each
for i,cell_name in enumerate(cell_names):
    df_last_day, last_day = KPIForecaster.getLastDay(input_report, cell = cell_name)
    ret, forecast = KPIForecaster.getForecastData(cell_name, KPI = KPI)
    
    if ret:
        foreLD, long_forecast = KPIForecaster.analyzeData(forecast, df_last_day, last_day, cell = cell_name)
        print(str(i+1) + " of " + str(len(cell_names)) + " cells processed.")
        appended_data.append(foreLD)
        full_forecast.append(long_forecast)

    #if i == 2:
    #   break

# Concatenate all dataframes from appended_data list
appended_data = pd.concat(appended_data, axis=0)

full_forecast = pd.concat(full_forecast, axis=0)

# Rename columns as per SQL DWH naming convention
appended_data = appended_data.rename({'ds':'START_TIME',
                      'Date':'DATE',
                      'pred_upper_15':'HISTORICAL_UPPER_BOUND',
                      'pred_lower_15':'HISTORICAL_LOWER_BOUND',
                      'Expected_Value':'HISTORICAL_PREDICTION',
                      'Actual_Value':'ACTUAL_VALUE',
                      'Exceeds_Thresh':'EXCEEDS_THRESHOLD',
                      'Under_Thresh':'UNDER_THRESHOLD',
                      'Investigate_Cell':'OUT_OF_RANGE',
                      'Delta':'DELTA_FROM_HIST_PREDICTION',
                      'Delta_from_Bound':'DELTA_FROM_HIST_BOUND' 
                     }, axis='columns')

# Change datatypes to string
appended_data['START_TIME'] = appended_data['START_TIME'].astype(str)
appended_data['EXCEEDS_THRESHOLD'] = appended_data['EXCEEDS_THRESHOLD'].astype(str)
appended_data['UNDER_THRESHOLD'] = appended_data['UNDER_THRESHOLD'].astype(str)
appended_data['OUT_OF_RANGE'] = appended_data['OUT_OF_RANGE'].astype(str)
appended_data = appended_data.fillna(0)    
appended_data['KEY'] = appended_data['CELL_NAME'] + appended_data['START_TIME']

# Get AI Predictions
predictions = KPIForecaster.getPredictions(input_df)
fin = pd.merge(appended_data, predictions, on=['KEY'], how='inner')
final = fin[['CELL_NAME',
            'START_TIME',
            'DATE',
            'HISTORICAL_UPPER_BOUND',
            'HISTORICAL_LOWER_BOUND',
            'EXCEEDS_THRESHOLD',
            'UNDER_THRESHOLD',
            'OUT_OF_RANGE',
            'DELTA_FROM_HIST_PREDICTION',
            'DELTA_FROM_HIST_BOUND',
             0,
            'ACTUAL_VALUE',
            'HISTORICAL_PREDICTION']].copy()

final = final.rename({0:'AI_PREDICTION',
                      'DELTA_FROM_HIST_PREDICTION':'PCT_DELTA_FROM_HIST_PREDICTION',
                      'DELTA_FROM_HIST_BOUND':'PCT_DELTA_FROM_HIST_BOUND'}, axis='columns')

final['DELTA_FROM_AI_PREDICTION'] = final['ACTUAL_VALUE'] - final['AI_PREDICTION']
final['DELTA_FROM_HIST_PREDICTION'] = final['ACTUAL_VALUE'] - final['HISTORICAL_PREDICTION']

#final = final[['CELL_NAME', 'START_TIME', 'DATE', 'HISTORICAL_UPPER_BOUND', 'HISTORICAL_LOWER_BOUND',
#                '0', '1', '2', '3', '0', '1', '2', '3']]

final['START_TIME'] = pd.to_datetime(final['START_TIME'])
final['DATE'] = final['START_TIME'].dt.strftime('%m/%d/%Y')
final['START_TIME'] = final['START_TIME'].dt.strftime('%H:%M:%S')
final['START_TIME'] = final['START_TIME'].astype(str)
final['DATE'] = final['DATE'].astype(str)

# Add Maintenance Window filter
maintenance_window = ['00:00:00','01:00:00','02:00:00' ,'03:00:00' ,'04:00:00','05:00:00']
final['MAINTENANCE_WINDOW'] = final['START_TIME'].isin(maintenance_window)

# Output Statistics 
t0 =  time.time()
completion_time = t0-T_START
print("******* Total Time to Produce Reports: " + str(completion_time)) 
print("******* Average Time Per Model " + str(completion_time/len(cell_names))) 

path = os.path.join(path,"./Reports/ANOMALY/")
KPIForecaster.makeDir(path)
date = datetime.today().strftime('%Y_%m_%d')
file_name = path + "DAILY_ANOMALY_REPORT_" + KPI + "_" + str(date) + ".csv" 

appended_data.to_csv(file_name)
print("[INFO] Analysis Completed.")
print("[INFO] Uploading Report to DWH...")

sql.dumpToDWH(final, "KPI_ANOMALY")

## This should be in Train Model ##
if StoreForecast == True:
    full_forecast_df = full_forecast[['CELL_NAME', 'ds',
                        'pred_upper_15','pred_lower_15','yhat']].copy()
    full_forecast_df = full_forecast_df.rename({'ds':'TIMESTAMP',
                        'yhat':'PREDICTED',
                        'pred_upper_15':'UPPER_PREDICTION',
                        'pred_lower_15':'LOWER_PREDICTION'                                            
                        }, axis='columns')
    full_forecast_df['TIMESTAMP'] = full_forecast_df['TIMESTAMP'].astype(str)
    sql.dumpToDWH(full_forecast_df, "FORECAST_DATA", if_exists = 'append')
