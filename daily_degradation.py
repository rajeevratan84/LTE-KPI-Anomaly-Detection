#!/usr/bin/env python
from configuration.settings import Conf
from database.sql_connect import SQLDatabase
from KPIForecaster.forecaster import KPIForecaster
from datetime import datetime
import pandas as pd
import numpy as np
import time
import sys
import os.path

path = sys.argv[0].rsplit("/", 1)[0]

# Create configuration and Database connection and our KPI Forecaster Object
try:
    conf = Conf(os.path.join(path,"config.json"))
except:
    conf = Conf("config.json")
    
sql = SQLDatabase(conf)
# Creating out KPI Forecaster Object
KPIForecaster = KPIForecaster(conf)

#df_train = pd.read_csv('FT_CELL_NOV.csv')

# Starting Timer for benchmarking
T_START = time.time()
df_train = sql.getHourlyKPIReportXDays(160)
t0 =  time.time()
completion_time = t0-T_START

print(f'[INFO] Total Time to Download Report: {completion_time}') 
print("[INFO] Report Loaded")

# Replace UTC string from time
df_train['START_TIME'] = df_train['START_TIME'].str.replace('\(UTC-04:00\)', '')

# Set KPI here
KPI = 'DL_USER_THROUGHPUT_MBPS'

cell_names = df_train.CELL_NAME.unique()

df_train['START_TIME'] = pd.to_datetime(df_train['START_TIME'])
df_train['DATE'] = df_train['START_TIME'].dt.date


df = pd.DataFrame()
appended_data = []
number_of_cells = len(cell_names)

for (i,cell_name) in enumerate(cell_names):
    
    df = df_train[df_train["CELL_NAME"] == cell_name]
    df2 = df.groupby(['CELL_NAME','DATE']).mean().pct_change().reset_index()
    df2['KEY'] = df2['CELL_NAME'] + df2['DATE'].astype(str)
    
    df3 = df.groupby(['CELL_NAME','DATE']).mean().reset_index()
    df3['KEY'] = df3['CELL_NAME'] + df3['DATE'].astype(str)
    df3 = df3[['DL_USER_THROUGHPUT_MBPS', 'KEY']].copy()
    df4 = pd.merge(df2, df3, on='KEY')
    df2 = df4.rename({"DL_USER_THROUGHPUT_MBPS_x": "DL_USER_THROUGHPUT_MBPS_PCT_CHANGE",
                 "DL_USER_THROUGHPUT_MBPS_y": "DL_USER_THROUGHPUT_MBPS_AVERAGE"
                 }, axis='columns')
    df2 = df2[['CELL_NAME', 'DL_USER_THROUGHPUT_MBPS_PCT_CHANGE','DATE',
              'DL_USER_THROUGHPUT_MBPS_AVERAGE']]
    df2 = df2.fillna(0)
    df2['DEGRADED'] = df2['DL_USER_THROUGHPUT_MBPS_PCT_CHANGE'].apply(lambda x: 1 if x <= -0.05 else 0)
    df2 = KPIForecaster.findDegradation(df2, 4)
    appended_data.append(df2)
    print(f'[INFO] {i+1} of {number_of_cells} completed.')
    
    #if i == 40:
    #    break
          
appended_data = pd.concat(appended_data, axis=0)

name = KPI + "_PCT_CHANGE"
appended_data = appended_data.rename({KPI: name,}, axis='columns')

result = appended_data.sort_values(['CELL_NAME','DATE'])

T_START = time.time()
final = KPIForecaster.getConsecutiveSequences(result)
t0 =  time.time()
completion_time = t0-T_START
print(f'[INFO] Total Time to Iterate Report: {completion_time}') 

final = final.fillna(0)

# Saving and Uploading to DWH
#path = "./Reports/DEGRADATION/"
#KPIForecaster.makeDir(path)
#date = datetime.today().strftime('%Y_%m_%d')

#file_name = path + "DAILY_DEGRADATION_REPORT_" + KPI + "_" + str(date) + ".csv" 

#final.to_csv(file_name)

# Clean and upload Final Report
print("[INFO] Uploading Report to DWH.")
final['DL_USER_THROUGHPUT_MBPS_PCT_CHANGE'].replace(np.inf, 0, inplace=True)
final = final[['CELL_NAME', 'DATE', 'DL_USER_THROUGHPUT_MBPS_AVERAGE',
               'DL_USER_THROUGHPUT_MBPS_PCT_CHANGE', 'DEGRADED','FLAG']]

final['CELL_NAME'] = final['CELL_NAME'].astype(str)
final['DATE'] = final['DATE'].astype(str)

sql.deleteTable("KPI_DEGRADATION_DAILY")
sql.dumpToDWH(final, "KPI_DEGRADATION_DAILY", if_exists = 'append')

# Create and upload Summary Report
summary = KPIForecaster.getSummaryReport(final)

sql.deleteTable("KPI_DEGRADATION_DAILY_SUMMARY")
sql.dumpToDWH(summary, "KPI_DEGRADATION_DAILY_SUMMARY", if_exists = 'append')
