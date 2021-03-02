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

def findDegradation(df, weeks = 3):
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

def getConsecutiveSequencesWeekly(df):
    i = 0
    ind = []

    for index, row in df.iterrows():
        if row['FLAG'] == 1:
            ind.append(i)
        i += 1

    for i in ind:
        for j in range(1,7):
            s = i+j
            if df.iloc[i+j,5] == 1:
                df.iloc[s,6] = 1
            else:
                break
    return df

def getSummaryReport(df):
        dates = df['START_DATE'].unique()
        dates = pd.to_datetime(dates)
        dates = dates.sort_values()
        dates = dates[-3:]
        dates = pd.DataFrame(dates)
        dates[0] = dates[0].dt.strftime('%Y-%m-%d')
        recent = list(dates[0])
        flagged_only_df = df[df['FLAG'] == 1]
        recent_df = flagged_only_df[flagged_only_df['START_DATE'].isin(recent)]
        recent_df = recent_df.groupby(['CELL_NAME']).mean().reset_index()
        recent_df = recent_df[['CELL_NAME', 'DL_USER_THROUGHPUT_MBPS_AVERAGE',
                                     'DL_USER_THROUGHPUT_MBPS_PCT_CHANGE']]
        #recent_df.loc[recent_df.FLAG > 0, 'FLAG'] = 1
        return recent_df


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
#df_train = sql.getHourlyKPIReportDegradation()
df_train = sql.getHourlyKPIReportXDays()
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
df_train['Week_Number'] = df_train['START_TIME'].dt.isocalendar().week
df_train['Year'] = df_train['START_TIME'].dt.year
df_train['YEAR_WEEK'] = df_train['Year'].astype(str) + "_" + df_train['Week_Number'].astype(str)

df_train['START_DATE'] = df_train['START_TIME'].dt.date
df_train['START_DATE'] = df_train['START_DATE'].astype(str)
start_dates = df_train[['START_DATE', 'YEAR_WEEK']].copy()
start_dates = start_dates.drop_duplicates(subset=['YEAR_WEEK'], keep='first').reset_index()
del start_dates['index']

df = pd.DataFrame()
appended_data = []
number_of_cells = len(cell_names)

for (i,cell_name) in enumerate(cell_names):
    
    df = df_train[df_train["CELL_NAME"] == cell_name]
    df2 = df.groupby(['CELL_NAME','YEAR_WEEK']).mean().pct_change().reset_index()
    df2['KEY'] = df2['CELL_NAME'] + df2['YEAR_WEEK']
    
    df3 = df.groupby(['CELL_NAME','YEAR_WEEK']).mean().reset_index()
    df3['KEY'] = df3['CELL_NAME'] + df3['YEAR_WEEK']
    df3 = df3[['DL_USER_THROUGHPUT_MBPS', 'KEY']].copy()
    df4 = pd.merge(df2, df3, on='KEY')
    df2 = df4.rename({"DL_USER_THROUGHPUT_MBPS_x": "DL_USER_THROUGHPUT_MBPS_PCT_CHANGE",
                 "DL_USER_THROUGHPUT_MBPS_y": "DL_USER_THROUGHPUT_MBPS_AVERAGE"
                 }, axis='columns')
    df2 = df2[['CELL_NAME','YEAR_WEEK', 'DL_USER_THROUGHPUT_MBPS_PCT_CHANGE',
              'DL_USER_THROUGHPUT_MBPS_AVERAGE']]
    df2 = df2.fillna(0)
    df2['DEGRADED'] = df2['DL_USER_THROUGHPUT_MBPS_PCT_CHANGE'].apply(lambda x: 1 if x <= -0.05 else 0)
    df2 = findDegradation(df2, 3)
    appended_data.append(df2)
    print(f'[INFO] {i+1} of {number_of_cells} completed.')
    
    #if i == 100:
    #    break
          
appended_data = pd.concat(appended_data, axis=0)

name = KPI + "_PCT_CHANGE"
appended_data = appended_data.rename({KPI: name,}, axis='columns')

result = pd.merge(appended_data, start_dates, on='YEAR_WEEK')
result = result.sort_values(['CELL_NAME','YEAR_WEEK'])
result = result[['CELL_NAME', 'YEAR_WEEK','START_DATE','DL_USER_THROUGHPUT_MBPS_AVERAGE',
               'DL_USER_THROUGHPUT_MBPS_PCT_CHANGE','DEGRADED','FLAG']]

# Adding Flag Sequences
result = getConsecutiveSequencesWeekly(result)
result = result.fillna(0)

# Saving and Uploading to DWH
#path = "./Reports/DEGRADATION/"
#KPIForecaster.makeDir(path)
#date = datetime.today().strftime('%Y_%m_%d')

#file_name = path + "WEEKLY_DEGRADATION_REPORT_" + KPI + "_" + str(date) + ".csv" 

#result.to_csv(file_name)

print("[INFO] Uploading Report to DWH.")
result['DL_USER_THROUGHPUT_MBPS_PCT_CHANGE'].replace(np.inf, 0, inplace=True)
sql.dumpToDWH(result, "KPI_DEGRADATION_WEEKLY", if_exists = 'append')


summary = getSummaryReport(result)
sql.deleteTable("KPI_DEGRADATION_WEEKLY_SUMMARY")
sql.dumpToDWH(summary, "KPI_DEGRADATION_WEEKLY_SUMMARY")