#!/usr/bin/env python
from KPIForecaster.forecaster import KPIForecaster
from configuration.settings import Conf
from database.sql_connect import SQLDatabase
import pandas as pd
import time
import sys
import os.path

path = sys.argv[0].rsplit("/", 1)[0]
#import warnings
#warnings.filterwarnings("ignore")

# Get our configuations
print(path)
conf = Conf(os.path.join(path,"config.json"))
#conf = os.path.join(path,"config.json")

# Create our database object
sql = SQLDatabase(conf)

# Creating out KPI Forecaster Object
KPIF = KPIForecaster(conf, crontab=True)

# Loading our training data
#df_train = pd.read_csv('FT_CELL_NOV.csv')
df_train = sql.getHourlyKPIReportXDays(30)
print(df_train.head())
cell_names = df_train.CELL_NAME.unique()

df_train = KPIF.filterDates(df_train)
print(df_train.head())
print("[INFO] Training " + str(len(cell_names)) + " Models")

# Starting Timer for benchmarking
T_START = time.time()

# Train Baseline Model
KPIF.baseLineNetworkModel(df_train)

for i,cell_name in enumerate(cell_names):
    df_model = KPIF.getTrainingData(df_train, cell = cell_name)
    prophet, forecast, m = KPIF.getForecast(df_model)
    KPIF.saveModel(prophet, m, forecast, cell = cell_name, KPI = "DL_USER_THROUGHPUT_MBPS")
    print(str(i+1) + " of " + str(len(cell_names)) + " Models Created.")
        
t0 =  time.time()
completion_time = t0-T_START
print("******* Total Time to Create all Models: " + str(completion_time)) 
print("******* Average Time Per Model " + str(completion_time/(i+1)))
print("[INFO] Completed Training Models.")