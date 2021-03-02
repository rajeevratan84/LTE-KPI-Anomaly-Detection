from sqlalchemy import create_engine
from sqlalchemy import types
from datetime import datetime, timedelta
import sqlalchemy 
import pyodbc
import cx_Oracle
import pandas as pd
import urllib
import sys
import numpy as np
import time

class SQLDatabase:
    def __init__(self, conf):
        self.conf = conf
        self.engine = self._connect_to_sql()
        
    def _connect_to_sql(self):
        # This is the path to the ORACLE client files
        lib_dir = r"/Users/rajeevratan/Downloads/instantclient_19_8"
            
        try:
            cx_Oracle.init_oracle_client(lib_dir=lib_dir)
        except Exception as err:
            print("Error connecting: cx_Oracle.init_oracle_client()")
            
        DIALECT = 'oracle'
        SQL_DRIVER = 'cx_oracle'
        USERNAME = '' #enter your username
        PASSWORD = '' #enter your password
        HOST = 'IP' #enter the oracle db host url
        PORT = 1522 # enter the oracle port number
        SERVICE = '' # enter the oracle db service name
        ENGINE_PATH_WIN_AUTH = DIALECT + '+' + SQL_DRIVER + '://' + USERNAME + ':' + PASSWORD +'@' + HOST + ':' + str(PORT) + '/?service_name=' + SERVICE
        engine = create_engine(ENGINE_PATH_WIN_AUTH, max_identifier_length=128)
        return engine
    
    def getYesterdaysReport(self):
         
        yesterday = datetime.now() - timedelta(1)
        start = datetime.strftime(yesterday, '%Y-%m-%d') + ' 00:00:00(UTC-04:00)'
        end = datetime.strftime(yesterday, '%Y-%m-%d') + ' 23:00:00(UTC-04:00)'
        
        print(f"[INFO] Pulling report from {start} to {end}...")
        
        sql_query_get_yesterday_auto = f"SELECT START_TIME, CELL_NAME,\
        DL_USER_THROUGHPUT_MBPS FROM TECHNICAL_KPI.FT_CELL_HOURLY_LTE WHERE\
        START_TIME BETWEEN '{start}' AND '{end}' ORDER BY START_TIME"
        
        df_yesterday = pd.read_sql(sql_query_get_yesterday_auto, self.engine) 
        return df_yesterday
    
    def getDailyKPIData(self):
         
        yesterday = datetime.now() - timedelta(1)
        start = datetime.strftime(yesterday, '%Y-%m-%d') + ' 00:00:00(UTC-04:00)'
        end = datetime.strftime(yesterday, '%Y-%m-%d') + ' 23:00:00(UTC-04:00)'
        
        print(f"[INFO] Pulling KPI Daily report from {start} to {end}...")  

        sql_query_input_data = f"SELECT START_TIME, CELL_NAME, DL_USER_THROUGHPUT_MBPS,\
        MAX_PHYS_DL_THROUGHPUT_MBPS, DL_64QAM_MODULATION_UTIL, GOOD_CQI_10_15, DL_PRB_UTILIZATION,\
        BAD_CQI_0_6, UL_PRB_UTILIZATION, TOTAL_DATA_VOLUME_GB, DL_DATA_VOLUME_GB, LATENCY_MS,\
        INTERFERENCE_DBM FROM TECHNICAL_KPI.FT_CELL_HOURLY_LTE WHERE\
        START_TIME BETWEEN '{start}' AND '{end}' ORDER BY START_TIME"
        
        df_input = pd.read_sql(sql_query_input_data, self.engine) 
        df_input.columns = map(str.upper, df_input.columns)
        return df_input
   
    def getHourlyKPIReportXDays(self, days = 30):
         
        yesterday = datetime.now() - timedelta(1)
        X_days_ago = datetime.now() - timedelta(days)
        #start = datetime.strftime(X_days_ago, '%Y-%m-%d') + ' 00:00:00(UTC-04:00)'
        start = datetime.strftime(X_days_ago, '2020-09-01') + ' 00:00:00(UTC-04:00)'
        end = datetime.strftime(yesterday, '%Y-%m-%d') + ' 23:00:00(UTC-04:00)'
        
        print(f"[INFO] Pulling report from {start} to {end}...")
        
        sql_query_get_report = f"SELECT START_TIME, CELL_NAME,\
        DL_USER_THROUGHPUT_MBPS FROM TECHNICAL_KPI.FT_CELL_HOURLY_LTE WHERE\
        START_TIME BETWEEN '{start}' AND '{end}' ORDER BY START_TIME"
        
        df = pd.read_sql(sql_query_get_report, self.engine) 
        df.columns = map(str.upper, df.columns)
        df = df.rename({'start_time':'START_TIME',
                      'cell_name':'CELL_NAME',
                      'dl_user_throughput_mbps':'DL_USER_THROUGHPUT_MBPS'
                     }, axis='columns')
        
        return df    
    
    def getHourlyKPIReport(self):
         
        start = self.conf["training_data_start_date"]  + ' 00:00:00(UTC-04:00)'
        end = self.conf["training_data_end_date"]  + ' 23:00:00(UTC-04:00)'
        
        print(f"[INFO] Pulling report from {start} to {end}...")
        
        sql_query_get_report = f"SELECT START_TIME, CELL_NAME,\
        DL_USER_THROUGHPUT_MBPS FROM TECHNICAL_KPI.FT_CELL_HOURLY_LTE WHERE\
        START_TIME BETWEEN '{start}' AND '{end}' ORDER BY START_TIME"
        
        df = pd.read_sql(sql_query_get_report, self.engine) 
        df.columns = map(str.upper, df.columns)
        df = df.rename({'start_time':'START_TIME',
                      'cell_name':'CELL_NAME',
                      'dl_user_throughput_mbps':'DL_USER_THROUGHPUT_MBPS'
                     }, axis='columns')
        
        return df    
    
    def getHourlyKPIReportDegradation(self):
         
        start = self.conf["degradation_data_start_date"]  + ' 00:00:00(UTC-04:00)'
        end = self.conf["degradation_data_end_date"]  + ' 23:00:00(UTC-04:00)'
        
        print(f"[INFO] Pulling report from {start} to {end}...")
        
        sql_query_get_report = f"SELECT START_TIME, CELL_NAME,\
        DL_USER_THROUGHPUT_MBPS FROM TECHNICAL_KPI.FT_CELL_HOURLY_LTE WHERE\
        START_TIME BETWEEN '{start}' AND '{end}' ORDER BY START_TIME"
        
        df = pd.read_sql(sql_query_get_report, self.engine) 
        df.columns = [x.upper() for x in df.columns]
        return df    
 
 
    def getDegradationReport(self):
                
        sql_query_get_report = f"SELECT START_TIME, CELL_NAME,\
        DL_USER_THROUGHPUT_MBPS FROM TECHNICAL_KPI.FT_CELL_HOURLY_LTE WHERE\
        START_TIME BETWEEN '{start}' AND '{end}' ORDER BY START_TIME"
        
        df = pd.read_sql(sql_query_get_report, self.engine) 
        df.columns = [x.upper() for x in df.columns]
        return df 
    
    def dumpToDWH(self, df, table, if_exists = 'append'):
        try:
            dtyp = {c:types.VARCHAR(df[c].str.len().max()) 
                    for c in df.columns[df.dtypes == 'object'].tolist()}
        except:
            raise Exception("ERROR: Possibly column not string data")
        try:
            T_START = time.time()
            df.to_sql(table, self.engine, index=False, if_exists = if_exists, dtype=dtyp)
            t0 =  time.time()
            completion_time = t0-T_START
            print(f'[INFO] {table} report successfully uploaded to DHW in {completion_time} seconds.')
        except:
            raise Exception("ERROR Uploading Data to DWH")
        
    def deleteTable(self, table):
        try:
            with self.engine.connect() as con:
                rs = con.execute(f'DELETE FROM AI_ADMIN.{table}')
        except:
            raise Exception("Error Deleting Table")
            
    def dumpReportByCellDegrade(self, df, table):   
        print("[INFO] Uploading Report to DHW...") 
        cell_names = df.CELL_NAME.unique()   
        num_cells = len(cell_names)
        for (i,cell_name) in enumerate(cell_names):
            print(f'{i} of {num_cells} uploaded')
            cell_df = df[df["CELL_NAME"] == cell_name].copy()
            cell_df['DL_USER_THROUGHPUT_MBPS_PCT_CHANGE'].replace(np.inf, 0, inplace=True)
            self.dumpReport(cell_df, table)
            
    def dumpReportByCellForecast(self, df, table):   
        print("[INFO] Uploading Report to DHW...") 
        cell_names = df.CELL_NAME.unique()   
        num_cells = len(cell_names)
        for (i,cell_name) in enumerate(cell_names):
            print(f'{i} of {num_cells} uploaded')
            cell_df = df[df["CELL_NAME"] == cell_name].copy()
            self.dumpReport(cell_df, table)