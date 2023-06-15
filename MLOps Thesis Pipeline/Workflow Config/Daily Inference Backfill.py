# Databricks notebook source
# data = [(database, date, yesterdate)]
# config_columns = ["database","execution_date","execution_yesterdate"]
# spark.createDataFrame(data=data, schema=config_columns).write.saveAsTable(f'{database}.config_inference_daily')

# COMMAND ----------

from datetime import datetime, timedelta
import requests
import json
import time


auth = {"Authorization": "Bearer dapi77cd93bf66efb1da5641abc2283ec2ee-2"}

def get_next_date(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (date + timedelta(days=1)).strftime('%Y-%m-%d')

# COMMAND ----------

# DBTITLE 1,Daily Inference
job_id = {"job_id": "1081513940150965"}

for i in range (1):
    config = spark.sql('select * from df_dev.config_inference_daily').collect()
    database, date, yesterdate = config[0]

    date = get_next_date(date)
    yesterdate = get_next_date(yesterdate)

    # spark.sql(f"""
    # UPDATE {database}.config_inference_daily AS A
    # SET execution_yesterdate = "{yesterdate}",
    #     execution_date = "{date}"
    # WHERE database = "{database}"
    # """)
    response = requests.post('https://adb-4287679896047209.9.azuredatabricks.net/api/2.0/jobs/run-now', json = job_id, headers=auth).json()
    print(response)
    while requests.get('https://adb-4287679896047209.9.azuredatabricks.net/api/2.0/jobs/runs/list', json = job_id, headers=auth) \
            .json()['runs'][0]['state']['life_cycle_state'] != 'TERMINATED':
        time.sleep(20)

# COMMAND ----------

display(spark.sql('select * from df_dev.config_inference_daily'))

# COMMAND ----------

job_id = {"job_id": "630864861974070"}
for i in range (1):
    # config = spark.sql('select * from df_dev.config_inference_daily').collect()
    # database, date, yesterdate = config[0]

    # date = get_next_date(date)
    # yesterdate = get_next_date(yesterdate)

    # spark.sql(f"""
    # UPDATE {database}.config_inference_daily AS A
    # SET execution_yesterdate = "{yesterdate}",
    #     execution_date = "{date}"
    # WHERE database = "{database}"
    # """)
    response = requests.post('https://adb-4287679896047209.9.azuredatabricks.net/api/2.0/jobs/run-now', json = job_id, headers=auth).json()
    print(response)
    while requests.get('https://adb-4287679896047209.9.azuredatabricks.net/api/2.0/jobs/runs/list', json = job_id, headers=auth) \
            .json()['runs'][0]['state']['life_cycle_state'] != 'TERMINATED':
        time.sleep(20)

# COMMAND ----------

train_start = '2013-06-01' #the retrain start date
train_end = '2018-06-20'   #the retrain end date (20/06/2018 - 30/06/2018) 10 days for testing
test_end = '2018-06-30'
spark.sql(f'''CREATE TABLE df_dev.config_retrain_monthly AS 
(SELECT  "{train_start}" as train_start,
"{train_end}" as train_end,
"{test_end}" as test_end
)''')
