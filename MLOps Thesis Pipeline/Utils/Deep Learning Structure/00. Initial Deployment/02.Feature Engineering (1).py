# Databricks notebook source
# MAGIC %pip install databricks && pip install databricks-feature-store

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

db = 'df_landing'
scenarios = ['hour', 'day', 'month']
current_scenario = scenarios[0]
input_table_name = 'greece_total_consuption_hourly'
date_column = 'Time'
target_column = 'Load_Actual'
output_table_name = 'DL_forecasting_features_' + current_scenario
save_into_feature_store = True
version = '1_1'
delete_fs = False

if current_scenario == 'hour':
    granularity = 24
    maximum_past_feature = 28 * granularity
    horizon = 24
elif current_scenario == 'day':
    granularity = 1
    maximum_past_feature = 7
    horizon = 1
elif current_scenario == 'month':
    granularity = 1
    maximum_past_feature = 3
    horizon = 1


# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

from pyspark.sql.functions import col, sum
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
import pandas as pd
from databricks import feature_store

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Dataset

# COMMAND ----------

table = spark.table(f'{db}.{input_table_name}')
table.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Features for Model Training

# COMMAND ----------

def create_date_features(date, scenario = 'hour'):
    pd_dt = pd.to_datetime(date)
    if scenario in ['hour', 'day']:
        dayofweek = pd_dt.dayofweek
        month = pd_dt.month
        quarter = pd_dt.quarter
        year = pd_dt.year
        dayofyear = pd_dt.dayofyear
        day = pd_dt.day
        weekofyear = pd_dt.weekofyear
        return [pd_dt, dayofweek, month, year, quarter, dayofyear, day, weekofyear]
    elif scenario == 'month':
        month = pd_dt.month
        quarter = pd_dt.quarter
        return [pd_dt, month, quarter]

def create_historical_load_features(past_load, granularity, scenario = 'hour'):
    if scenario == 'hour':
        previous_day_load = past_load[-granularity:]
        previous_week_load = past_load[-(7 * granularity) : -(7 * granularity) + granularity]
        previous_month_load = past_load[-(28 * granularity) : -(28 * granularity) + granularity]
        return previous_day_load + previous_week_load + previous_month_load
    elif scenario == 'day':
        previous_days_load = list(reversed(past_load[-granularity:]))
        return previous_days_load
    elif scenario == 'month':
        previous_months_load = list(reversed(past_load[-granularity:]))
        return previous_months_load

def create_names(list_of_names, granularity):
    output_names = []
    for name in list_of_names:
        for i in range(0, granularity):
            output_names.append(name + '_' + str(i))
    return output_names

def create_dictionary_of_lists(column_names):
    dictionary = {}
    for column in column_names:
        dictionary[column] = []
    return dictionary

def add_multiple_values_to_multiple_keys(list_of_values, key_name, dictionary):
    for key, value in zip(key_name,list_of_values):
        dictionary[key] += [value]
    return dictionary
 
def create_feature_dataset_hour( df, date_column, target_column, maximum_past_feature, granularity, horizon):
    dates = df[date_column]
    targets = df[target_column].values.tolist()
    maximum = max(targets)
    targets = [value / maximum for value in targets]
    start_id = maximum_past_feature
    final_id = len(dates) - horizon + 1
    historical_data_columns = ['previous_day_load', 'previous_week_load', 'previous_month_load']
    historical_data_columns = create_names(historical_data_columns, granularity)
    date_columns = ['date', 'day_of_week', 'month', 'year', 'quarter', 'day_of_year', 'day_of_month', 'week_of_year' ]
    target_column_name = ['y']
    target_column_names = create_names(target_column_name, horizon)
    column_names = date_columns + historical_data_columns + target_column_names
    final_dictionary = create_dictionary_of_lists(column_names)
    for i in range(start_id, final_id, granularity):
        labels = targets[i:i + horizon]
        date_features = create_date_features(dates[i])
        historical_load_features = create_historical_load_features(targets[:i], granularity)
        final_dictionary = add_multiple_values_to_multiple_keys(date_features + historical_load_features + labels, column_names, final_dictionary)
    dataframe = pd.DataFrame(final_dictionary)
    return dataframe, maximum

def create_feature_dataset_day( df, date_column, target_column, maximum_past_feature, horizon):
    #df[date_column] = df[date_column].apply(lambda x: x.strftime('%Y-%m-%d'))
    df[date_column] = df[date_column].apply(lambda x: x[:10])
    dates = df[date_column].unique().tolist()
    targets = (df.groupby(date_column).sum())[target_column].values.tolist()
    maximum = max(targets)
    targets = [value / maximum for value in targets]
    start_id = maximum_past_feature
    final_id = len(dates) - horizon + 1
    historical_data_columns = ['previous_day_load']
    historical_data_columns = create_names(historical_data_columns, maximum_past_feature)
    date_columns = ['date', 'day_of_week', 'month', 'year', 'quarter', 'day_of_year', 'day_of_month', 'week_of_year' ]
    target_column_name = ['y']
    target_column_names = create_names(target_column_name, horizon)
    column_names = date_columns + historical_data_columns + target_column_names
    final_dictionary = create_dictionary_of_lists(column_names)
    for i in range(start_id, final_id):
        labels = targets[i:i + horizon]
        date_features = create_date_features(dates[i])
        historical_load_features = create_historical_load_features(targets[:i], maximum_past_feature, 'day')
        final_dictionary = add_multiple_values_to_multiple_keys(date_features + historical_load_features + labels, column_names, final_dictionary)
    dataframe = pd.DataFrame(final_dictionary)
    return dataframe, maximum

def create_feature_dataset_month( df, date_column, target_column, maximum_past_feature, horizon):
    #df[date_column] = df[date_column].apply(lambda x: x.strftime('%Y-%m-%d'))
    df[date_column] = df[date_column].apply(lambda x: x[:7])
    dates = df[date_column].unique().tolist()
    targets = (df.groupby(date_column).sum())[target_column].values.tolist()
    maximum = max(targets)
    targets = [value / maximum for value in targets]
    start_id = maximum_past_feature
    final_id = len(dates) - horizon + 1
    historical_data_columns = ['previous_month_load']
    historical_data_columns = create_names(historical_data_columns, maximum_past_feature)
    date_columns = ['date', 'month', 'quarter' ]
    target_column_name = ['y']
    target_column_names = create_names(target_column_name, horizon)
    column_names = date_columns + historical_data_columns + target_column_names
    final_dictionary = create_dictionary_of_lists(column_names)
    for i in range(start_id, final_id):
        labels = targets[i:i + horizon]
        date_features = create_date_features(dates[i], 'month')
        historical_load_features = create_historical_load_features(targets[:i], maximum_past_feature, 'month')
        final_dictionary = add_multiple_values_to_multiple_keys(date_features + historical_load_features + labels, column_names, final_dictionary)
    dataframe = pd.DataFrame(final_dictionary)
    return dataframe, maximum

def prepare_data(scenario, df, date_column, target_column, granularity, maximum_past_feature, horizon):
    if scenario == 'hour':
        output_df, maximum = create_feature_dataset_hour( df, date_column, target_column, maximum_past_feature, granularity, horizon)
    elif scenario == 'day':
        output_df, maximum = create_feature_dataset_day( df, date_column, target_column, maximum_past_feature, horizon)
    elif scenario == 'month':
        output_df, maximum = create_feature_dataset_month( df, date_column, target_column, maximum_past_feature, horizon)
    return output_df, maximum

# COMMAND ----------

final_df, maximum = prepare_data(current_scenario, table.toPandas(), date_column, target_column, granularity, maximum_past_feature, horizon)

# COMMAND ----------

final_df['key'] = ['key'] * len(final_df)
final_df['ID'] = final_df.key + '_' + final_df.date.astype(str)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save features dataset into Feature Store

# COMMAND ----------

sql("SET spark.databricks.delta.preview.enabled=true")
sql("SET spark.databricks.delta.optimize.zorder.checkStatsCollection.enabled = false")
if save_into_feature_store:

    features = spark.createDataFrame(final_df)

    fs = feature_store.FeatureStoreClient()

    fs.create_table(
        name=f'{db}.{output_table_name}',
        primary_keys=['ID'],
        timestamp_keys='date',
        df=features
    )

# COMMAND ----------

if delete_fs:
    from databricks.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    fs.drop_table(name=f'{db}.{output_table_name}')
    print("Feature Store was succesfuly deleted")
