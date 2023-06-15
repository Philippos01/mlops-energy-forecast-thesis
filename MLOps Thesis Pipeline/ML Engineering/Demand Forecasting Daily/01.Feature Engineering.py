# Databricks notebook source
# MAGIC %run "/Repos/filippos.priovolos01@gmail.com/mlops-energy-forecast-thesis/MLOps Thesis Pipeline/Workflow Config/Daily Inference"

# COMMAND ----------

# MAGIC %md
# MAGIC * Import inference data and convert them to timestamp

# COMMAND ----------

from datetime import datetime
from pyspark.sql.functions import hour, when

# Convert string dates to timestamps
df = spark.sql(f"""
SELECT
    TO_TIMESTAMP('{yesterdate}', 'yyyy-MM-dd') AS yesterdate_ts,
    TO_TIMESTAMP('{date}', 'yyyy-MM-dd') AS date_ts
""")



# COMMAND ----------

# MAGIC %md
# MAGIC * Define the date to be predicted as date_to_predict, which is to be extracted from the index of the DataFrame df
# MAGIC * Read data from the table "final_consumption_countries_hourly" 
# MAGIC * Filter the df_cons DataFrame to retain only rows from the last 720 hours (30 days) prior to the date_to_predict
# MAGIC * Convert the resulting Spark DataFrame df_cons to a Pandas DataFrame

# COMMAND ----------

# Select the date portion of the timestamp and convert it to string format
date_df = df.select(date_format("date_ts", "yyyy-MM-dd").alias("date_string"))

# Extract the date string from the DataFrame
date_to_predict = date_df.first()["date_string"]
# Read data from the table
df_cons = spark.read.table("final_consumption_countries_hourly")

# Before filtering
print("Before filtering:")
print(df_cons.select("DATETIME").agg({"DATETIME": "min"}).collect()[0])
print(df_cons.select("DATETIME").agg({"DATETIME": "max"}).collect()[0])

# Filter the data to include only rows from the exact previous month
df_cons = df_cons.filter(
    (col("DATETIME") < to_date(lit(date_to_predict))) &
    (col("DATETIME") >= add_months(to_date(lit(date_to_predict)), -1))
)

# After filtering
print("After filtering:")
print(df_cons.select("DATETIME").agg({"DATETIME": "min"}).collect()[0])
print(df_cons.select("DATETIME").agg({"DATETIME": "max"}).collect()[0])

# Convert Spark DataFrame to pandas DataFrame
df_cons = df_cons.toPandas()

# Sort 'final_consumption' dataframe by 'DATETIME' and 'COUNTRY'
df_cons.sort_values(by=['DATETIME', 'COUNTRY'], inplace=True)

# Display DataFrame
df_cons

# COMMAND ----------

# MAGIC %md
# MAGIC * Print earliest and latest timestamp to validate dates

# COMMAND ----------

# Display min and max 'DATETIME'
print("Earliest timestamp:", df_cons['DATETIME'].min())
print("Latest timestamp:", df_cons['DATETIME'].max())

# COMMAND ----------

# MAGIC %md
# MAGIC * Create a Dataframe with 24x11 rows each, timestamp for each country will have value 1, others will have value 0(one-hot-encoding)

# COMMAND ----------

# Create an array with 24 hours
hours = list(range(24))

# Create a DataFrame with 24x11 rows, each timestamp for each country will have value 1, others will have value 0
data = []
for hour in hours:
    for country in countries:
        timestamp_str = f"{date} {str(hour).zfill(2)}:00:00"
        row = [timestamp_str] + [1 if c == country else 0 for c in countries]
        data.append(Row(*row))

# Define column names
columns = ['DATETIME'] + countries

# Create DataFrame
df = spark.createDataFrame(data, columns)

# Convert string timestamp to actual timestamp
df = df.withColumn("DATETIME", expr(f"to_timestamp(DATETIME, 'yyyy-MM-dd HH:mm:ss')"))

df = df.toPandas()
df


# COMMAND ----------

# MAGIC %md
# MAGIC The given code transforms df from wide to long format by melting it, filters the rows to include only those with a value of 1, and then drops the "VALUE" column. This operation helps reshape and manipulate the data, to deal with ategorical variables.

# COMMAND ----------

df_melted = df.melt(id_vars='DATETIME', var_name='COUNTRY', value_name='VALUE')
df_melted = df_melted[df_melted['VALUE'] == 1]
df_melted = df_melted.drop('VALUE', axis=1)
df_melted

# COMMAND ----------

df_combined = pd.concat([df_melted, df_cons], axis=0)

#  sort the resulting dataframe by 'DATETIME'
df_combined = df_combined.sort_values('DATETIME')
df_combined

# COMMAND ----------

# MAGIC %md
# MAGIC * The function create_lag_features takes in a DataFrame df with columns 'COUNTRY', 'DATETIME', and 'HOURLY_CONSUMPTION_MW'.
# MAGIC * It sorts df based on 'COUNTRY' and 'DATETIME'.
# MAGIC * Creates lag features for the previous day, week, and month's consumption.
# MAGIC * Forward fills NaN values in 'HOURLY_CONSUMPTION_MW' for each country.
# MAGIC * Calculates rolling statistics (mean, standard deviation, sum) for the past 24 hours and 7 days.
# MAGIC * Backward fills NaN values in the lag features for each country.
# MAGIC * Returns the modified DataFrame with new lag and rolling features.

# COMMAND ----------

def create_lag_features(df):
    """
    Creates lag features from datetime index
    """
    df = df.sort_values(['COUNTRY', 'DATETIME']).reset_index(drop=True)  # Sort by 'COUNTRY' and 'DATETIME' and reset index
    # Group by country and shift to create lagged features
    df['PREV_DAY_CONSUMPTION'] = df.groupby('COUNTRY')['HOURLY_CONSUMPTION_MW'].shift(24)
    df['PREV_WEEK_CONSUMPTION'] = df.groupby('COUNTRY')['HOURLY_CONSUMPTION_MW'].shift(24 * 7)
    df['PREVIOUS_MONTH_CONSUMPTION'] = df.groupby('COUNTRY')['HOURLY_CONSUMPTION_MW'].shift(24 * 30)

    # Forward fill to handle NaN values in HOURLY_CONSUMPTION_MW for rolling window calculations
    df['HOURLY_CONSUMPTION_MW'] = df.groupby('COUNTRY')['HOURLY_CONSUMPTION_MW'].fillna(method='ffill')

    # Calculate rolling statistics for each country
    df['ROLLING_MEAN_24H'] = df.groupby('COUNTRY')['HOURLY_CONSUMPTION_MW'].rolling(window=24,min_periods=1).mean().reset_index(0,drop=True)
    df['ROLLING_STD_24H'] = df.groupby('COUNTRY')['HOURLY_CONSUMPTION_MW'].rolling(window=24,min_periods=1).std().reset_index(0,drop=True)
    df['ROLLING_SUM_7D'] = df.groupby('COUNTRY')['HOURLY_CONSUMPTION_MW'].rolling(window=7 * 24, min_periods=1).sum().reset_index(0,drop=True)

    # Backward fill only the rows that end up as null after shifting for each country
    df['PREV_DAY_CONSUMPTION'] = df.groupby('COUNTRY')['PREV_DAY_CONSUMPTION'].fillna(method='bfill')
    df['PREV_WEEK_CONSUMPTION'] = df.groupby('COUNTRY')['PREV_WEEK_CONSUMPTION'].fillna(method='bfill')
    df['PREVIOUS_MONTH_CONSUMPTION'] = df.groupby('COUNTRY')['PREVIOUS_MONTH_CONSUMPTION'].fillna(method='bfill')

    return df

df_combined = create_lag_features(df_combined)


# COMMAND ----------

df_combined

# COMMAND ----------

# MAGIC %md
# MAGIC The given code converts a variable to datetime format, extracts rows from a DataFrame based on a specific date, and drops a specified column from the resulting DataFrame. This allows for working with a subset of data for a specific date and removing unnecessary columns for further analysis or processing.

# COMMAND ----------

# Convert your predicting_date to datetime format
date_to_predict = pd.to_datetime(date_to_predict)

# Extract the date from the 'DATETIME' column, compare it to predicting_date
df_final = df_combined[df_combined['DATETIME'].dt.date == date_to_predict.date()]
df_final.drop(columns=['HOURLY_CONSUMPTION_MW'],inplace=True)

# COMMAND ----------

df_final

# COMMAND ----------

# MAGIC %md
# MAGIC * The function create_time_features takes a DataFrame df with 'DATETIME' as one of its columns.
# MAGIC * Sets 'DATETIME' as the index of the DataFrame.
# MAGIC * Extracts and creates new features such as 'HOUR', 'DAY_OF_WEEK', 'MONTH', 'QUARTER', 'YEAR', 'DAY_OF_YEAR', 'DAY_OF_MONTH', and 'WEEK_OF_YEAR' from the 'DATETIME' index.
# MAGIC * Sorts the DataFrame based on the 'DATETIME' index.
# MAGIC * Returns the modified DataFrame with the new time-related features.

# COMMAND ----------

def create_time_features(df):
    """
    Creates time series features from datetime index
    """
    # Ensure 'DATETIME' is the index
    df.set_index('DATETIME', inplace=True)

    # Create date-related features
    df['HOUR'] = df.index.hour  
    df['DAY_OF_WEEK'] = df.index.dayofweek
    df['MONTH'] = df.index.month
    df['QUARTER'] = df.index.quarter
    df['YEAR'] = df.index.year
    df['DAY_OF_YEAR'] = df.index.dayofyear
    df['DAY_OF_MONTH'] = df.index.day
    df['WEEK_OF_YEAR'] = df.index.isocalendar().week

    # Sort the DataFrame by the datetime index
    df.sort_index(inplace=True)
    
    return df

df_final = create_time_features(df_final)
df_final

# COMMAND ----------

df_final

# COMMAND ----------

# MAGIC %md
# MAGIC * The function one_hot_encode takes a DataFrame df with 'COUNTRY' as one of its columns.
# MAGIC * Defines a list of country names to be one-hot encoded.
# MAGIC * Iterates through each country in the list:
# MAGIC * For each country, it creates a new column in the DataFrame, named after the country.
# MAGIC * Each entry in the new column is set to 1 if the 'COUNTRY' column matches the country name, otherwise it's set to 0.
# MAGIC * Returns the modified DataFrame with new one-hot encoded columns for countries.

# COMMAND ----------

def one_hot_encode(df):
    countries = ['belgium','denmark','france','germany','greece','italy','luxembourg','netherlands','spain','sweden','switzerland']
    countries.sort()
    for country in countries:
        df[country] = df.apply(lambda row: 1 if row['COUNTRY'] == country else 0, axis=1)
    return df

df_final = one_hot_encode(df_final)
df_final = df_final.reset_index()
df_final

# COMMAND ----------

df_final.columns

# COMMAND ----------

df_final[df_final['COUNTRY']=="greece"]

# COMMAND ----------

# MAGIC %md
# MAGIC * The code converts a Pandas DataFrame (df_final) to a Spark DataFrame (spark_df).
# MAGIC * The Spark DataFrame is saved as a table in Databricks using the name specified in table_name.
# MAGIC * A Delta table is created based on the Spark DataFrame.
# MAGIC * The purpose is to store the data in a table format in Databricks, facilitating further analysis and querying using Spark SQL.

# COMMAND ----------

spark_df = spark.createDataFrame(df_final)
# Save the Spark DataFrame as a table in Databricks
table_name = 'inferenece_features'
spark_df.createOrReplaceTempView(table_name)
spark.sql(f"CREATE TABLE IF NOT EXISTS {table_name} USING delta AS SELECT * FROM {table_name}")

# COMMAND ----------

df_final.columns

# COMMAND ----------

# MAGIC %md
# MAGIC * Create a Spark DataFrame: The Pandas DataFrame df_final is converted to a Spark DataFrame using the spark.createDataFrame() function. This conversion allows for working with the DataFrame using Spark's distributed computing capabilities.
# MAGIC * Create a temporary view: The Spark DataFrame df_final_spark is assigned as a temporary view named 'df_final' using the createOrReplaceTempView() function. This creates a temporary view of the DataFrame within the Spark session, enabling the execution of SQL queries and operations on the DataFrame.

# COMMAND ----------

df_final_spark=spark.createDataFrame(df_final) 
df_final_spark.createOrReplaceTempView('df_final')

# COMMAND ----------

# MAGIC %md
# MAGIC * SQL Query: It executes an SQL query to select all columns from the 'df_final' table/view.
# MAGIC * Add CONSUMPTION_ID column: Using the withColumn() function, a new column named 'CONSUMPTION_ID'(Primary Key) is added to the DataFrame. The values in this column are created by concatenating 'COUNTRY' and 'DATETIME' columns with an underscore ('_') separator.
# MAGIC * Convert DATETIME column: Using the withColumn() function again, the 'DATETIME' column is converted to a timestamp data type by casting it with the CAST() function.
# MAGIC * Drop COUNTRY column: The 'COUNTRY' column is dropped from the DataFrame using the drop() function.
# MAGIC * Create temporary view: Finally, the modified DataFrame is used to create a new temporary view named 'daily_features'.

# COMMAND ----------

import pyspark.sql.functions as f

spark.sql('select * from df_final') \
    .withColumn('CONSUMPTION_ID', f.expr('concat_ws("_", COUNTRY, DATETIME)')) \
    .withColumn('DATETIME', f.expr('CAST(DATETIME AS timestamp)')) \
    .drop('COUNTRY').createOrReplaceTempView('daily_features')

# COMMAND ----------

# MAGIC %md
# MAGIC * The variable columns is assigned the list of column names from a table named daily_features retrieved through a Spark SQL query.
# MAGIC * feature_columns is created by taking all column names in columns except for 'DATETIME' and 'CONSUMPTION_ID'.
# MAGIC * update_columns is a string created by joining elements from feature_columns with ' = B.' prefixed to each element and separated by commas. * This could be used in an SQL UPDATE statement.
# MAGIC * insert_columns is a string created by joining 'B.' prefixed elements from feature_columns, separated by commas. This could be used in an SQL INSERT statement.

# COMMAND ----------

columns = spark.sql('select * from daily_features').columns
feature_columns = [column for column in columns if column not in ('DATETIME', 'CONSUMPTION_ID')]
update_columns = ', '.join([f'{column} = B.{column}' for column in feature_columns])
insert_columns = ', '.join([f'B.{column}' for column in feature_columns])

# COMMAND ----------

# MAGIC %md
# MAGIC * The query is merging data from a table named daily_features (aliased as B) into another table called hourly_forecasting_features (aliased as A).
# MAGIC * The merge is based on the condition that the 'DATETIME' and 'CONSUMPTION_ID' columns in both tables must be equal (A.DATETIME = B.DATETIME AND A.CONSUMPTION_ID = B.CONSUMPTION_ID).
# MAGIC * If there is a match between the records in tables A and B (based on 'DATETIME' and 'CONSUMPTION_ID'), then the corresponding records in table A are updated with the values from table B. The columns to be updated are defined by the string update_columns, which was created earlier to have the form column1 = B.column1, column2 = B.column2, ....
# MAGIC * If there is no match between the records in table A and B, then a new record is inserted into table A with values from table B. The columns that will be inserted are 'DATETIME', 'CONSUMPTION_ID', and the additional feature columns. The columns to be inserted are defined in the format (column1, column2, ...) and the values to be inserted are in the format (B.column1, B.column2, ...).

# COMMAND ----------

spark.sql(f"""
MERGE INTO hourly_forecasting_features A
USING daily_features B
ON A.DATETIME = B.DATETIME AND A.CONSUMPTION_ID = B.CONSUMPTION_ID
WHEN MATCHED THEN
  UPDATE SET
    {update_columns}
WHEN NOT MATCHED
  THEN INSERT (
    DATETIME,
    CONSUMPTION_ID,
    {', '.join(feature_columns)}
) VALUES (  
    B.DATETIME,
    B.CONSUMPTION_ID,
   {insert_columns}
)
""")

# COMMAND ----------


