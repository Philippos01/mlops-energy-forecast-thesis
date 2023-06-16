# Databricks notebook source
#%run "/Repos/CI ADO Repo/01.Develop/Workflow Config/Daily Inference"
from pyspark.sql import functions as F
from pyspark.sql.functions import concat, col, lit, lpad

# COMMAND ----------

database= 'df_dev'

# COMMAND ----------

spark.sql(f'USE {database}')

# COMMAND ----------

# MAGIC %md
# MAGIC * Loads the data from the consumption_countries_hourly table in the df_dev database.

# COMMAND ----------

df = spark.read.table('df_dev.consumption_countries_hourly')

# COMMAND ----------

 display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Extracts the date and hour from the DATETIME column.
# MAGIC 1. Groups the data by country, date, and hour, and sums up the hourly consumption.
# MAGIC 1. Renames the summed column to HOURLY_CONSUMPTION_MW.
# MAGIC 1. Constructs a new DATETIME column by concatenating the date and hour.
# MAGIC 1. Converts the DATETIME column to timestamp format.
# MAGIC 1. Selects and reorders the columns to match the desired schema.

# COMMAND ----------

# Extract the date and hour from the start_time
df = df.withColumn('date', F.to_date(df['DATETIME']))
df = df.withColumn('hour', F.hour(df['DATETIME']))

# Group by country, date and hour, and sum up the hourly consumption
df_hourly = df.groupBy('COUNTRY', 'date', 'hour').sum('HOURLY_CONSUMPTION_MW')

# Rename the sum column
df_hourly = df_hourly.withColumnRenamed('sum(HOURLY_CONSUMPTION_MW)', 'HOURLY_CONSUMPTION_MW')

# Make sure the hour is a two-digit string
df_hourly = df_hourly.withColumn('hour', lpad(col('hour'), 2, '0'))

# Construct a new 'DATETIME' column
df_hourly = df_hourly.withColumn('DATETIME', 
                                 concat(col('date'), lit(' '), col('hour'), lit(':00:00')))

# Convert 'DATETIME' to timestamp type
df_hourly = df_hourly.withColumn('DATETIME', 
                                 F.to_timestamp(df_hourly['DATETIME'], 'yyyy-MM-dd HH:mm:ss'))

# Select and reorder the columns
df_hourly = df_hourly.select('DATETIME', 'HOURLY_CONSUMPTION_MW', 'COUNTRY')

# COMMAND ----------

df_hourly.count()

# COMMAND ----------

display(df_hourly)

# COMMAND ----------

# MAGIC %md 
# MAGIC * Writes the transformed DataFrame into a new table named final_consumption_countries_hourly in the df_dev database. The mode overwrite is used to replace the existing data in the table (if any).

# COMMAND ----------

# Write the DataFrame into a new table
df_hourly.write.format('delta').mode('overwrite').saveAsTable('df_dev.final_consumption_countries_hourly')

# COMMAND ----------


