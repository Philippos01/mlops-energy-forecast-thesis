# Databricks notebook source
# MAGIC %run "/Repos/filippos.priovolos01@gmail.com/mlops-energy-forecast-thesis/MLOps Thesis Pipeline/Workflow Config/Daily Inference"

# COMMAND ----------

# MAGIC %md
# MAGIC * The code uses Apache Spark SQL to query and manipulate data.
# MAGIC * It first selects data from a table for a specific date and casts the DATETIME column to a string, storing the result in a temporary view called 'daily_features'.
# MAGIC * It then performs a merge operation between a target table 'predictions_xgb' and the temporary view.
# MAGIC * For rows that have matching DATETIME and COUNTRY in both the target table and temporary view, it updates the RESIDUAL and ACTUAL_CONSUMPTION columns in the target table based on the data in the temporary view.

# COMMAND ----------

spark.sql(f"""SELECT *, CAST(DATETIME AS STRING) AS STR_DATETIME
              FROM db_monitor.final_monitoring_consumption_countries_hourly
              WHERE DATETIME >= '{date} 00:00' AND DATETIME <= '{date} 23:59' """).createOrReplaceTempView('daily_features')

spark.sql(f"""
MERGE INTO df_dev.predictions_xgb A
USING daily_features B
ON A.DATETIME = B.STR_DATETIME AND A.COUNTRY = B.COUNTRY
WHEN MATCHED THEN
  UPDATE SET 
    A.RESIDUAL = A.PREDICTED_CONSUMPTION - B.HOURLY_CONSUMPTION_MW,
    A.ACTUAL_CONSUMPTION = B.HOURLY_CONSUMPTION_MW
""")


# COMMAND ----------

df = spark.sql(f"SELECT * FROM df_dev.predictions_xgb WHERE DATETIME >= '{date} 00:00' AND DATETIME <= '{date} 23:59' ")

# Convert the data types of ACTUAL_CONSUMPTION and PREDICTED_CONSUMPTION columns to DoubleType
df = df.withColumn('ACTUAL_CONSUMPTION', col('ACTUAL_CONSUMPTION').cast(DoubleType()))
df = df.withColumn('PREDICTED_CONSUMPTION', col('PREDICTED_CONSUMPTION').cast(DoubleType()))

valuesAndPreds = df.select(['ACTUAL_CONSUMPTION', 'PREDICTED_CONSUMPTION'])
valuesAndPreds = valuesAndPreds.rdd.map(tuple)

metrics = RegressionMetrics(valuesAndPreds)

# Squared Error
print("MSE = %s" % metrics.meanSquaredError)
print("RMSE = %s" % metrics.rootMeanSquaredError)

# Mean absolute error
print("MAE = %s" % metrics.meanAbsoluteError)


# COMMAND ----------

# Calculate the percentage difference by dividing the difference by the absolute value of actual consumption
df = df.withColumn('PERCENTAGE_DIFFERENCE', (col('RESIDUAL') / abs(col('ACTUAL_CONSUMPTION'))) * 100)

# Calculate the absolute value of the percentage difference
df = df.withColumn('ABS_PERCENTAGE_DIFFERENCE', abs(col('PERCENTAGE_DIFFERENCE')))

# Calculate the average absolute percentage difference
average_absolute_percentage_difference = df.selectExpr('avg(ABS_PERCENTAGE_DIFFERENCE)').collect()[0][0]

# Calculate the average percentage difference
average_percentage_difference = df.selectExpr('avg(PERCENTAGE_DIFFERENCE)').collect()[0][0]

display(df)
# Print the average percentage difference
print('Average Percentage Difference:', average_percentage_difference)
# Print the average absolute percentage difference
print('Average Absolute Percentage Difference:', average_absolute_percentage_difference)

# COMMAND ----------

display(df.filter(df['COUNTRY'] == 'greece'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Inference Data to main table

# COMMAND ----------

# Retrieve the predictions_xgb DataFrame using the table name
predictions_xgb = spark.table('df_dev.predictions_xgb')

# Select the columns from the first table and cast appropriate columns to match the second table's schema
merged_df = predictions_xgb.select(
    col('DATETIME').cast('timestamp').alias('DATETIME'),
    col('COUNTRY'),
    col('PREDICTED_CONSUMPTION').cast(DoubleType()).alias('HOURLY_CONSUMPTION_MW')
)

# Perform a merge operation to insert new records into the second table if they don't already exist
merged_df.createOrReplaceTempView('temp_table')

spark.sql("""
    MERGE INTO df_dev.final_consumption_countries_hourly AS target
    USING temp_table AS source
    ON target.DATETIME = source.DATETIME AND target.COUNTRY = source.COUNTRY
    WHEN NOT MATCHED THEN
        INSERT (DATETIME, HOURLY_CONSUMPTION_MW, COUNTRY)
        VALUES (source.DATETIME, source.HOURLY_CONSUMPTION_MW, source.COUNTRY)
""")

# COMMAND ----------


