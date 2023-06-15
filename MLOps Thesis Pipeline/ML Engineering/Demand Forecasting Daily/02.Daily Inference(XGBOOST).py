# Databricks notebook source
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %run "/Repos/filippos.priovolos01@gmail.com/mlops-energy-forecast-thesis/MLOps Thesis Pipeline/Workflow Config/Daily Inference"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

input_table = 'hourly_forecasting_features'
output_table = 'predictions_xgb'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Inference Input

# COMMAND ----------

inference_df = spark.sql(f"SELECT CONSUMPTION_ID, DATETIME FROM {db}.{input_table} WHERE DATETIME BETWEEN '{date} 00:00:00' AND '{date} 23:00:00'")
#inference_data = inference_df.drop("CONSUMPTION_ID","DATETIME")
display(inference_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model's Prediction

# COMMAND ----------

client=mlflow.tracking.MlflowClient()
latest_version= client.get_latest_versions(model_name,stages=['Production'])[0].version

# COMMAND ----------

# MAGIC %md
# MAGIC * The following code performs batch scoring on the inference_df(which is the future date we want to predict), using the latest model deployed in Production 

# COMMAND ----------

results = fs.score_batch(
    f"models:/{model_name}/{latest_version}",
    inference_df,
    result_type="float",
)
display(results)

# COMMAND ----------

greece_predictions = results.filter(results["greece"] == 1).select("prediction","HOUR")
greece_predictions.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Store Results

# COMMAND ----------

# MAGIC %md
# MAGIC * It selects relevant columns from the initial results and converts them into a Pandas DataFrame for easy manipulation.
# MAGIC * It renames and creates new columns, including predicted consumption, country extracted from the consumption ID, and placeholders for actual consumption and residuals.
# MAGIC * The Pandas DataFrame is converted back to a Spark DataFrame with a specific selection of columns.
# MAGIC * Data types for certain columns are cast to float.
# MAGIC * Finally, the data is registered as a temporary SQL view named 'Inference_Output', allowing for SQL-based analysis and querying.

# COMMAND ----------

df = results.select(['CONSUMPTION_ID', 'DATETIME', 'prediction']).toPandas()
df.rename(columns={'prediction': 'PREDICTED_CONSUMPTION'}, inplace=True)
df['DATETIME'] = df.DATETIME.astype(str)
df['COUNTRY'] = df['CONSUMPTION_ID'].apply(lambda x: x.split('_')[0])
df['ACTUAL_CONSUMPTION'] = None
df['RESIDUAL'] = None
df['MODEL_USED'] = f"models:/{model_name}/{latest_version}"
output_cols = ['DATETIME', 'COUNTRY', 'PREDICTED_CONSUMPTION', 'ACTUAL_CONSUMPTION', 'RESIDUAL', 'MODEL_USED']
output_df = spark.createDataFrame(df[output_cols])
output_df.withColumn('ACTUAL_CONSUMPTION', col('ACTUAL_CONSUMPTION').cast('float'))\
         .withColumn('RESIDUAL', col('RESIDUAL').cast('float'))\
         .createOrReplaceTempView('Inference_Output')

# COMMAND ----------

# MAGIC %md
# MAGIC * It prepares the list of columns to be inserted or updated.
# MAGIC * It uses Spark SQL to merge data from a temporary view Inference_Output into a target table.
# MAGIC * If a record with matching DATETIME and COUNTRY is found, it updates the existing record in the target table with the new data.
# MAGIC * If no matching record is found, it inserts the new data as a new record in the target table.

# COMMAND ----------

insert_columns = [f"B.{col}" for col in output_cols]
update_columns = [f"{col}=B.{col}" for col in output_cols]
spark.sql(f"""
MERGE INTO {db}.{output_table} A
USING Inference_Output B
ON A.DATETIME = B.DATETIME AND A.COUNTRY = B.COUNTRY
WHEN MATCHED THEN
  UPDATE SET
    {', '.join(update_columns)}
WHEN NOT MATCHED
  THEN INSERT (
    {', '.join(output_cols)}
) VALUES (  
   {', '.join(insert_columns)}
)
""")

# COMMAND ----------


