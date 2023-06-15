# Databricks notebook source
# MAGIC %run "/Repos/CI ADO Repo/01.Develop/Config/Daily Inference"

# COMMAND ----------

test_labels = '2018-06-25' 
regions = ['AEP','COMED','DAYTON','DEOK','DOM','DUQ','EKPC','FE','PJME','PJMW'] # Exlude region NI

# COMMAND ----------

fs = feature_store.FeatureStoreClient()
df = fs.read_table(name=f'{db}.forecasting_features_daily')# define the date you want to filter by
target_date = "2018-06-25"

# filter the DataFrame to keep only the rows with the target date
filtered_df = df.filter(col("DATE") == test_labels).select("CONSUMPTION_ID","DATE")

# show the resulting DataFrame
filtered_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve Models from Staging environment

# COMMAND ----------

import requests
import json

model_name = 'energy_consumption_model'
access_token = 'dapi6bd0817a274b1baa27ec51d055818d62-2'
databricks_instance = 'adb-4287679896047209.9.azuredatabricks.net'

# Set the headers and query parameters for the request
headers = {"Authorization": f"Bearer {access_token}"}
params = {"name": model_name}

# Send the GET request to the MLflow REST API to retrieve all versions of the model
response = requests.get(f"https://{databricks_instance}/api/2.0/preview/mlflow/registered-models/get", headers=headers, params=params)

# Check if the response was successful
if response.status_code != 200:
    raise Exception(f"Failed to retrieve registered models. Status code: {response.status_code}. Error message: {response.json()['error_code']}: {response.json()['message']}")

model_versions = response.json()['registered_model']

# Print the list of model versions
print(json.dumps(model_versions, indent=2))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Models Comparison Test

# COMMAND ----------

prod_uri = None
staging_uri = None

for i in range(len(model_versions['latest_versions'])):
    if model_versions['latest_versions'][i]['current_stage'] == 'Production':
        prod_uri = f"models:/{model_name}/{model_versions['latest_versions'][i]['version']}"
    elif model_versions['latest_versions'][i]['current_stage'] == 'Staging':
        staging_uri = f"models:/{model_name}/{model_versions['latest_versions'][i]['version']}"

if prod_uri is None:
    print('No model versions found in production')
else:
    print(f'Latest production model version: {prod_uri}')
    
if staging_uri is None:
    print('No model versions found in staging')
else:
    print(f'Latest staging model version: {staging_uri}')


# COMMAND ----------

# Check if both the production and staging URIs have been retrieved
if not prod_uri:
    raise Exception("Failed to retrieve the production model URI.")
if not staging_uri:
    raise Exception("Failed to retrieve the staging model URI.")

# Score the test dataset using the production and staging models
staging_scores = fs.score_batch(staging_uri, filtered_df, result_type='float')
prod_scores = fs.score_batch(prod_uri, filtered_df, result_type='float')


staging_scores = staging_scores.withColumnRenamed("prediction", "staging_prediction")
prod_scores = prod_scores.withColumnRenamed("prediction", "prod_prediction")

display(prod_scores)
display(staging_scores)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert to Pandas df

# COMMAND ----------

# convert the dataframes to Pandas
staging_scores = staging_scores.toPandas()
prod_scores = prod_scores.toPandas()
prod_scores['staging_prediction'] = staging_scores['staging_prediction'] #pass staging_pred to the prod_scores df


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Regions Column

# COMMAND ----------

# Extract the region name from the 'Consumption_id' column
prod_scores['REGION_NAME'] = prod_scores['CONSUMPTION_ID'].str.extract(r'^(.*?)_')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Dataset and Retrieve Actual Values

# COMMAND ----------

df_predictions = spark.table(f'{db}.performance_monitor')
actual_values =  df_predictions.filter(df_predictions.DATE == "2018-06-25").toPandas()
actual_values

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge Dataframes on Regions 

# COMMAND ----------

prod_scores = pd.merge(prod_scores, actual_values[['ACTUAL_VALUE','REGION_NAME']], on='REGION_NAME', how='left')
prod_scores

# COMMAND ----------

from pyspark.sql import SparkSession
# create a SparkSession
spark = SparkSession.builder.appName("pandas-to-pyspark").getOrCreate()

# convert to PySpark DataFrame
df = spark.createDataFrame(prod_scores)

# display the PySpark DataFrame
display(df)

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go
# Remove rows with NaN values in the 'actual_values' column
prod_scores_clean = prod_scores.dropna(subset=['ACTUAL_VALUE'])
# Create a scatter plot using Plotly
fig = px.scatter(
    data_frame=prod_scores_clean,
    x="prod_prediction",
    y="staging_prediction",
    color="REGION_NAME",
    size="ACTUAL_VALUE",
    hover_data=["prod_prediction", "staging_prediction", "DATE", "ACTUAL_VALUE"],
    title="Comparison of Prod and Staging Predictions vs. Actual Values"
)

# Add a perfect predictions line to the scatter plot
fig.add_trace(go.Scatter(
    x=[prod_scores["ACTUAL_VALUE"].min(), prod_scores["ACTUAL_VALUE"].max()],
    y=[prod_scores["ACTUAL_VALUE"].min(), prod_scores["ACTUAL_VALUE"].max()],
    mode="lines",
    line=dict(color="gray", dash="dash")
))

# Customize the plot layout
fig.update_layout(
    xaxis_title="Prod Prediction",
    yaxis_title="Staging Prediction",
    legend_title="REGION_NAME",
    hoverlabel=dict(bgcolor="white", font_size=12),
    font=dict(family="Arial", size=14)
)

# Display the plot
fig.show()



# COMMAND ----------

@udf
def udf_something(a):
    return a + 1
spark.sql('select * from a').withColumn('dsds',  udf_something).orderby().groupBy()

# COMMAND ----------

df.select('a')count()
df.select('a').distinct().count()
