# Databricks notebook source
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %run "/Repos/filippos.priovolos01@gmail.com/mlops-energy-forecast-thesis/MLOps Thesis Pipeline/Workflow Config/Initial Deployment"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

date_object = datetime.strptime(train_end, '%Y-%m-%d')
new_train_end = (date_object + relativedelta(months=3)).strftime('%Y-%m-%d')
date_object = datetime.strptime(test_start, '%Y-%m-%d')
new_test_start = (date_object + relativedelta(months=3)).strftime('%Y-%m-%d')
new_test_end = '2023-01-01'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load all the Current Data

# COMMAND ----------

# MAGIC %md
# MAGIC * Load energy consumption data from a database into a Pandas DataFrame.
# MAGIC * Create a new column CONSUMPTION_ID by concatenating country codes with the date-time information.
# MAGIC * Convert the DATETIME column to a proper datetime data type for time-based operations.
# MAGIC * Define test labels based on date-time ranges.
# MAGIC * Convert the test labels back into Spark DataFrames and select only the CONSUMPTION_ID, DATETIME, and HOURLY_CONSUMPTION_MW columns for further processing

# COMMAND ----------

# Load Consumption Region Table
consumption_countries_hourly = spark.table('df_dev.final_consumption_countries_hourly').toPandas()
consumption_countries_hourly['CONSUMPTION_ID'] = consumption_countries_hourly.COUNTRY + '_' + consumption_countries_hourly.DATETIME.astype(str)
consumption_countries_hourly['DATETIME'] = pd.to_datetime(consumption_countries_hourly['DATETIME'])
test_labels = consumption_countries_hourly.loc[(consumption_countries_hourly.DATETIME > new_test_start) & (consumption_countries_hourly.DATETIME <= new_test_end)]
test_labels = spark.createDataFrame(test_labels).select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Initial Deployment Training Runs Based on Experiment ID

# COMMAND ----------

# MAGIC %md
# MAGIC This code snippet uses the mlflow library to search for experiment runs based on a specific experiment ID (experiment_id_training), orders them by the Mean Absolute Error (MAE) metric, and stores the results in a DataFrame called runs_training. It then displays the first 5 rows of this DataFrame.

# COMMAND ----------

runs_training = mlflow.search_runs(experiment_ids=experiment_id_training,
                          order_by=['metrics.MAE'])
runs_training.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Best Runs of Past Month for the Initial Model Training

# COMMAND ----------

#earliest_start_time = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
#recent_runs = runs_training[runs_training.start_time >= earliest_start_time]
runs_training = runs_training.assign(Run_Date=runs_training.start_time.dt.floor(freq='D'))

# Filter the rows to only include those with non-null values in the "metrics.MAE" column
runs_training = runs_training[runs_training['metrics.MAE'].notna()]
#print("Length of recent_runs before filtering: ", len(runs_training))
#print("Length of recent_runs after filtering: ", len(runs_training))

best_runs_per_day_idx = runs_training.groupby(['Run_Date'])['metrics.MAE'].idxmin()
best_runs = runs_training.loc[best_runs_per_day_idx]

# Select the required columns for display
metrics_columns = ['Run_Date', 'metrics.MAE', 'metrics.Training Time(sec)', 'metrics.RMSE', 'metrics.R2', 'metrics.MSE']
display(best_runs[metrics_columns])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Retraining Best Runs Based on Experiment Id

# COMMAND ----------

runs_retraining = mlflow.search_runs(experiment_ids=experiment_id_retraining,
                          order_by=['metrics.MAE'])
runs_retraining.head(5)

# COMMAND ----------

#earliest_start_time = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
#recent_runs = runs_retraining[runs_retraining.start_time >= earliest_start_time]
runs_retraining = runs_retraining.assign(Run_Date=runs_retraining.start_time.dt.floor(freq='D'))

# Filter the rows to only include those with non-null values in the "metrics.MAE" column
runs_retraining = runs_retraining[runs_retraining['metrics.MAE'].notna()]
#print("Length of recent_runs before filtering: ", len(runs_retraining))
#print("Length of recent_runs after filtering: ", len(recent_runs))

best_runs_per_day_idx = runs_retraining.groupby(['Run_Date'])['metrics.MAE'].idxmin()
best_runs = runs_retraining.loc[best_runs_per_day_idx]

# Select the required columns for display
metrics_columns = ['Run_Date', 'metrics.MAE', 'metrics.Training Time(sec)', 'metrics.RMSE', 'metrics.R2', 'metrics.MSE']
display(best_runs[metrics_columns])


# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Number of Initial Training Runs for Past Month 

# COMMAND ----------

# MAGIC %md
# MAGIC * Calculates the date 30 days ago.
# MAGIC * Filters experiment runs from the last 30 days.
# MAGIC * Adds a column representing the date of each run.
# MAGIC * Groups runs by date and counts the number of runs per day.
# MAGIC * Formats the date for display.
# MAGIC * Renames a column for clarity.
# MAGIC * Displays a DataFrame showing the number of experiment runs for each day over the last 30 days.

# COMMAND ----------

earliest_start_time = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
recent_runs = runs_training[runs_training.start_time >= earliest_start_time]

recent_runs['Run Date'] = recent_runs.start_time.dt.floor(freq='D')

runs_per_day = recent_runs.groupby(
  ['Run Date']
).count()[['run_id']].reset_index()
runs_per_day['Run Date'] = runs_per_day['Run Date'].dt.strftime('%Y-%m-%d')
runs_per_day.rename({ 'run_id': 'Number of Runs' }, axis='columns', inplace=True)

display(runs_per_day)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Number of Retraining Runs for Past Month 

# COMMAND ----------

# MAGIC %md
# MAGIC * Calculates the date 30 days ago.
# MAGIC * Filters experiment runs from the last 30 days.
# MAGIC * Adds a column representing the date of each run.
# MAGIC * Groups runs by date and counts the number of runs per day.
# MAGIC * Formats the date for display.
# MAGIC * Renames a column for clarity.
# MAGIC * Displays a DataFrame showing the number of experiment runs for each day over the last 30 days.

# COMMAND ----------

earliest_start_time = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
recent_runs = runs_retraining[runs_retraining.start_time >= earliest_start_time]

recent_runs['Run Date'] = recent_runs.start_time.dt.floor(freq='D')

runs_per_day = recent_runs.groupby(
  ['Run Date']
).count()[['run_id']].reset_index()
runs_per_day['Run Date'] = runs_per_day['Run Date'].dt.strftime('%Y-%m-%d')
runs_per_day.rename({ 'run_id': 'Number of Runs' }, axis='columns', inplace=True)

display(runs_per_day)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison (Staging - Production)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Request for Latest Models of Each Environment

# COMMAND ----------

# MAGIC %md
# MAGIC * Sets up HTTP headers with an authorization token and query parameters with a model name for an API request.
# MAGIC * Sends a GET request to the MLflow REST API of a Databricks instance to retrieve all versions of a registered machine learning model.
# MAGIC * Checks if the API response was successful. If not, an exception is raised with an error message containing the status code and error message from the API response.
# MAGIC * If the API response was successful, it extracts the registered model details from the response JSON.
# MAGIC * Prints the list of model versions in a JSON formatted output.

# COMMAND ----------

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
# MAGIC ## Retrieve Latest Staging & Production Models

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

# MAGIC %md
# MAGIC ### Perform Batch Score to the Models

# COMMAND ----------

# MAGIC %md
# MAGIC * Perform batch scoring for the latest deployed staging & production model

# COMMAND ----------

# Check if both the production and staging URIs have been retrieved
if not prod_uri:
    raise Exception("Failed to retrieve the production model URI.")
if not staging_uri:
    raise Exception("Failed to retrieve the staging model URI.")

# Score the test dataset using the production and staging models
staging_scores = fs.score_batch(staging_uri, test_labels, result_type='float')
prod_scores = fs.score_batch(prod_uri, test_labels, result_type='float')

prod_scores = prod_scores.withColumnRenamed("prediction", "prod_prediction")
staging_scores = staging_scores.withColumnRenamed("prediction", "staging_prediction")

display(prod_scores)
display(staging_scores)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join Staging & Production Dataframes

# COMMAND ----------

# Join the two dataframes on the `consumption_id` column, keeping all columns from `staging_df` and only the `prod_prediction` column from `prod_df`
merged_df = staging_scores.join(prod_scores.select('consumption_id', 'prod_prediction'), 'consumption_id', 'inner').select(staging_scores.columns + [col('prod_prediction')])

# Define the column expression to extract the correct region
country_col_expr = (
    concat(*[
        when(col(country) == 1, country).otherwise("")
        for country in countries
    ])
)

# Add a new column to the DataFrame with the concatenated region name
merged_df = merged_df.withColumn("COUNTRY", country_col_expr)

display(merged_df)

# COMMAND ----------

from pyspark.sql.functions import year, month, col, concat, when,weekofyear
# Filter to keep only the data for April 2022
filtered_df = merged_df.filter((year(col('DATETIME')) == 2022) & (month(col('DATETIME')) == 4) & (weekofyear(col('DATETIME')) == 14))

# Display the filtered DataFrame
display(filtered_df)

# COMMAND ----------

display(merged_df.filter(col('Country') == 'greece'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare Staging vs Production

# COMMAND ----------

# MAGIC %md
# MAGIC A function named calculate_smape is defined. This function takes three arguments:
# MAGIC * df: A DataFrame that contains the prediction and actual values.
# MAGIC * prediction_col: The name of the column that contains the predicted values.
# MAGIC * actual_col: The name of the column that contains the actual values.
# MAGIC     
# MAGIC The function computes the SMAPE based on these input values. The SMAPE is calculated as the mean absolute difference between the predicted and actual values, divided by the average of the absolute predicted and actual values, all multiplied by 100.
# MAGIC
# MAGIC 1. The calculate_smape function is used to calculate the SMAPE for the staging and production models, using the respective prediction and actual values.
# MAGIC 1. Based on the calculated SMAPE values, the code determines which model (staging or production) is better. The model with the lower SMAPE is considered the better one since a lower SMAPE indicates a better fit of the model.

# COMMAND ----------

def calculate_smape(df, prediction_col, actual_col):
    from pyspark.sql.functions import abs
    # Calculate SMAPE using PySpark functions
    diff = col(prediction_col) - col(actual_col)
    denominator = (abs(col(prediction_col)) + abs(col(actual_col))) / 2
    smape = df.select(mean((abs(diff) / denominator) * 100).alias("SMAPE")).collect()[0]["SMAPE"]
    
    return smape

# Calculate SMAPE for staging predictions
staging_smape = calculate_smape(staging_scores, 'staging_prediction', 'HOURLY_CONSUMPTION_MW')
print(f"Staging Model SMAPE: {staging_smape}%")

# Calculate SMAPE for production predictions
prod_smape = calculate_smape(prod_scores, 'prod_prediction', 'HOURLY_CONSUMPTION_MW')
print(f"Production Model SMAPE: {prod_smape}%")

# Determine which model is better based on SMAPE
if staging_smape < prod_smape:
    print(f"Staging Model is better with a SMAPE of {staging_smape:.2f}%.")
    best_model = staging_uri
else:
    print(f"Production Model is better with a SMAPE of {prod_smape:.2f}%.")
    best_model = prod_uri

# Print the URI of the best model
print(best_model)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Transit the Best Model to Production Stage

# COMMAND ----------

# MAGIC %md
# MAGIC * The function initializes the MlflowClient and assigns it to the variable client.
# MAGIC * It retrieves the latest version of the registered model in the Staging stage by calling the get_latest_versions method of the MlflowClient and assigns it to the variable model_version.
# MAGIC * It defines the endpoint URL for sending the transition request to the Databricks MLflow API. The URL is assigned to the variable endpoint_url.
# MAGIC * The stage to which the model should be transitioned is defined as 'Production'. Additionally, a comment for the transition request is set.
# MAGIC * It sets the request headers to include the authorization token.
# MAGIC * It constructs the request body, which includes the version of the model to be transitioned, the model name, the desired stage, a flag indicating whether to archive existing versions in the target stage, the comment, and a flag to indicate that this is a transition request.
# MAGIC * It sends a POST request to the API endpoint with the defined headers and request body.
# MAGIC * Finally, it checks the status code of the response. If the status code is 200, it prints a message indicating that the model transition request was sent successfully. Otherwise, it prints an error message with the response text.

# COMMAND ----------

def request_model_transition_to_production():
    
    # Get the latest version of the registered model in the Staging stage
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Define the endpoint URL
    endpoint_url = f"https://{databricks_instance}/api/2.0/mlflow/transition-requests/create"

    stage = 'Production' #Define the stage you want your model to transit
    comment = "Requesting transition to Production environment after comparing models"
    headers = { "Authorization": "Bearer " + access_token }

    request_body = {
        "version": f"{model_version}",
        "name": model_name, 
        "stage" : stage, #Specifies the environment we want to transit our model
        "archive_existing_versions": True, #Specifies whether to archive all current model versions in the target stage.
        "comment": comment,
        "request_transition": True
    }
    print(model_version,model_name)
    # Make the request
    response = requests.post(endpoint_url, headers=headers,json=request_body)

    # Check the response status code
    if response.status_code == 200:
        print("Model version transition request sent")
    else:
        print(f"Error sending transition request: {response.text}")


# COMMAND ----------

# MAGIC %md
# MAGIC * Initializes an MLflow Client by assigning it to the variable client. The MLflow Client provides a programmatic way to interact with an MLflow tracking server.
# MAGIC * Extracts the model_name and model_version from the best_model string, which presumably holds a URI for the model. It does so by splitting the string and accessing the relevant parts.
# MAGIC * Queries the current stage (e.g., Staging, Production) of the model version using the get_model_version method of the MLflow Client. It assigns this stage to the variable best_model_stage.
# MAGIC * Checks if the current stage of the best model is not 'Production'. If it isn't, it calls the previously defined function request_model_transition_to_production to request transitioning this model to the Production stage.
# MAGIC * If the best model is already in the Production stage, it prints "Best model is already in Production".

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model_name = best_model.split('/')[1]
model_version = best_model.split('/')[-1]
best_model_stage = client.get_model_version(name=model_name, version=model_version).current_stage
if best_model_stage != 'Production':
    # transit model to production
    request_model_transition_to_production()
else:
    print("Best model is already in Production")


# COMMAND ----------


