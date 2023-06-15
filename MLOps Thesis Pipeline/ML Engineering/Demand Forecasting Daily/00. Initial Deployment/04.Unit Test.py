# Databricks notebook source
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %run "/Repos/filippos.priovolos01@gmail.com/mlops-energy-forecast-thesis/MLOps Thesis Pipeline/Workflow Config/Initial Deployment"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Datasets

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC * Load energy consumption data from a database into a Pandas DataFrame.
# MAGIC * Create a new column CONSUMPTION_ID by concatenating country codes with the date-time information.
# MAGIC * Convert the DATETIME column to a proper datetime data type for time-based operations.
# MAGIC * Define test labels, based on date-time ranges.
# MAGIC * Convert the subsets back into Spark DataFrames and select only the CONSUMPTION_ID, DATETIME, and HOURLY_CONSUMPTION_MW columns for further processing

# COMMAND ----------

# Load Consumption Region Table
consumption_countries_hourly = spark.table(f'{db}.final_consumption_countries_hourly').toPandas()
consumption_countries_hourly['CONSUMPTION_ID'] = consumption_countries_hourly.COUNTRY + '_' + consumption_countries_hourly.DATETIME.astype(str)
consumption_countries_hourly['DATETIME'] = pd.to_datetime(consumption_countries_hourly['DATETIME'])
# Split the labels into training and test
test_labels = consumption_countries_hourly.loc[(consumption_countries_hourly.DATETIME > test_start) & (consumption_countries_hourly.DATETIME <= test_end)]
# Transforms to Spark DataFranes
test_labels = spark.createDataFrame(test_labels).select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")

# COMMAND ----------

# MAGIC %md
# MAGIC * Search for runs: The mlflow.search_runs function is called to search for all runs associated with the specified experiment_id_training. The runs are sorted by start time in descending order, meaning the latest run will be the first one in the list. The result is stored in the runs variable.
# MAGIC
# MAGIC * Select the latest run: The latest_run_id is assigned the run ID of the first run in the runs list (i.e., the latest run). This ID will be used to retrieve the details of the latest run.
# MAGIC
# MAGIC * Get the latest run details: The mlflow.get_run function is called with the latest_run_id to retrieve the details of the latest run. The details are stored in the latest_run variable.
# MAGIC
# MAGIC * Get the logged metrics: The metrics logged during the latest run are extracted from the latest_run.data.metrics attribute and stored in the metrics variable.

# COMMAND ----------

# Search for all runs associated with the experiment ID, sorted by start time
runs = mlflow.search_runs(experiment_ids=experiment_id_training, order_by=["start_time desc"])

#Select the first run in the list (i.e., the latest run)
latest_run_id = runs.iloc[0]["run_id"]
latest_run = mlflow.get_run(latest_run_id)

# Get the metrics logged during the latest run
metrics = latest_run.data.metrics

# Print the metrics
for key, value in metrics.items():
    print(key, value)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Performance Testing

# COMMAND ----------

# MAGIC %md
# MAGIC * We define some thresholds for our model to meet

# COMMAND ----------

mse_threshold = 1000000000.0
mae_threshold = 30000.0
rmse_threshold = 40000.0
r2_threshold = 0.9
training_time_threshold = 3600.0

# COMMAND ----------

# MAGIC %md
# MAGIC The test_model_performance() function evaluates the performance of a model by comparing specific metrics against defined thresholds. It checks if metrics such as MSE, MAE, RMSE, R2 score, and training time meet the specified thresholds. Success or failure messages are printed for each test, and a boolean variable (all_tests_passed) is updated accordingly. The function returns the overall result indicating whether all tests passed (True) or if any of them failed (False).

# COMMAND ----------

def test_model_performance():
    all_tests_passed = True
    try:
        assert metrics['MSE'] < mse_threshold
        print(f"MSE test passed with {metrics['MSE']} mean squared error")
    except AssertionError:
        print(f"MSE test failed. Expected < {mse_threshold} but got {metrics['MSE']}")
        all_tests_passed = False

    try:
        assert metrics['MAE'] < mae_threshold
        print(f"MAE test passed with {metrics['MAE']} mean absolute error")
    except AssertionError:
        print(f"MAE test failed. Expected < {mae_threshold} but got {metrics['MAE']}")
        all_tests_passed = False

    try:
        assert metrics['RMSE'] < rmse_threshold
        print(f"RMSE test passed with {metrics['RMSE']} root mean squared error")
    except AssertionError:
        print(f"RMSE test failed. Expected < {rmse_threshold} but got {metrics['RMSE']}")
        all_tests_passed = False
    
    try:
        assert metrics['R2'] > r2_threshold
        print(f"R2 test passed with {metrics['R2']} score")
    except AssertionError:
        print(f"R2 test failed. Expected > {r2_threshold} but got {metrics['R2']}")
        all_tests_passed = False


    try:
        assert metrics['Training Time(sec)'] < training_time_threshold #1hour
        print(f"Model training time test passed with {metrics['Training Time(sec)']} seconds")
    except AssertionError:
        print(f"Model training time test failed. Expected < {training_time_threshold} seconds but got {metrics['Training Time(sec)']} seconds")
        all_tests_passed = False

    return all_tests_passed

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metrics Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC We create a DataFrame that shows the metric values and their corresponding thresholds, along with pass/fail status for each test. It checks if the metric values meet the defined thresholds and assigns "Test Passed" or "Test Failed" based on the comparison. The purpose is to provide a visual representation of the test results for easy interpretation and evaluation of the model's performance against the thresholds.

# COMMAND ----------

# Create a DataFrame with the metric values and their corresponding thresholds
df = spark.createDataFrame([
    ("MSE", metrics['MSE'], mse_threshold),
    ("MAE", metrics['MAE'], mae_threshold),
    ("RMSE", metrics['RMSE'], rmse_threshold),
    ("R2", metrics['R2'], r2_threshold),
    ("Training Time(sec)", metrics['Training Time(sec)'], training_time_threshold )
], ["Metric", "Value", "Threshold"])

# Cast the "Threshold" column to DoubleType
df = df.withColumn("Threshold", df["Threshold"].cast(DoubleType()))
df = df.withColumn("Pass", when(df["Metric"].isin(["MSE", "MAE", "RMSE","Training Time(sec)"]), df["Value"] <= df["Threshold"]).otherwise(df["Value"] >= df["Threshold"]))
# Add a column to show pass/fail as strings
df = df.withColumn("Status", when(df["Pass"], "Test Passed").otherwise("Test Failed"))
# Show the DataFrame
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Staging

# COMMAND ----------

# MAGIC %md
# MAGIC * The code automates the transition of the latest version of a registered model to the staging environment.
# MAGIC * It checks the performance of the model using performance tests.
# MAGIC * If all performance tests pass, the model is transitioned to the staging environment.
# MAGIC * If any test fails, the model is not staged and a message is printed indicating the failure.
# MAGIC * The purpose is to ensure that only models meeting the performance criteria are moved to the staging environment.

# COMMAND ----------

def proceed_model_to_staging():
    # Get the latest version of the registered model
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_latest_versions(model_name, stages=["None"])[0].version

    # Define the endpoint URL
    endpoint_url = f"https://{databricks_instance}/api/2.0/mlflow/databricks/model-versions/transition-stage"

    stage = 'Staging' #Define the stage you want your model to transit
    comment = "Transitioning to staging environment after performance testing"
    headers = { "Authorization": "Bearer " + access_token }

    request_body = {
        "version": f"{model_version}",
        "name": model_name, 
        "stage" : stage, #Specifies the environment we want to transit our model
        "archive_existing_versions": False, #Specifies whether to archive all current model versions in the target stage.
        "comment": comment 
    }

    # Make the request
    response = requests.post(endpoint_url, headers=headers,json=request_body)

    # Check the response status code
    if response.status_code == 200:
        print("Model version transitioned to staging")
    else:
        print(f"Error transitioning model version to staging: {response.text}")


# Call function for staging

all_tests_passed = test_model_performance()
# run performance tests here
if all_tests_passed:
    # proceed with model staging
    proceed_model_to_staging()
else:
    print("Model performance tests failed. Model will not be staged.")

# COMMAND ----------


