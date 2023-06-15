# Databricks notebook source
# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

# MAGIC %run "/Repos/filippos.priovolos01@gmail.com/mlops-energy-forecast-thesis/MLOps Thesis Pipeline/Workflow Config/Initial Deployment"

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Configuration

# COMMAND ----------

config = spark.sql('select train_start, train_end, test_end from df_dev.config_retrain_monthly').collect()
train_start, train_end, test_end = config[0]

# train_start = '2013-06-01' #the retrain start date
# train_end = '2018-06-20'   #the retrain end date (20/06/2018 - 30/06/2018) 10 days for testing
# test_end = '2018-06-30'
# Convert start and end dates to datetime objects
start_new_train_date = pd.to_datetime(validation_end) + pd.DateOffset(days=1) # 1 day after validation end
end_new_train_date = pd.to_datetime(train_end)
start_new_test_date = pd.to_datetime(train_end) + pd.DateOffset(days=1) # 1 day after train end
end_new_test_date = pd.to_datetime(test_end)
# Calculate the number of days between start and end dates
num_new_train_days = (end_new_train_date - start_new_train_date).days
num_new_test_days = (end_new_test_date - start_new_test_date).days

# COMMAND ----------

# MAGIC %md
# MAGIC ##Load Datasets

# COMMAND ----------

# Load Consumption Region Table
consumption_regions_daily = spark.table(f'{db}.{consumption_regions_daily}')
consumption_regions_daily = consumption_regions_daily.withColumn('CONSUMPTION_ID', concat(col('REGION'), lit('_'), col('DATE')))
consumption_regions_daily = consumption_regions_daily.withColumn('DATE', col('DATE').cast(DateType()))

# Split the labels into training and test
train_labels = consumption_regions_daily.filter((col('DATE') >= train_start) & (col('DATE') <= train_end))
test_labels = consumption_regions_daily.filter((col('DATE') > train_end) & (col('DATE') <= test_end))
#val_labels = consumption_regions_daily.filter((col('DATE') > test_end) & (col('DATE') <= validation_end))

# Select the required columns
train_labels = train_labels.select("CONSUMPTION_ID", "DATE", "DAILY_CONSUMPTION_MW")
test_labels = test_labels.select("CONSUMPTION_ID", "DATE", "DAILY_CONSUMPTION_MW")
#val_labels = val_labels.select("CONSUMPTION_ID", "DATE", "DAILY_CONSUMPTION_MW")

# COMMAND ----------

def load_data(table_name, labels, lookup_key, ts_lookup_key):
    # In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key, timestamp_lookup_key=ts_lookup_key)]

    # fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
    training_set = fs.create_training_set(labels, 
                                          model_feature_lookups, 
                                          label="DAILY_CONSUMPTION_MW", 
                                          exclude_columns=["CONSUMPTION_ID", "DATE"])
    training_df = training_set.load_df()

    return training_set, training_df

# Cast the 'DATE' column to 'TIMESTAMP' data type
train_labels = train_labels.withColumn('DATE', col('DATE').cast(TimestampType()))
test_labels = test_labels.withColumn('DATE', col('DATE').cast(TimestampType()))
#val_labels = val_labels.withColumn('DATE', col('DATE').cast(TimestampType()))

# Load the data for the training set
training_set, train_df = load_data(f'{db}.forecasting_features_daily', train_labels, 'CONSUMPTION_ID', 'DATE')

# Load the data for the test set
_, test_df = load_data(f'{db}.forecasting_features_daily', test_labels, 'CONSUMPTION_ID', 'DATE')

# Load the data for the validation set
#_, val_df = load_data(f'{db}.forecasting_features_daily', val_labels, 'CONSUMPTION_ID', 'DATE')


# COMMAND ----------

concatenated_df = train_df.union(test_df)
display(concatenated_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Drift Test

# COMMAND ----------

# Convert year, month, and day columns to string and pad month and day with zeros
train_df_str = train_df.withColumn("YEAR", col("YEAR").cast("string"))
train_df_str = train_df_str.withColumn("MONTH", lpad(col("MONTH").cast("string"), 2, '0'))
train_df_str = train_df_str.withColumn("DAY_OF_MONTH", lpad(col("DAY_OF_MONTH").cast("string"), 2, '0'))

# Concatenate year, month, and day columns with '-' separator and convert to date
date_df = train_df_str.withColumn(
    'date', 
    to_date(concat_ws('-', train_df_str["YEAR"], train_df_str["MONTH"], train_df_str["DAY_OF_MONTH"]), 'yyyy-MM-dd')
)

# Extract the most recent num_new_train_days days of data
max_date_row = date_df.agg(max_("date").alias("max_date")).first()
max_date = max_date_row["max_date"]

new_data = date_df.filter(col("date") >= date_sub(lit(max_date), num_new_train_days))

# Extract a random sample of num_new_train_days * 11 days data
old_data = date_df.filter(
    col("date") < date_sub(lit(max_date), num_new_train_days)
).orderBy(rand()).limit(num_new_train_days * 11)

# Concatenate the new and old data
all_data = new_data.union(old_data)


# COMMAND ----------

# Apply the ks_2samp test to each feature
for feature_name in regions:
    old_feature_data = old_data.select(feature_name).rdd.flatMap(lambda x: x).collect()
    new_feature_data = new_data.select(feature_name).rdd.flatMap(lambda x: x).collect()
    
    _, p_value = ks_2samp(old_feature_data, new_feature_data)
    
    if p_value < 0.05:
        print(f"The distribution of {feature_name} has drifted.")
    else:
        print(f"The distribution of {feature_name} has not drifted.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrain the Machine Learning Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC * Create temporal views of these dataframes in order to be passed int the source notebook

# COMMAND ----------

concatenated_df.createOrReplaceTempView("concatenated_df_view")
train_df.createOrReplaceTempView("train_df_view")
test_df.createOrReplaceTempView("test_df_view")

# COMMAND ----------

# MAGIC %run "/Repos/CI ADO Repo/01.Develop/Utils/Train ML Pipeline"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register Retrained Model to MLflow

# COMMAND ----------


    with mlflow.start_run(nested=True) as run:

        experiment = mlflow.get_experiment(experiment_id_retraining)
        if experiment:
            experiment_name = experiment.name
            mlflow.set_experiment(experiment_name)
            print(f"Active experiment set to '{experiment_name}'")
        else:
            print(f"No experiment found with name '{experiment_name}'")
        
        # Define the output schema
        output_schema = sch.Schema([sch.ColSpec("float", "DAILY_CONSUMPTION_MW")])

        # Create a model signature from the input and output schemas
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Log the model input schema
        schema = {"input_schema": list(concatenated_df.columns[:-1]),"output_schema":concatenated_df.columns[-1]}
        mlflow.log_dict(schema, "schema.json")

        # Log some tags for the model
        mlflow.set_tags(tags)

        # Log some parameters for the model
        mlflow.log_dict(hyperparameters, "hyperparams.json")

        # Log the evaluation metrics as metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        #Log the time taken to train as metric
        mlflow.log_metric("Training Time(sec)", training_time)

        # Log evaluation metrics as artifact
        metrics = {"R2": r2, "MSE": mse, "RMSE": rmse, 'MAE':mae,'Training Time(sec)':training_time}
        mlflow.log_dict(metrics, "metrics.json")

        # Log the model description as artifact
        mlflow.log_text(description, "description.txt")
        
        # Log the current timestamp as the code version
        mlflow.log_param("code_version", current_time)

        # Log additional important parameters for comparison
        mlflow.log_param("n_estimators", hyperparameters["n_estimators"])
        mlflow.log_param("max_depth", hyperparameters["max_depth"])
        mlflow.log_param("learning_rate", hyperparameters["learning_rate"])
        mlflow.log_param("training_data_size", training_size)
        mlflow.log_param("testing_data_size", testing_size)

        # Log the model with its signature
        mlflow.spark.log_model(xgb_model, artifact_path="model", signature=signature,pip_requirements=pip_requirements)

        # Register the model with its signature
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="pyspark_mlflow_model")

        # Get the latest model version(The one that we now registered)
        client = MlflowClient()
        model_version = client.get_latest_versions("pyspark_mlflow_model")[0].version

        # Save your data to a new DBFS directory for each run
        data_path = f"dbfs:/FileStore/Data_Versioning/data_model_v{model_version}.parquet"
        concatenated_df.write.format("parquet").save(data_path)
 
        # Log the DBFS path as an artifact
        with open("data_path.txt", "w") as f:
            f.write(data_path)
        mlflow.log_artifact("data_path.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Staging

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


# COMMAND ----------

all_tests_passed = True
# run performance tests here
if all_tests_passed:
    # proceed with model staging
    proceed_model_to_staging()
else:
    print("Model performance tests failed. Model will not be staged.")


# COMMAND ----------


