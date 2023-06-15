# Databricks notebook source
# MAGIC %md 
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %run "/Repos/filippos.priovolos01@gmail.com/mlops-energy-forecast-thesis/MLOps Thesis Pipeline/Workflow Config/Initial Deployment"

# COMMAND ----------

# MAGIC %md
# MAGIC ##Load Datasets

# COMMAND ----------

train_start = '2015-01-01' 
train_end = '2021-12-31'
test_start = '2022-01-01'
test_end = '2023-01-01'

# COMMAND ----------

# Load Consumption Region Table
consumption_countries_hourly = spark.table(f'{db}.{consumption_countries_hourly}')

# Update the key column construction in the PySpark code
consumption_countries_hourly = consumption_countries_hourly.withColumn('CONSUMPTION_ID', concat(col('COUNTRY'), lit('_'), col('DATETIME').cast('string')))

# Split the labels into training and test
train_labels = consumption_countries_hourly.filter((col('DATETIME') >= train_start) & (col('DATETIME') <= train_end))
test_labels = consumption_countries_hourly.filter((col('DATETIME') > test_start) & (col('DATETIME') <= test_end))
#val_labels = consumption_countries_hourly.filter((col('DATETIME') > test_end) & (col('DATETIME') <= validation_end))

# Select the required columns
train_labels = train_labels.select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")
test_labels = test_labels.select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")
#val_labels = val_labels.select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")

# COMMAND ----------

display(train_labels)

# COMMAND ----------

def load_data(table_name, labels, lookup_key, ts_lookup_key):
    # In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key, timestamp_lookup_key=ts_lookup_key)]
    # fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
    training_set = fs.create_training_set(labels, 
                                          model_feature_lookups, 
                                          label="HOURLY_CONSUMPTION_MW", 
                                          exclude_columns=["CONSUMPTION_ID", "DATETIME"])
    training_df = training_set.load_df()

    return training_set, training_df

# Cast the 'DATETIME' column to 'TIMESTAMP' data type
train_labels = train_labels.withColumn('DATETIME', col('DATETIME').cast(TimestampType()))
test_labels = test_labels.withColumn('DATETIME', col('DATETIME').cast(TimestampType()))
#val_labels = val_labels.withColumn('DATETIME', col('DATETIME').cast(TimestampType()))

# Load the data for the training set
training_set, train_df = load_data(f'{db}.hourly_forecasting_features', train_labels, 'CONSUMPTION_ID', 'DATETIME')

# Load the data for the test set
_, test_df = load_data(f'{db}.hourly_forecasting_features', test_labels, 'CONSUMPTION_ID', 'DATETIME')

# Load the data for the validation set
#_, val_df = load_data(f'{db}.hourly_forecasting_features', val_labels, 'CONSUMPTION_ID', 'DATETIME')


# COMMAND ----------

display(train_df)

# COMMAND ----------

concatenated_df = train_df.union(test_df)
display(concatenated_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Train the Machine Learning Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the machine learning pipeline
# MAGIC Now that we have reviewed the data and prepared it as a DataFrame with numeric values, we're ready to train a model to predict future energy consumption.
# MAGIC
# MAGIC MLlib pipelines combine multiple steps into a single workflow, making it easier to iterate as we develop the model.
# MAGIC
# MAGIC In this example, we create a pipeline using the following functions:
# MAGIC
# MAGIC * `VectorAssembler`: Assembles the feature columns into a feature vector.
# MAGIC * `VectorIndexer`: Identifies columns that should be treated as categorical. This is done heuristically, identifying any column with a small number of distinct values as categorical. In this example, all the region columns are considered categorical(2 values)
# MAGIC * `SparkXGBRegressor`: Uses the SparkXGBRegressor estimator to learn how to predict energy consumption from the feature vectors.
# MAGIC * `CrossValidator`: The XGBoost regression algorithm has several hyperparameters. This notebook illustrates how to use hyperparameter tuning in Spark. This capability automatically tests a grid of hyperparameters and chooses the best resulting model.

# COMMAND ----------

# MAGIC %md
# MAGIC * The first step is to create the VectorAssembler and VectorIndexer steps.

# COMMAND ----------

# Remove the target column from the input feature set.
featuresCols = concatenated_df.columns
featuresCols.remove('HOURLY_CONSUMPTION_MW')

# vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
 
# vectorIndexer identifies categorical features and indexes them, and creates a new column "features". 
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=3)

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC * Next, define the model. To use distributed training, set num_workers to the number of spark tasks you want to concurrently run during training xgboost model.

# COMMAND ----------

# The next step is to define the model training stage of the pipeline. 
# The following command defines a XgboostRegressor model that takes an input column "features" by default and learns to predict the labels in the "cnt" column.
# Set `num_workers` to the number of spark tasks you want to concurrently run during training xgboost model.
xgb_regressor = SparkXGBRegressor(label_col="HOURLY_CONSUMPTION_MW")

# COMMAND ----------

# MAGIC %md
# MAGIC * The third step is to wrap the model you just defined in a CrossValidator stage. CrossValidator calls the XgboostRegressor estimator with different hyperparameter settings. It trains multiple models and selects the best one, based on minimizing a specified metric. In this example, the metric is root mean squared error (RMSE).

# COMMAND ----------

# Define a grid of hyperparameters to test:
#  - maxDepth: maximum depth of each decision tree 
#  - maxIter: iterations, or the total number of trees 
paramGrid = ParamGridBuilder()\
  .addGrid(xgb_regressor.max_depth, [8])\
  .addGrid(xgb_regressor.n_estimators, [200])\
  .addGrid(xgb_regressor.learning_rate, [0.1])\
  .build()

# Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
evaluator = RegressionEvaluator(metricName="rmse",
                                labelCol=xgb_regressor.getLabelCol(),
                                predictionCol=xgb_regressor.getPredictionCol())
 


# COMMAND ----------

# MAGIC %md
# MAGIC * Create the pipeline

# COMMAND ----------

pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, xgb_regressor])

# COMMAND ----------

 # Declare the CrossValidator, which performs the model tuning.
cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

# MAGIC %md
# MAGIC Train the pipeline:
# MAGIC
# MAGIC Now that we have set up the workflow, we can train the pipeline with a single call.
# MAGIC When we call fit(), the pipeline runs feature processing, model tuning, and training and returns a fitted pipeline with the best model it found. This step takes several minutes.

# COMMAND ----------

start_time = time.time()
cvModel  = cv.fit(train_df)
end_time = time.time()
# Retrieve best model in the pipeline
xgb_model = cvModel.bestModel.stages[-1]

# COMMAND ----------

# MAGIC %md
# MAGIC Make predictions and evaluate results:
# MAGIC
# MAGIC The final step is to use the fitted model to make predictions on the test dataset and evaluate the model's performance. The model's performance on the test dataset provides an approximation of how it is likely to perform on new data.
# MAGIC
# MAGIC Computing evaluation metrics is important for understanding the quality of predictions, as well as for comparing models and tuning parameters.

# COMMAND ----------

# MAGIC %md
# MAGIC The `transform()` method of the pipeline model applies the full pipeline to the input dataset. The pipeline applies the feature processing steps to the dataset and then uses the fitted Xgboost Regressor model to make predictions. The pipeline returns a DataFrame with a new column predictions.

# COMMAND ----------

predictions = cvModel.transform(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC A common way to evaluate the performance of a regression model is the calculate the root mean squared error (RMSE). The value is not very informative on its own, but you can use it to compare different models. `CrossValidator` determines the best model by selecting the one that minimizes RMSE.

# COMMAND ----------

display(predictions.select("HOURLY_CONSUMPTION_MW", "prediction", *featuresCols))

# COMMAND ----------

rmse = evaluator.evaluate(predictions)
print("RMSE on our test set:", rmse)

# COMMAND ----------

display(predictions.select("HOURLY_CONSUMPTION_MW", "prediction"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Metrics/Parameters to be logged

# COMMAND ----------

# Metrcis
mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

#Hyperparameters

# Get the index of the best model
best_model_index = cvModel.avgMetrics.index(min(cvModel.avgMetrics))

# Get the parameters of the best model
best_model_params = cvModel.getEstimatorParamMaps()[best_model_index]

# Store the parameters in a dictionary
hyperparameters = {}

# Loop over the parameters and store them in the dictionary
for param, value in best_model_params.items():
    hyperparameters[param.name] = value

#Model Training Time
training_time = end_time - start_time

#Model Training/Testing Data Size
training_size = train_df.count()
testing_size = test_df.count()

#Current Time
current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

#Description
description = "The logged model is an XGBoost regressor that has been trained to predict DAILY_CONSUMPTION_MW based on various input features. It performs well in accurately estimating energy consumption. The model takes into account important factors and patterns present in the data to make reliable predictions. It has been fine-tuned using cross-validation and optimized hyperparameters to ensure its effectiveness"

#Model Tags
tags = {
    "model_type": "XGBoost Regressor",
    "dataset": "Energy Consumption",
    "application": "Energy Management",
    "framework": "PySpark"
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register Model to MLflow

# COMMAND ----------

    with mlflow.start_run(nested=True) as run:
      
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
        # Search for all versions of the registered model
        versions = client.search_model_versions("name='pyspark_mlflow_model'")
        # Sort the versions by creation timestamp in descending order
        sorted_versions = sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)
        # Get the latest version
        latest_version = sorted_versions[0]
        # Access the version number
        model_version = latest_version.version

        # Save your data to a new DBFS directory for each run
        data_path = f"dbfs:/FileStore/Data_Versioning/data_model_v{model_version}.parquet"
        concatenated_df.write.format("parquet").save(data_path)
 
        # Log the DBFS path as an artifact
        with open("data_path.txt", "w") as f:
            f.write(data_path)
        mlflow.log_artifact("data_path.txt")

# COMMAND ----------


