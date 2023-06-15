# Databricks notebook source
# MAGIC %md 
# MAGIC ## Retrieve Variables from the Calling Notebook

# COMMAND ----------

concatenated_df = spark.table("concatenated_df_view")
train_df = spark.table("train_df_view")
test_df_view = spark.table("test_df_view")

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
# MAGIC * `SparkXGBRegressor`: Uses the SparkXGBRegressor estimator to learn how to predict rental counts from the feature vectors.
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
  .addGrid(xgb_regressor.max_depth, [4, 5, 6, 7])\
  .addGrid(xgb_regressor.n_estimators, [50, 75, 100])\
  .addGrid(xgb_regressor.learning_rate, [0.1, 0.01])\
  .build()

# Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
evaluator = RegressionEvaluator(metricName="rmse",
                                labelCol=xgb_regressor.getLabelCol(),
                                predictionCol=xgb_regressor.getPredictionCol())
 
# Declare the CrossValidator, which performs the model tuning.
cv = CrossValidator(estimator=xgb_regressor, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

# MAGIC %md
# MAGIC * Create the pipeline

# COMMAND ----------

pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])

# COMMAND ----------

# MAGIC %md
# MAGIC Train the pipeline:
# MAGIC
# MAGIC Now that we have set up the workflow, we can train the pipeline with a single call.
# MAGIC When we call fit(), the pipeline runs feature processing, model tuning, and training and returns a fitted pipeline with the best model it found. This step takes several minutes.

# COMMAND ----------

start_time = time.time()
pipelineModel = pipeline.fit(train_df)
end_time = time.time()
xgb_model = pipelineModel.stages[-1].bestModel # Retrieve best model

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

predictions = pipelineModel.transform(test_df)

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

# Get the stages of the pipeline
stages = pipelineModel.stages

# Iterate over the stages and retrieve their parameter values
param_dict = {}
for stage in stages:
    stage_params = stage.extractParamMap()
    param_dict.update({param.name: stage_params[param] for param in stage_params.keys()})

hyperparameters = {
    param.name: param_dict['estimatorParamMaps'][0][param]
    for param in param_dict['estimatorParamMaps'][0].keys()
}

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
