# Databricks notebook source
# MAGIC %run "/Repos/filippos.priovolos01@gmail.com/mlops-energy-forecast-thesis/MLOps Thesis Pipeline/Workflow Config/Initial Deployment"

# COMMAND ----------

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler, MinMaxScaler
from pyspark.sql.functions import rand
from pyspark.mllib.evaluation import MulticlassMetrics
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
# initialize SparkSession

# COMMAND ----------

# Load Consumption Region Table
consumption_countries_hourly = spark.table('df_dev.final_consumption_countries_hourly')

# Update the key column construction in the PySpark code
consumption_countries_hourly = consumption_countries_hourly.withColumn('CONSUMPTION_ID', concat(col('COUNTRY'), lit('_'), col('DATETIME').cast('string')))

# Split the labels into training and test
train_labels = consumption_countries_hourly.filter((col('DATETIME') >= train_start) & (col('DATETIME') <= train_end))
test_labels = consumption_countries_hourly.filter((col('DATETIME') > train_end) & (col('DATETIME') <= test_end))
val_labels = consumption_countries_hourly.filter((col('DATETIME') > test_end) & (col('DATETIME') <= validation_end))

# Select the required columns
train_labels = train_labels.select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")
test_labels = test_labels.select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")
val_labels = val_labels.select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")

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
val_labels = val_labels.withColumn('DATETIME', col('DATETIME').cast(TimestampType()))

# Load the data for the training set
training_set, train_df = load_data(f'{db}.hourly_forecasting_features', train_labels, 'CONSUMPTION_ID', 'DATETIME')

# Load the data for the test set
_, test_df = load_data(f'{db}.hourly_forecasting_features', test_labels, 'CONSUMPTION_ID', 'DATETIME')

# Load the data for the validation set
_, val_df = load_data(f'{db}.hourly_forecasting_features', val_labels, 'CONSUMPTION_ID', 'DATETIME')


# COMMAND ----------

display(train_df)

# COMMAND ----------

train_df

# COMMAND ----------

# MAGIC %md
# MAGIC Define the features and label columns: We first need to specify which columns in the dataframe are features and which column is the label

# COMMAND ----------

featuresCols = train_df.columns[:-1]
target_names = [train_df.columns[-1]]

# COMMAND ----------

# MAGIC %md
# MAGIC Create VectorAssembler and MinMaxScaler objects: VectorAssembler combines the specified feature columns into a single vector column. MinMaxScaler normalizes these feature vectors to be in the range [0, 1].

# COMMAND ----------

# Assuming you have loaded your dataset into a DataFrame called 'data'
# Assuming the label column name is 'label'

# Extract the labels from the DataFrame
labels = train_df.select('HOURLY_CONSUMPTION_MW').rdd.flatMap(lambda x: x).collect()

# Find the minimum and maximum values of the labels
min_label = min(labels)
max_label = max(labels)


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, MinMaxScaler

vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol='assembled_features')
# Create a separate MinMaxScaler for features
scaler_features = MinMaxScaler(min=0.0, max=1.0, inputCol='assembled_features', outputCol='features')
vectorAssemblerLabel = VectorAssembler(inputCols=target_names, outputCol='label')
scaler_label = MinMaxScaler(min=0.0, max=1.0, inputCol='label', outputCol='scaled_label')

# COMMAND ----------

# MAGIC %md
# MAGIC Create a pipeline of transformations: The pipeline includes vector assembly and scaling stages.

# COMMAND ----------

from pyspark.ml import Pipeline

stages = [vectorAssembler,scaler_features,vectorAssemblerLabel,scaler_label]
pipeline = Pipeline(stages=stages)

# COMMAND ----------

# MAGIC %md
# MAGIC Apply the transformations to each DataFrame:

# COMMAND ----------

# Fit the pipeline to the training data
pipeline_model = pipeline.fit(train_df)

# Transform each DataFrame
train_transformed = pipeline_model.transform(train_df)
val_transformed = pipeline_model.transform(val_df)
test_transformed = pipeline_model.transform(test_df)


# COMMAND ----------

# Save the fitted pipeline for later use
pipeline_model.write().overwrite().save("/dbfs/FileStore/Fitted_Pipeline")

# COMMAND ----------

train_transformed.show(truncate=False, vertical=True, n=1)

# COMMAND ----------

# MAGIC %md
# MAGIC Convert to Pandas DataFrames

# COMMAND ----------

# Convert to pandas
train_pd = train_transformed.toPandas()
val_pd = val_transformed.toPandas()
test_pd = test_transformed.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Exctract features and labels

# COMMAND ----------

train_pd

# COMMAND ----------

    # Extract features and labels
    import numpy as np
    X_train = np.array(train_pd['features'].to_list())
    y_train = np.array(train_pd['scaled_label'].to_list())

    X_val = np.array(val_pd['features'].to_list())
    y_val = np.array(val_pd['scaled_label'].to_list())

    X_test = np.array(test_pd['features'].to_list())
    y_test = np.array(test_pd['scaled_label'].to_list())

# COMMAND ----------

y_train[0:1]

# COMMAND ----------

X_train[0:1]

# COMMAND ----------

# MAGIC %md
# MAGIC Reshape data for LSTM:

# COMMAND ----------

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# COMMAND ----------

y_train.shape

# COMMAND ----------

X_train.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Define and compile LSTM model:

# COMMAND ----------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import os

experiment_log_dir = f"/dbfs/{user}/tb"
checkpoint_path = f"/dbfs/{user}/keras_checkpoint_weights_day_ckpt"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

epochs = 100
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights = True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_log_dir)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)

# COMMAND ----------

model = Sequential()
model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(64, activation='tanh', return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',metrics=['mae'])

# COMMAND ----------

start_time = time.time()
history = model.fit(X_train, y_train, validation_data = (X_val , y_val), epochs=epochs, callbacks=[tensorboard_callback, model_checkpoint, early_stopping],verbose=1)
end_time = time.time()

# COMMAND ----------

# Validate the model
val_loss = model.evaluate(X_val, y_val)

# Test the model
test_loss = model.evaluate(X_test, y_test)

# COMMAND ----------

y_pred = model.predict(X_test)

# COMMAND ----------

import numpy as np

# Assuming you have loaded the predicted scaled labels into a variable called 'y_pred'

# Define the minimum and maximum values for the labels
min_label = 201.0
max_label = 324310.0

# Compute the scaled label range
scaled_label_range = max_label - min_label

# Perform inverse scaling on the predicted labels
y_pred_original = (y_pred * scaled_label_range) + min_label
y_test_original = (y_test * scaled_label_range) + min_label


# COMMAND ----------

y_pred.flatten()

# COMMAND ----------

# create a dataframe
compare_df = pd.DataFrame({'Actual': y_test_original.flatten(), 'Predicted': y_pred_original.flatten()})
compare_df

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Generate predictions
y_pred = model.predict(X_test)

# Flatten y_test and y_pred to 1D arrays (this may not be necessary depending on the shape of your arrays)
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
print("Root Mean Square Error: ", rmse)

# Compute MAE
mae = mean_absolute_error(y_test_flat, y_pred_flat)
print("Mean Absolute Error: ", mae)

# Compute R2 Score
r2 = r2_score(y_test_flat, y_pred_flat)
print("R-squared: ", r2)


# COMMAND ----------

# Since your output might be multi-dimensional, you might want to select a specific dimension for plotting
# Here's an example for the first dimension
dim = 0
y_test_dim = y_test[:, dim]
y_test_pred_dim = y_pred[:, dim]

# Create a new figure
plt.figure(figsize=(10, 6))

# Plot the actual values
plt.plot(y_test_dim, 'b-', label='actual')

# Plot the predicted values
plt.plot(y_test_pred_dim, 'r-', label='predicted')

# Create the legend
plt.legend()

# Show the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Metrics/Parameters to be logged

# COMMAND ----------

# Metrcis
mse = mean_squared_error(y_test_flat, y_pred_flat)
mae = mean_absolute_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mse)  # or mse**(0.5)  
r2 = r2_score(y_test_flat, y_pred_flat)

#Hyperparameters
hyperparameters = {
    "epochs": epochs,
    "batch_size": 21088, # if you defined a batch size
    "early_stopping_patience": early_stopping.patience,
    "optimizer": str(type(model.optimizer).__name__),
    "loss_function": model.loss.__name__ if callable(model.loss) else str(model.loss),
    "first_layer_units": model.layers[0].units,
    "first_layer_activation": model.layers[0].activation.__name__ if callable(model.layers[0].activation) else str(model.layers[0].activation),
    "second_layer_units": model.layers[1].units,
    "second_layer_activation": model.layers[1].activation.__name__ if callable(model.layers[1].activation) else str(model.layers[1].activation),
    "min_label" : min_label,
    "max_label" : max_label,
    "training_size":len(X_train),
    "training_range": {
        'start': '2015-01-01',
        'end': '2021-12-31'
    },
    "testing_size":len(X_test),
    "testing_range":{
        'start':'2022-01-01',
        'end':'2022-09-30'
    },
    "validation_size" : len(X_val),
    "validation_range":{
        'start':'2022-10-01',
        'end':'2023-01-01'
    }

}

#Model Training Time
training_time = end_time - start_time

#Current Time
current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

# Description
description = "The logged model is an LSTM-based recurrent neural network that has been trained to predict DAILY_CONSUMPTION_MW based on various input features. It leverages the temporal dependencies present in the data, making it suitable for energy consumption prediction. The model has been fine-tuned with the optimal number of epochs and other hyperparameters to ensure its effectiveness."

# Model Tags
tags = {
    "model_type": "RNN LSTM",
    "dataset": "Energy Consumption",
    "application": "Energy Management",
    "framework": "TensorFlow/Keras"
}


# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model to Mlflow

# COMMAND ----------

from mlflow.models.signature import infer_signature
signature = infer_signature(X_train, model.predict(X_train))

# COMMAND ----------

model_name = 'lstm_model'

# COMMAND ----------

    with mlflow.start_run(nested=True) as run:
      
        # Log the model input schema
        schema = {"input_schema": list(train_df.columns[:-1]),"output_schema":train_df.columns[-1]}
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
        metrics = {"R2": r2, "MSE": mse, "RMSE": rmse, 'MAE':mae}
        mlflow.log_dict(metrics, "metrics.json")

        # Log the model description as artifact
        mlflow.log_text(description, "description.txt")
        
        # Log the current timestamp as the code version
        mlflow.log_param("code_version", current_time)

        # Log all hyperparameters
        mlflow.log_params(hyperparameters)

        fs.log_model(
                    model=model,
                    artifact_path=f"{model_name}_artifact_path",
                    flavor=mlflow.tensorflow,
                    training_set = training_set ,
                    registered_model_name = model_name
                )

# COMMAND ----------

    with mlflow.start_run(nested=True) as run:
      
        # Log the model input schema
        schema = {"input_schema": list(train_df.columns[:-1]),"output_schema":train_df.columns[-1]}
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
        #mlflow.log_metric("Training Time(sec)", training_time)

        # Log evaluation metrics as artifact
        metrics = {"R2": r2, "MSE": mse, "RMSE": rmse, 'MAE':mae,'Training Time(sec)':training_time}
        mlflow.log_dict(metrics, "metrics.json")

        # Log the model description as artifact
        mlflow.log_text(description, "description.txt")
        
        # Log the current timestamp as the code version
        mlflow.log_param("code_version", current_time)

        # Log all hyperparameters
        mlflow.log_params(hyperparameters)

        # Log the model with its signature
        mlflow.keras.log_model(model,artifact_path="model", signature=signature)

        # Register the model with its signature
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="lstm_model")

        # Get the latest model version(The one that we now registered)
        client = MlflowClient()
        # Search for all versions of the registered model
        versions = client.search_model_versions("name='lstm_model'")
        # Sort the versions by creation timestamp in descending order
        sorted_versions = sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)
        # Get the latest version
        latest_version = sorted_versions[0]
        # Access the version number
        model_version = latest_version.version

        # Save your data to a new DBFS directory for each run
        data_path = f"dbfs:/FileStore/Data_Versioning/data_model_v{model_version}.parquet"
        train_df.write.format("parquet").save(data_path)
 
        # Log the DBFS path as an artifact
        with open("data_path.txt", "w") as f:
            f.write(data_path)
        mlflow.log_artifact("data_path.txt")

# COMMAND ----------


