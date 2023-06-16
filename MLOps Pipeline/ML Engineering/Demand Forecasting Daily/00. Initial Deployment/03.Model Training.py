# Databricks notebook source
# MAGIC %md 
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %run "/Repos/filippos.priovolos01@gmail.com/mlops-energy-forecast-thesis/MLOps Thesis Pipeline/Workflow Config/Initial Deployment"

# COMMAND ----------

# MAGIC %md
# MAGIC ##Load Datasets

# COMMAND ----------

# MAGIC %md
# MAGIC * Load energy consumption data from a database into a Pandas DataFrame.
# MAGIC * Create a new column CONSUMPTION_ID by concatenating country codes with the date-time information.
# MAGIC * Convert the DATETIME column to a proper datetime data type for time-based operations.
# MAGIC * Split the data into two subsets: train_labels and test_labels, based on date-time ranges.
# MAGIC * Convert the subsets back into Spark DataFrames and select only the CONSUMPTION_ID, DATETIME, and HOURLY_CONSUMPTION_MW columns for further processing

# COMMAND ----------

# Load Consumption Region Table
consumption_countries_hourly = spark.table(f'{db}.final_consumption_countries_hourly').toPandas()
consumption_countries_hourly['CONSUMPTION_ID'] = consumption_countries_hourly.COUNTRY + '_' + consumption_countries_hourly.DATETIME.astype(str)
consumption_countries_hourly['DATETIME'] = pd.to_datetime(consumption_countries_hourly['DATETIME'])

# Split the labels into training and test
train_labels = consumption_countries_hourly.loc[(consumption_countries_hourly.DATETIME >= train_start) & (consumption_countries_hourly.DATETIME <= train_end)]
test_labels = consumption_countries_hourly.loc[(consumption_countries_hourly.DATETIME > test_start) & (consumption_countries_hourly.DATETIME <= test_end)]
#val_labels = consumption_countries_hourly.loc[(consumption_countries_hourly.DATETIME > test_end) & (consumption_countries_hourly.DATETIME <= validation_end)]

# Transforms to Spark DataFranes
train_labels = spark.createDataFrame(train_labels).select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")
test_labels = spark.createDataFrame(test_labels).select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")
#val_labels = spark.createDataFrame(val_labels).select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")

# COMMAND ----------

# MAGIC %md
# MAGIC * Define load_data function to create training sets by fetching features based on specified keys.
# MAGIC * Inside the function, initialize feature lookups and create a training set by matching keys from input data.
# MAGIC * Convert the training set to a Pandas DataFrame.
# MAGIC * Call the load_data function to create training and test sets, and store them in variables training_set, train_df, and test_df.

# COMMAND ----------

def load_data(table_name, labels, lookup_key, ts_lookup_key):
    # In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key, timestamp_lookup_key=ts_lookup_key)]

    # fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
    training_set = fs.create_training_set(labels, 
                                          model_feature_lookups, 
                                          label="HOURLY_CONSUMPTION_MW", 
                                          exclude_columns=["CONSUMPTION_ID", "DATETIME"])
    training_pd = training_set.load_df().toPandas()

    return training_set, training_pd

training_set, train_df = load_data(f'{db}.{feauture_store}', train_labels, 'CONSUMPTION_ID', "DATETIME")
_, test_df = load_data(f'{db}.{feauture_store}', test_labels, 'CONSUMPTION_ID', "DATETIME")
#_, val_df = load_data(f'{db}.{feauture_store}', val_labels, 'CONSUMPTION_ID', "DATE")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Split to Features/Labels

# COMMAND ----------

X_train = train_df.drop(columns=['HOURLY_CONSUMPTION_MW'])
y_train = train_df['HOURLY_CONSUMPTION_MW']
X_test = test_df.drop(columns=['HOURLY_CONSUMPTION_MW'])
y_test = test_df['HOURLY_CONSUMPTION_MW']

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create XGB Regressor and Register Model to Feature Store

# COMMAND ----------

# MAGIC %md
# MAGIC * The regressor is configured with the following hyperparameters:
# MAGIC * n_estimators: The number of trees in the ensemble (200).
# MAGIC * max_depth: The maximum depth of each tree (8).
# MAGIC * learning_rate: The step size shrinkage used in each boosting iteration (0.1).
# MAGIC * objective: The loss function to be optimized, using squared error for regression ('reg:squarederror').
# MAGIC * booster: The type of booster to use, specifically the gradient boosting tree ('gbtree').
# MAGIC * subsample: The fraction of training samples used for training each tree (0.8).
# MAGIC * colsample_bytree: The fraction of features used for training each tree (0.8).
# MAGIC * random_state: The random seed used for reproducibility (42).

# COMMAND ----------

def create_regressor():
    return XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        objective='reg:squarederror',
        booster='gbtree',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC * mse (Mean Squared Error): It measures the average squared difference between the true and predicted values.
# MAGIC * rmse (Root Mean Squared Error): It is the square root of the MSE, providing a more interpretable measure of the error.
# MAGIC * mae (Mean Absolute Error): It calculates the average absolute difference between the true and predicted values.
# MAGIC * r2 (R-squared): It indicates the proportion of the variance in the true values that is explained by the predicted values.
# MAGIC The calculated metrics are returned as a tuple (mse, rmse, mae, r2).

# COMMAND ----------

def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mae, r2

# COMMAND ----------


def log_metrics(mse, rmse, mae, r2, training_time):
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("Training Time(sec)", training_time)

# COMMAND ----------

# MAGIC %md
# MAGIC * It starts an MLflow run within a with statement to encapsulate the training and logging process.
# MAGIC * An XGBoost regressor is created using the create_regressor() function and trained on the training data using the fit() method. The regressor's predictions are then calculated using the testing data.
# MAGIC * The performance of the model is evaluated using the evaluate_model() function, which calculates metrics such as MSE, RMSE, MAE, and R2.
# MAGIC * The input schema, model hyperparameters, metrics, feature importances, and other relevant information are logged using MLflow's tracking capabilities.
# MAGIC * The trained model is logged as an artifact using the feature store's log_model() function, and various parameters and information are logged for comparison and tracking purposes.
# MAGIC
# MAGIC The purpose of this function is to train and log the model to feature store

# COMMAND ----------

def train_model(X_train, X_test, y_train, y_test, training_set, fs, model_name, input_schema):
    experiment_id = experiment_id_training
    experiment = mlflow.get_experiment(experiment_id)
    
    if experiment:
        experiment_name = experiment.name
        mlflow.set_experiment(experiment_name)
        print(f"Active experiment set to '{experiment_name}'")
    else:
        print(f"No experiment found with name '{experiment_name}'")
    
    with mlflow.start_run(nested=True) as run:
        # Create and train XGBoost regressor
        reg = create_regressor()
        start_time = time.time()
        reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=10, verbose=False)
        end_time = time.time()
        
        # Make predictions
        y_pred = reg.predict(X_test)
        
        # Evaluate the model
        mse, rmse, mae, r2 = evaluate_model(y_test, y_pred)
        
        # Log the model input schema
        input_schema = {"feature_names": list(X_train.columns)}
        mlflow.log_dict(input_schema, "input_schema.json")

        # Log some tags for the model
        tags = {"model_type": "XGBoost", "dataset": "energy_consumption","Workflow Type": "Initial Training"}
        mlflow.set_tags(tags)

        # Log some parameters for the model
        params = reg.get_params()
        mlflow.log_dict(params, "hyperparams.json")

        # Log metrics
        training_time = end_time - start_time
        log_metrics(mse, rmse, mae, r2, training_time)
        
        # Log the feature importances of the model
        importance = reg.get_booster().get_score(importance_type="gain")
        mlflow.log_dict(importance, "importance.json")
        # Log the model and its description as artifacts
        description = "This is an XGBoost model trained to predict energy consumption of 11 European Countries in hourly basis."
        mlflow.log_text(description, "description.txt")
        
        # Log the current timestamp as the code version
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        mlflow.log_param("code_version", current_time)

        # Log additional important parameters for comparison
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("learning_rate", params["learning_rate"])
        mlflow.log_param("subsample", params["subsample"])
        mlflow.log_param("colsample_bytree", params["colsample_bytree"])
        mlflow.log_param("random_state", params["random_state"])
        # Log the training data size
        training_size = len(X_train)
        testing_size = len(X_test)
        training_range = {
            'start': train_start,
            'end': train_end
        }
        testing_range = {
            'start': test_start,
            'end': test_end
        }
        mlflow.log_param("training_range", training_range)
        mlflow.log_param("testing_range", testing_range)
        mlflow.log_param("training_data_size", training_size)
        mlflow.log_param("testing_data_size", testing_size)

        # Log the model
        fs.log_model(
            model=reg,
            artifact_path=f"{model_name}_artifact_path",
            flavor=mlflow.xgboost,
            training_set=training_set,
            registered_model_name=model_name
        )
    
    return {"R2": r2, "MSE": mse, "RMSE": rmse, "MAE": mae, "Training Time(sec)": training_time}


metrics = train_model(X_train, X_test, y_train, y_test, training_set, fs, model_name, input_schema)


# COMMAND ----------


