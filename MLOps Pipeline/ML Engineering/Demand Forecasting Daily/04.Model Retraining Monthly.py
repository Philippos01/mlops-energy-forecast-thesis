# Databricks notebook source
# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

# MAGIC %run "/Repos/filippos.priovolos01@gmail.com/mlops-energy-forecast-thesis/MLOps Thesis Pipeline/Workflow Config/Initial Deployment"
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC * The train_end string is converted to a datetime object using the datetime.strptime() function, with the format '%Y-%m-%d'.
# MAGIC * The relativedelta() function is used to add 3 months to the date_object, and the result is converted back to a string in the format '%Y-%m-%d'. This new date is stored in the variable new_train_end.
# MAGIC * The test_start string is converted to a datetime object using the datetime.strptime() function, with the format '%Y-%m-%d'.
# MAGIC * Again, the relativedelta() function is used to add 3 months to the date_object, and the result is converted back to a string in the format '%Y-%m-%d'. This new date is stored in the variable new_test_start.
# MAGIC * The new_test_end variable is set to the string '2023-01-01'.
# MAGIC
# MAGIC The purpose of this code is to define the new dates for retraining the model after 3 months of the initial deployment with the new data

# COMMAND ----------

# Convert the string to a datetime object
date_object = datetime.strptime(train_end, '%Y-%m-%d')
new_train_end = (date_object + relativedelta(months=3)).strftime('%Y-%m-%d')
date_object = datetime.strptime(test_start, '%Y-%m-%d')
new_test_start = (date_object + relativedelta(months=3)).strftime('%Y-%m-%d')
new_test_end = '2023-01-01'
# Convert start and end dates to datetime objects
# Convert new dates to datetime
new_train_end = pd.to_datetime(new_train_end)
new_test_start = pd.to_datetime(new_test_start)
new_test_end = pd.to_datetime(new_test_end)

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
train_labels = consumption_countries_hourly.loc[(consumption_countries_hourly.DATETIME >= train_start) & (consumption_countries_hourly.DATETIME <= new_train_end)]
test_labels = consumption_countries_hourly.loc[(consumption_countries_hourly.DATETIME > new_test_start) & (consumption_countries_hourly.DATETIME <= new_test_end)]
#val_labels = consumption_countries_hourly.loc[(consumption_countries_hourly.DATETIME > test_end) & (consumption_countries_hourly.DATETIME <= validation_end)]

# Transforms to Spark DataFranes
train_labels = spark.createDataFrame(train_labels).select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")
test_labels = spark.createDataFrame(test_labels).select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")
#val_labels = spark.createDataFrame(val_labels).select("CONSUMPTION_ID", "DATETIME", "HOURLY_CONSUMPTION_MW")

# COMMAND ----------

display(test_labels)

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
# MAGIC ## Data Drift Test

# COMMAND ----------

# MAGIC %md
# MAGIC * Convert Date: Convert train_end to datetime format and store it in old_train_end.
# MAGIC * Extract Date Columns: Create a new DataFrame date_df by extracting the 'YEAR', 'MONTH', and 'DAY_OF_MONTH' columns from X_train.
# MAGIC * Combine Date Parts: Rename 'DAY_OF_MONTH' to 'day', combine the year, month, and day columns into a single datetime column named 'date', and remove the separate year, month, and day columns from date_df.
# MAGIC * Merge DataFrames: Concatenate the new date_df DataFrame with the original X_train, excluding its original date columns, and store the result in df_with_date.
# MAGIC * Filter New Training Data: Select rows in df_with_date where the 'date' is between old_train_end and new_train_end, and store them in new_training_data.
# MAGIC * Filter Old Training Data: Select rows in df_with_date where the 'date' is before or equal to old_train_end, and store them in old_training_data.
# MAGIC * Sample Old Training Data: Take a random sample of old_training_data with the same number of rows as new_training_data, and store it in old_training_data_sample.
# MAGIC
# MAGIC This code prepares datasets for analyzing data drift by filtering and sampling old and new data based on dates.

# COMMAND ----------

old_train_end = pd.to_datetime(train_end)

# Create a new DataFrame with year, month, and day columns
date_df = X_train[['YEAR', 'MONTH', 'DAY_OF_MONTH']]
date_df = date_df.rename(columns={'DAY_OF_MONTH': 'day'})

# Convert date columns to datetime format and add as a new column
date_df['date'] = pd.to_datetime(date_df[['YEAR', 'MONTH', 'day']])
date_df = date_df.drop(['YEAR', 'MONTH', 'day'], axis=1)

# Combine the date DataFrame with the original DataFrame
df_with_date = pd.concat([date_df, X_train.drop(['YEAR', 'MONTH', 'DAY_OF_MONTH'], axis=1)], axis=1)

# Extract the data between old_train_end and new_train_end for Greece with the same day of the week
new_training_data = df_with_date[(df_with_date['date'] > old_train_end) & (df_with_date['date'] <= new_train_end) ]

# Extract old training data for Greece with the same day of the week
old_training_data = df_with_date[(df_with_date['date'] <= old_train_end) ]

# Sample equal amounts of old training data for Greece
sample_size = len(new_training_data)
old_training_data_sample = old_training_data.sample(n=sample_size, random_state=123)

# COMMAND ----------

old_training_data_sample

# COMMAND ----------

new_training_data

# COMMAND ----------

# MAGIC %md
# MAGIC This code segment is for analyzing data drift by comparing the distribution of various features in old and new datasets. Here's a breakdown in bullets:
# MAGIC
# MAGIC * Label Data Groups: Add a column data_group to new_training_data and old_training_data_sample to label them as 'new' and 'old', respectively.
# MAGIC * Concatenate Data: Concatenate new_training_data and old_training_data_sample into a single DataFrame all_data.
# MAGIC * Define Feature Lists: Create lists of categorical features (categorical_features) and continuous features (continuous_features) that need to be analyzed for data drift.
# MAGIC * Set Tolerance Threshold: Define a tolerance_threshold for the effect size, which will be used to determine if there is significant drift in the distribution of continuous features.
# MAGIC * Analyze Categorical Features: For each categorical feature, create a contingency table between the feature and the data_group column. Perform a Chi-squared test to determine if the distribution of categories has changed. Adjust the alpha value for multiple testing and print the result.
# MAGIC * Analyze Continuous Features: For each continuous feature, perform a Kolmogorov-Smirnov test to compare the distributions in the old and new data. Calculate the effect size and compare it to the tolerance_threshold. Print the result.
# MAGIC
# MAGIC This code essentially uses statistical tests to determine if there has been a significant change (drift) in the distributions of features in the dataset over time.

# COMMAND ----------

# Adding a data_group column to identify old and new data
new_training_data['data_group'] = 'new'
old_training_data_sample['data_group'] = 'old'

# Concatenate the new and old data
all_data = pd.concat([new_training_data, old_training_data_sample])

# List of categorical features
categorical_features = ['belgium', 'denmark', 'france', 'germany', 'greece', 'italy', 'luxembourg','spain', 'sweden', 'switzerland', 'netherlands']

# List of continuous features
continuous_features = ['DAY_OF_WEEK', 'QUARTER', 'HOUR' , 'PREV_DAY_CONSUMPTION', 'PREV_WEEK_CONSUMPTION', 'PREVIOUS_MONTH_CONSUMPTION', 'ROLLING_MEAN_24H', 'ROLLING_STD_24H', 'ROLLING_SUM_7D' ]

# Define the tolerance threshold for the effect size
tolerance_threshold =0.001  # Adjust as needed

# Iterate over the categorical features
for feature in categorical_features:
    contingency_table = pd.crosstab(index=all_data[feature], columns=all_data['data_group'])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    adjusted_alpha = 0.05 / len(categorical_features)  # Adjust alpha for multiple testing
    if p_value < adjusted_alpha:
        print(f"The distribution of {feature} has drifted (p-value: {p_value:.3f}).")
    else:
        print(f"The distribution of {feature} has not drifted (p-value: {p_value:.3f}).")

# Iterate over the continuous features
for feature in continuous_features:
    statistic, p_value = stats.ks_2samp(old_training_data_sample[feature], new_training_data[feature])
    effect_size = abs(statistic) / len(old_training_data_sample[feature])  # Calculate effect size
    if effect_size > tolerance_threshold:
        print(f"The distribution of {feature} has drifted (effect size: {effect_size:.3f}).")
    else:
        print(f"The distribution of {feature} has not drifted (effect size: {effect_size:.3f}).")



# COMMAND ----------

# MAGIC %md
# MAGIC ### Register Retrained Model to MLflow

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
# MAGIC * The retrained model is logged as an artifact using the feature store's log_model() function, and various parameters and information are logged for comparison and tracking purposes.
# MAGIC
# MAGIC The purpose of this function is to retrain and log the model to feature store

# COMMAND ----------

def train_model(X_train, X_test, y_train, y_test, training_set, fs, model_name, input_schema):
    experiment_id = experiment_id_retraining
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
        tags = {"model_type": "XGBoost", "dataset": "energy_consumption","Workflow Type": "Retraining"}
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
        description = "This is an XGBoost model retrained after 3 months to predict energy consumption of 11 European Countries in hourly basis."
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
            'end': new_train_end
        }
        testing_range = {
            'start': new_test_start,
            'end': new_test_end
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

# MAGIC %md
# MAGIC ## Model Staging

# COMMAND ----------

# MAGIC %md
# MAGIC * Initialize Client: Initialize the MLflow client for interacting with the tracking server.
# MAGIC * Fetch Model Version: Get the latest version of the registered machine learning model.
# MAGIC * Configure API Request: Construct the API endpoint URL, headers, and request body necessary for transitioning the model to the staging environment in Databricks.
# MAGIC * Send API Request: Make a POST request to the Databricks API to transition the model to the staging environment.
# MAGIC * Verify Response: Check the response status code to determine if the model transition was successful, and print an appropriate message.

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


