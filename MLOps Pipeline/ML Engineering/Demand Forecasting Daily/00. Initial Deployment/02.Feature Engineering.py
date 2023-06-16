# Databricks notebook source
# MAGIC %run "/Repos/filippos.priovolos01@gmail.com/mlops-energy-forecast-thesis/MLOps Thesis Pipeline/Workflow Config/Initial Deployment"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

input_table_name = 'final_consumption_countries_hourly'
output_table_name = 'hourly_forecasting_features'
save_into_feature_store = True
delete_fs = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Dataset

# COMMAND ----------

table = spark.table(f'{db}.{input_table_name}')
table.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## One-Hot-Encoding of Categorical Columns (Countries)

# COMMAND ----------

# MAGIC %md
# MAGIC The create_country_features function adds binary country-specific features to a DataFrame. It iterates over distinct country values in the 'COUNTRY' column, creates a new column for each country, and assigns a value of 1 if the row corresponds to that country, and 0 otherwise. The updated DataFrame with the added features is returned and displayed using the display function.

# COMMAND ----------

from pyspark.sql import functions as F
def create_country_features(df):
    # for col in df.columns: 
    countries = [row['COUNTRY'] for row in df.select('COUNTRY').distinct().collect()]
    countries.sort()
    for country in countries: 
        df = df.withColumn("{}".format(country), F.when((df['COUNTRY'] == country), 1).otherwise(0))
    return df

features = create_country_features(table)
display(features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Features for Model Training

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Preprocessing and Sorting
# MAGIC     * Convert the 'DATETIME' column to datetime format.
# MAGIC     * Set this converted column as the index of the DataFrame.
# MAGIC     * Sort the DataFrame by 'COUNTRY' and 'DATETIME' columns.
# MAGIC 2. Extracting Date Features
# MAGIC     * Create new columns for various date components: 'HOUR', 'DAY_OF_WEEK', 'MONTH', 'QUARTER', 'YEAR', 'DAY_OF_YEAR', 'DAY_OF_MONTH', and 'WEEK_OF_YEAR'.
# MAGIC 3. Calculate Rolling Statistics & Lagged Features
# MAGIC     * For each country, calculate rolling mean, rolling standard deviation, and rolling sum of the 'HOURLY_CONSUMPTION_MW' over specific windows (24 hours and 7 days).
# MAGIC     * Create lagged features for 'HOURLY_CONSUMPTION_MW' such as the consumption of the previous day, previous week, and previous month.
# MAGIC 4. Handling Null Values
# MAGIC     * Backward fill the null values generated due to shifting (lagged features) and rolling operations.
# MAGIC 5. Drop Original Consumption Column
# MAGIC     * Drop the 'HOURLY_CONSUMPTION_MW' column as we have generated statistical features from it.
# MAGIC 6. Return the Modified DataFrame
# MAGIC     * The function returns the DataFrame with the newly created features.

# COMMAND ----------

def create_features(df):
    """
    Creates time series features from datetime index in order to save them in Features Store
    """
    # Convert 'DATETIME' column to datetime format and set it as the index
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.set_index('DATETIME', inplace=True)
    df.sort_values(['COUNTRY', 'DATETIME'], inplace=True)

    # Extract date-related features
    df['HOUR'] = df.index.hour  
    df['DAY_OF_WEEK'] = df.index.dayofweek
    df['MONTH'] = df.index.month
    df['QUARTER'] = df.index.quarter
    df['YEAR'] = df.index.year
    df['DAY_OF_YEAR'] = df.index.dayofyear
    df['DAY_OF_MONTH'] = df.index.day
    df['WEEK_OF_YEAR'] = df.index.isocalendar().week

    # Calculate rolling statistics and lagged features for each country
    for country in df['COUNTRY'].unique():
        df.loc[df['COUNTRY'] == country, 'ROLLING_MEAN_24H'] = df.loc[df['COUNTRY'] == country, 'HOURLY_CONSUMPTION_MW'].rolling(window=24).mean()
        df.loc[df['COUNTRY'] == country, 'ROLLING_STD_24H'] = df.loc[df['COUNTRY'] == country, 'HOURLY_CONSUMPTION_MW'].rolling(window=24).std()
        df.loc[df['COUNTRY'] == country, 'ROLLING_SUM_7D'] = df.loc[df['COUNTRY'] == country, 'HOURLY_CONSUMPTION_MW'].rolling(window=7 * 24, min_periods=1).sum()
        df.loc[df['COUNTRY'] == country, 'PREV_DAY_CONSUMPTION'] = df.loc[df['COUNTRY'] == country, 'HOURLY_CONSUMPTION_MW'].shift(24)
        df.loc[df['COUNTRY'] == country, 'PREV_WEEK_CONSUMPTION'] = df.loc[df['COUNTRY'] == country, 'HOURLY_CONSUMPTION_MW'].shift(24 * 7)
        df.loc[df['COUNTRY'] == country, 'PREVIOUS_MONTH_CONSUMPTION'] = df.loc[df['COUNTRY'] == country, 'HOURLY_CONSUMPTION_MW'].shift(24*30)

    # Backward fill only the rows that end up as null after shifting
    df['PREV_DAY_CONSUMPTION'] = df['PREV_DAY_CONSUMPTION'].fillna(method='bfill')
    df['PREV_WEEK_CONSUMPTION'] = df['PREV_WEEK_CONSUMPTION'].fillna(method='bfill')
    df['PREVIOUS_MONTH_CONSUMPTION'] = df['PREVIOUS_MONTH_CONSUMPTION'].fillna(method='bfill')
    df['ROLLING_MEAN_24H'] = df['ROLLING_MEAN_24H'].fillna(method='bfill')
    df['ROLLING_STD_24H'] = df['ROLLING_STD_24H'].fillna(method='bfill')

    df = df.drop('HOURLY_CONSUMPTION_MW',axis=1)
    
    return df


# COMMAND ----------

# Convert features df from spark to pandas and call the create_features() 
features = create_features(features.toPandas())
features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Primary Key

# COMMAND ----------

# MAGIC %md
# MAGIC By concatenating the 'COUNTRY' and 'DATETIME' values with an underscore ('_'), the code aims to create a composite key that uniquely identifies each row in the DataFrame

# COMMAND ----------

features.reset_index(inplace=True)
features['CONSUMPTION_ID'] = features.COUNTRY + '_' + features.DATETIME.astype(str)
features.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save features dataset into Feature Store

# COMMAND ----------

if save_into_feature_store:

    features.drop(['COUNTRY'], axis=1, inplace=True)

    features = spark.createDataFrame(features)

    fs = feature_store.FeatureStoreClient()

    fs.create_table(
        name=f'{db}.{output_table_name}',
        primary_keys=['CONSUMPTION_ID'],
        timestamp_keys='DATETIME',
        df=features
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delete Features Store

# COMMAND ----------

if delete_fs:
    from databricks.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    fs.drop_table(name='df_dev.hourly_forecasting_features')
    print("Feature Store was succesfuly deleted")

# COMMAND ----------


