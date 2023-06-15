# Databricks notebook source
# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, split, to_timestamp, date_format

# COMMAND ----------

# Use df_landing database
spark.sql('USE df_landing') 
# create a Spark session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Retrieve a list of all tables in the current database
tables = spark.sql('SHOW TABLES') \
    .select('tableName') \
    .rdd.flatMap(lambda x: x) \
    .collect()

print(tables)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load & Aggregate Data from Database

# COMMAND ----------

# MAGIC %md
# MAGIC The load_data function in PySpark takes a table name as an input, and performs the following steps:
# MAGIC
# MAGIC 1. Load Data: Reads data from the specified table into a DataFrame.
# MAGIC 2. Split Datetime: Splits the 'datetime' string into start and end times, and assigns the start time to a new column 'start_time'.
# MAGIC 3. Convert Datetime: Transforms the 'start_time' string into a timestamp format.
# MAGIC 4. Extract Hourly Time: Reduces the 'start_time' to an hourly format, discarding minute and second information.
# MAGIC 5. Extract Country Name: Derives the country name from the table name (assumed to be the first part of the table name before an underscore).
# MAGIC 6. Add Country Column: Adds a new column 'country' to the DataFrame, populated with the extracted country name.
# MAGIC 7. Return DataFrame: Returns the modified DataFrame.
# MAGIC
# MAGIC This function prepares the loaded data for further analysis by transforming the timestamp into an hourly format and adding a country identifier.

# COMMAND ----------

# function to load data from a table and add a country column
def load_data(table_name):
    df = spark.read.table(table_name)

    # split the datetime string into start and end times
    split_col = split(df['datetime'], ' - ')
    df = df.withColumn('start_time', split_col.getItem(0))

    # convert the start time into timestamp format
    datetime_format = "dd.MM.yyyy HH:mm"
    df = df.withColumn('start_time', to_timestamp(df['start_time'], datetime_format))

    # floor the start_time to the hour
    #df = df.withColumn('start_time', date_format(df['start_time'], 'yyyy-MM-dd HH:00:00').cast('timestamp'))

    # get the country name from the table name
    country = table_name.split("_")[0]

    # add the country column
    df = df.withColumn("country", lit(country))

    # sort the values based on start_time in ascending order
    df = df.sort("start_time")

    return df


# COMMAND ----------

# MAGIC %md
# MAGIC ## Save data in each table

# COMMAND ----------

# dictionary to store dataframes
df_dict = {}

# load data from each table
for table in tables:
    df_dict[table.split('_')[0]] = load_data(table)

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Sorts the DataFrame by the 'start_time' column.
# MAGIC 2. Replaces null values in the 'Actual_MW' column with the previous non-null value using forward fill. This is achieved by applying the last() function with the ignorenulls=True argument over a window specification.
# MAGIC 3. Replaces invalid values (0 or less) in the 'Actual_MW' column with the previous non-invalid value using forward fill. This is done by using the when() function to check if the value is less than or equal to 0, and if so, replaces it with the previous non-invalid value from the window.
# MAGIC 4. Updates the DataFrame in the dictionary with the modified DataFrame.
# MAGIC The code ensures that null and invalid values are replaced with appropriate values using forward fill, maintaining the ordering of the data by 'start_time' for each country DataFrame.

# COMMAND ----------

from pyspark.sql.functions import col, when, last
from pyspark.sql.window import Window

# Iterate over each country DataFrame in the dictionary
for country, df_country in df_dict.items():
    # Sort the DataFrame by 'start_time'
    df_country = df_country.orderBy('start_time')

    # Replace invalid values (0 or less) with null
    df_country = df_country.withColumn('Actual_MW', when(col('Actual_MW') <= 0, None).otherwise(col('Actual_MW')))
    
    # Replace null values with previous non-null values using forward fill
    window_spec = Window.partitionBy('country').orderBy('start_time').rowsBetween(Window.unboundedPreceding, 0)
    df_country = df_country.withColumn('Actual_MW', last('Actual_MW', ignorenulls=True).over(window_spec))

    # Update the DataFrame in the dictionary
    df_dict[country] = df_country


# COMMAND ----------

# MAGIC %md
# MAGIC * The get_hourly_query function is defined to create a SQL query for each table (country). This query selects the start_time, Actual_MW (renamed as HOURLY_CONSUMPTION_MW), and the table name (representing the country).
# MAGIC
# MAGIC

# COMMAND ----------

# function to generate SQL query for a given table
def get_hourly_query(table_name):
    return f"""
    SELECT 
        start_time AS DATETIME, 
        Actual_MW AS HOURLY_CONSUMPTION_MW, 
        '{table_name}' AS COUNTRY
    FROM {table_name}
    """

# COMMAND ----------

# register each DataFrame as a temporary view in Spark
for table_name, df in df_dict.items():
    df.createOrReplaceTempView(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC * The final_hourly_query is created by applying the get_hourly_query function to each country's DataFrame in the dictionary, joining the resulting SQL queries with UNION ALL. The UNION ALL SQL operation combines the rows from these separate queries into a single set of results.

# COMMAND ----------

final_hourly_query = ' UNION ALL '.join([get_hourly_query(country) for country in df_dict.keys()])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC * The final_hourly_query is then executed using spark.sql(). This command runs the combined SQL query and creates a DataFrame.
# MAGIC
# MAGIC * The .dropDuplicates(['DATETIME', 'COUNTRY']) operation removes any duplicate rows from the DataFrame based on the DATETIME and COUNTRY columns.
# MAGIC
# MAGIC * The .createOrReplaceTempView('final_hourly_df') operation creates a temporary view with the name 'final_hourly_df'. This is a named logical plan that is used as a stand-in for the DataFrame in Spark SQL queries.

# COMMAND ----------

spark.sql(final_hourly_query) \
    .dropDuplicates(['DATETIME', 'COUNTRY']) \
    .createOrReplaceTempView('final_hourly_df')

spark.sql("""
SELECT * FROM final_hourly_df
ORDER BY DATETIME,COUNTRY
""").createOrReplaceTempView('final_hourly_df_ordered')

# COMMAND ----------

database= 'df_dev'

# COMMAND ----------

# MAGIC %md
# MAGIC * The MERGE INTO statement is a SQL command that updates the consumption_countries_hourly table in the database. If a record (based on DATETIME and COUNTRY) already exists in the table, it updates the existing record with the new data. If a record does not exist, it inserts a new record with the data.
# MAGIC
# MAGIC

# COMMAND ----------

spark.sql(f"""
MERGE INTO {database}.consumption_countries_hourly A
USING final_hourly_df B
ON A.DATETIME = B.DATETIME AND A.COUNTRY = B.COUNTRY
WHEN MATCHED THEN
  UPDATE SET
    DATETIME = B.DATETIME,
    HOURLY_CONSUMPTION_MW = B.HOURLY_CONSUMPTION_MW,
    COUNTRY = B.COUNTRY
WHEN NOT MATCHED
  THEN INSERT (
    DATETIME,
    HOURLY_CONSUMPTION_MW, 
    COUNTRY
) VALUES (  
    B.DATETIME,
    B.HOURLY_CONSUMPTION_MW, 
    B.COUNTRY
)
""")

# COMMAND ----------


