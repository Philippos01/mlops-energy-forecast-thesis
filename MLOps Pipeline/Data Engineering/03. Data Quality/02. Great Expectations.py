# Databricks notebook source
# MAGIC %md 
# MAGIC Note: You can find the official documentation of great-expectations [here](https://docs.greatexpectations.io/docs/deployment_patterns/how_to_use_great_expectations_in_databricks/)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Great Expectations

# COMMAND ----------

# MAGIC %run "/Users/filippos.priovolos01@gmail.com/Workflow Config/Great Expecations Config"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Great Expectations

# COMMAND ----------

root_directory = "/dbfs/great_expectations/"
data_context_config = DataContextConfig(
    store_backend_defaults=FilesystemStoreBackendDefaults(
        root_directory=root_directory
    ),
)
context = get_context(project_config=data_context_config)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Prepare data

# COMMAND ----------

df = spark.read.format("delta") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .table("df_dev.final_consumption_countries_hourly")


# COMMAND ----------

display(df)

# COMMAND ----------

# Sort the DataFrame by country and datetime
df_sorted = df.orderBy("COUNTRY","DATETIME")
display(df_sorted)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect to the data

# COMMAND ----------

my_spark_datasource_config = {
    "name": "delta_datasource",
    "class_name": "Datasource",
    "execution_engine": {"class_name": "SparkDFExecutionEngine"},
    "data_connectors": {
        "delta_connector": {
            "module_name": "great_expectations.datasource.data_connector",
            "class_name": "RuntimeDataConnector",
            "batch_identifiers": [
                "prod",
                "run_id1",
            ],
        }
    },
}


# COMMAND ----------

context.test_yaml_config(yaml.dump(my_spark_datasource_config))

# COMMAND ----------

context.add_datasource(**my_spark_datasource_config)

# COMMAND ----------

batch_request = RuntimeBatchRequest(
    datasource_name="delta_datasource",
    data_connector_name="delta_connector",
    data_asset_name="my_data_asset_name",
    batch_identifiers={
        "prod": "my_production_data",
        "run_id1": f"my_run_id{datetime.date.today().strftime('%Y%m%d')}",
    },
    runtime_parameters={"batch_data": df},
)


# COMMAND ----------

expectation_suite_name = "my_expectation_suite"
context.add_or_update_expectation_suite(expectation_suite_name=expectation_suite_name)
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name=expectation_suite_name,
)

print(validator.head())

# COMMAND ----------

from datetime import datetime 

# Define Expectations for the columns
validator.expect_column_values_to_not_be_null("DATETIME")
validator.expect_column_values_to_not_be_null("HOURLY_CONSUMPTION_MW")
validator.expect_column_values_to_not_be_null("COUNTRY")

validator.expect_column_values_to_be_of_type("DATETIME", "TimestampType")
validator.expect_column_values_to_be_of_type("HOURLY_CONSUMPTION_MW", "DoubleType")
validator.expect_column_values_to_be_of_type("COUNTRY", "StringType")

validator.expect_column_values_to_be_in_set("COUNTRY", ["belgium", "denmark", "france", "germany", "greece", "italy", "luxembourg", "netherlands", "spain", "sweden", "switzerland"])

validator.expect_column_values_to_be_between("HOURLY_CONSUMPTION_MW", min_value=0)

# This expectation checks if the mean of the HOURLY_CONSUMPTION_MW is within a certain range. Please adjust the min_value and max_value according to your data.
validator.expect_column_mean_to_be_between("HOURLY_CONSUMPTION_MW", min_value=25000, max_value=50000)

# This expectation checks if the median of the HOURLY_CONSUMPTION_MW is within a certain range. Please adjust the min_value and max_value according to your data.
validator.expect_column_median_to_be_between("HOURLY_CONSUMPTION_MW", min_value=20000, max_value=35000)

# This expectation checks if the standard deviation of the HOURLY_CONSUMPTION_MW is within a certain range. Please adjust the min_value and max_value according to your data.
validator.expect_column_stdev_to_be_between("HOURLY_CONSUMPTION_MW", min_value=40000, max_value=70000)

# Check if timestamps are in the correct range
start_date = datetime(2015, 1, 1)
end_date = datetime(2023, 1, 1)
validator.expect_column_values_to_be_between("DATETIME", min_value=start_date, max_value=end_date)



# COMMAND ----------

validator.save_expectation_suite(discard_failed_expectations=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate data

# COMMAND ----------

my_checkpoint_name = "my_data_validation_checkpoint"
checkpoint_config = {
    "name": my_checkpoint_name,
    "config_version": 1.0,
    "class_name": "SimpleCheckpoint",
    "run_name_template": "%Y%m%d-%H%M%S-my-run-name-template",
}


# COMMAND ----------

my_checkpoint = context.test_yaml_config(yaml.dump(checkpoint_config))

# COMMAND ----------

context.add_or_update_checkpoint(**checkpoint_config)

# COMMAND ----------

checkpoint_result = context.run_checkpoint(
    checkpoint_name=my_checkpoint_name,
    validations=[
        {
            "batch_request": batch_request,
            "expectation_suite_name": expectation_suite_name,
        }
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build and view Data Docs

# COMMAND ----------

html = '/dbfs/great_expectations/uncommitted/data_docs/local_site/index.html'
with open(html, "r") as f:
    data = "".join([l for l in f])
displayHTML(data)

# COMMAND ----------


