# Databricks notebook source
# TODO: Create the union query for all regions (dynamically using _sqldf dataframe ?)
# TODO: Decide on training and test period based on the min and max dates per region

# COMMAND ----------

# MAGIC %sql
# MAGIC USE df_dev;
# MAGIC SELECT * FROM aep_hourly;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES;

# COMMAND ----------

tables_initial = [row['tableName'] for row in _sqldf.collect()]

# Keep tables with valid format name
tables = [t for t in tables_initial if len(t.split('_'))==2]
print(tables)
print(len(tables))

# COMMAND ----------

# Union all tables into one
def get_union_query(table_name):
    region = table_name.split('_')[0]
    return f"""
    SELECT Datetime, {region}_MW AS Hourly_Consunption_MW, '{region.upper()}' AS Region
    FROM df_dev.{table_name}
    """
overall_final_query = 'UNION ALL'.join(map(get_union_query, tables))
print(overall_final_query)

df_regions_hourly = spark.sql(overall_final_query)
df_regions_hourly.show()
df_regions_hourly.createOrReplaceTempView ('consumption_regions_hourly')

# COMMAND ----------

# Transform hourly consumption to daily one
df_regions_daily = spark.sql("""
SELECT DATE_FORMAT(datetime, 'yyyy-MM-dd') AS Date, 
       Region,
       SUM(Hourly_Consunption_MW) AS Daily_Consunption_MW
FROM consumption_regions_hourly
GROUP BY DATE_FORMAT(datetime, 'yyyy-MM-dd'), Region
""")
df_regions_daily.show()

# COMMAND ----------

# Save permanent table into Hive
df_regions_daily.write.mode('overwrite').saveAsTable("df_dev.consumption_regions_daily")

# COMMAND ----------

# Get Minimum and Maximum Date per region
spark.sql("""
SELECT Region, MIN(Date) AS First_Day, MAX(Date) AS Last_Day
FROM consumption_regions_daily
GROUP BY Region
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC We conclude that: 
# MAGIC - The datapoints from 2013-06-01 to 2018-08-03 will be used
# MAGIC - The NI region will not be included as the time periods do not overlap
# MAGIC
# MAGIC Regarding the split of the datapoints, let's use:
# MAGIC - 2013-06-01 to 2017-12-31 for training
# MAGIC - 2018-01-01 to 2018-05-31 for testing
# MAGIC - 2018-06-01 to 2018-08-03 for deployment simulation
