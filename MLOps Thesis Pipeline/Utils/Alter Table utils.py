# Databricks notebook source
spark.sql("USE df_dev")
sqlContext.sql("set spark.sql.caseSensitive=True")
from pyspark.sql.utils import AnalysisException

table_names = spark.sql("SHOW TABLES").select('tableName').rdd.flatMap(lambda x: x).collect()
for table_name in table_names:
    spark.sql(f""" ALTER TABLE {table_name} SET TBLPROPERTIES('delta.columnMapping.mode' = 'name', 'delta.minReaderVersion' = '2', 'delta.minWriterVersion' = '5') """)
    columns = spark.sql(f'DESCRIBE {table_name}').select('col_name').rdd.flatMap(lambda x: x).collect()
    for column in columns:
        try:
            spark.sql(f'ALTER TABLE {table_name} RENAME COLUMN {column} TO {column.upper()}')
        except AnalysisException:
            continuespark.sql("USE df_dev")
table_name = 'consumption_regions_hourly'
spark.sql(f""" ALTER TABLE {table_name} SET TBLPROPERTIES('delta.columnMapping.mode' = 'name', 'delta.minReaderVersion' = '2', 'delta.minWriterVersion' = '5') """)
    # columns = spark.sql(f'DESCRIBE {table_name}').select('col_name').rdd.flatMap(lambda x: x).collect()
    # for column in columns:
    #     if column == 'DAILY_CONSUNPTION_MW':
spark.sql(f'ALTER TABLE {table_name} RENAME COLUMN HOURLY_CONSUNPTION_MW TO HOURLY_CONSUMPTION_MW')

# COMMAND ----------

spark.sql("USE df_dev")
sqlContext.sql("set spark.sql.caseSensitive=True")
from pyspark.sql.utils import AnalysisException

table_names = spark.sql("SHOW TABLES").select('tableName').rdd.flatMap(lambda x: x).collect()
for table_name in table_names:
    spark.sql(f""" ALTER TABLE {table_name} SET TBLPROPERTIES('delta.columnMapping.mode' = 'name', 'delta.minReaderVersion' = '2', 'delta.minWriterVersion' = '5') """)
    columns = spark.sql(f'DESCRIBE {table_name}').select('col_name').rdd.flatMap(lambda x: x).collect()
    for column in columns:
        if column == 'DAILY_CONSUNPTION_MW':
            spark.sql(f'ALTER TABLE {table_name} RENAME COLUMN {column} TO DAILY_CONSUMPTION_MW')

# COMMAND ----------

spark.sql("USE df_dev")
table_name = 'consumption_regions_hourly'
spark.sql(f""" ALTER TABLE {table_name} SET TBLPROPERTIES('delta.columnMapping.mode' = 'name', 'delta.minReaderVersion' = '2', 'delta.minWriterVersion' = '5') """)
    # columns = spark.sql(f'DESCRIBE {table_name}').select('col_name').rdd.flatMap(lambda x: x).collect()
    # for column in columns:
    #     if column == 'DAILY_CONSUNPTION_MW':
spark.sql(f'ALTER TABLE {table_name} RENAME COLUMN HOURLY_CONSUNPTION_MW TO HOURLY_CONSUMPTION_MW')
