# Databricks notebook source
! pip install plotly
from pyspark.sql.functions import count, when, isnull, col
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# COMMAND ----------

spark.sql('USE db_monitor') 

# COMMAND ----------

df = spark.read.table('final_monitoring_consumption_countries_hourly')

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Profiling

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity & Data Quality Checks 

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Missing Values Check: Counts the number of null values in each column of the DataFrame, calculates the total number of nulls, and calculates the percentage of null values relative to the total number of rows.
# MAGIC
# MAGIC 1. Duplicates Check: Determines the count of duplicate rows by subtracting the count of the DataFrame after dropping duplicates from the original count. It also calculates the percentage of duplicate rows relative to the total number of rows.
# MAGIC
# MAGIC 1. Invalid Values Check: Counts the number of invalid records in each dataframe(ex. negative/zero energy consumption)
# MAGIC
# MAGIC 1. Outlier Detection: Defines bounds based on the first and third quartiles of the 'Actual_MW' column using the approxQuantile function. It then identifies outliers by counting the number of rows where the 'Actual_MW' value falls outside the defined bounds.
# MAGIC
# MAGIC 1. Schema Verification: Compares the DataFrame's column names to the expected column names ('Datetime', 'Actual_MW', 'start_time', 'country') and checks if any unexpected columns exist.
# MAGIC
# MAGIC 1. Summary Print: Displays the results of the data quality checks, including the count and percentage of missing values, count and percentage of duplicate rows, presence of outliers, and whether the schema matches the expected columns.
# MAGIC
# MAGIC 1. Statistical Checks: Prints basic statistical measures of the DataFrame using the describe function.

# COMMAND ----------

print("\nData quality checks for concatenated dataframe...")

# 1. Missing values check
null_counts = df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).collect()
total_nulls = sum(row[c] for row in null_counts for c in df.columns)
nulls_percentage = (total_nulls / df.count()) * 100

# 2. Invalid values check
invalid_values = df.filter((df['HOURLY_CONSUMPTION_MW'] <= 0)).count()
invalid_percentage = (invalid_values / df.count()) * 100

# 3. Duplicates check
duplicates_count = df.count() - df.dropDuplicates().count()
duplicates_percentage = (duplicates_count / df.count()) * 100

# 4. Outlier detection
bounds = {
    c: dict(
        zip(["q1", "q3"], df.approxQuantile(c, [0.25, 0.75], 0))
    )
    for c in ["HOURLY_CONSUMPTION_MW"]
}
outliers = 0
for c in bounds:
    iqr = bounds[c]['q3'] - bounds[c]['q1']
    bounds[c]['lower'] = bounds[c]['q1'] - (iqr * 1.5)
    bounds[c]['upper'] = bounds[c]['q3'] + (iqr * 1.5)
    outliers += df.filter(
        (df[c] < bounds[c]['lower']) | 
        (df[c] > bounds[c]['upper'])
    ).count()

# 5. Schema verification 
expected_columns = ['DATETIME', 'HOURLY_CONSUMPTION_MW', 'COUNTRY']
schema_check = len(set(df.columns) - set(expected_columns)) == 0

# Summary print
print(f"Missing values: {total_nulls if total_nulls > 0 else 'None'} ({nulls_percentage:.4f}% of total rows)")
print(f"Duplicate rows: {duplicates_count if duplicates_count > 0 else 'None'} ({duplicates_percentage:.4f}% of total rows)")
print(f"Invalid values: {invalid_values if invalid_values > 0 else 'None'} ({invalid_percentage:.4f}% of total rows)")
print(f"Outliers: {'Found' if outliers else 'None'}")
print(f"Schema check: {'Unexpected schema' if not schema_check else 'Schema as expected'}")

# 6. Statistical checks
print("Basic statistical measures:")
df.describe().show()


# COMMAND ----------

from pyspark.sql.functions import col, lag, expr
from pyspark.sql.window import Window
from pyspark.sql import functions as F

# Specify the column names in your DataFrame
datetime_col = "DATETIME"
country_col = "COUNTRY"

# Sort the DataFrame by 'DATETIME' within each country
window_spec = Window.partitionBy(country_col).orderBy(datetime_col)
df_sorted = df.withColumn("start_time", col(datetime_col).cast("timestamp")).orderBy(country_col, datetime_col)

# Calculate the time difference between consecutive records within each country
df_sorted = df_sorted.withColumn("time_diff", col("start_time").cast("long") - lag(col("start_time").cast("long")).over(window_spec))

# Check if all time differences are exactly 1 hour within each country
country_continuity = df_sorted.groupBy(country_col).agg(F.min(F.when(col("time_diff") == 3600, True)).alias("is_continuous"))

# Show the results
country_continuity.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Datasets(Duplicates,Null,Invalid)

# COMMAND ----------

pandas_df = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## LinePlot of the Energy Consumption Forecasting

# COMMAND ----------

def create_plot(country, time_range_name, time_range):
    # Filter data for the specific country
    filtered_df = pandas_df[pandas_df['COUNTRY'] == country]

    # Filter data to the specified time range
    filtered_df = filtered_df.loc[(filtered_df['DATETIME'] >= time_range[0]) & (filtered_df['DATETIME'] <= time_range[1])]

    # Aggregate data to daily averages
    daily_df = filtered_df.groupby(pd.Grouper(key='DATETIME', freq='D')).mean()

    # Create a rolling average
    daily_df['Rolling_MW'] = daily_df['HOURLY_CONSUMPTION_MW'].rolling(window=7).mean()

    # Find the times corresponding to min and max Actual_MW in the filtered data
    min_time = filtered_df.loc[filtered_df['HOURLY_CONSUMPTION_MW'].idxmin(), 'DATETIME']
    max_time = filtered_df.loc[filtered_df['HOURLY_CONSUMPTION_MW'].idxmax(), 'DATETIME']

    # Create a line plot
    fig = go.Figure()

    # Add trace for actual MW
    fig.add_trace(go.Scatter(x=filtered_df['DATETIME'], y=filtered_df['HOURLY_CONSUMPTION_MW'], mode='markers',
                             name='Actual MW',
                             hovertemplate=
                             "<b>%{x}</b><br><br>" +
                             "Actual MW: %{y}<br>" +
                             "<extra></extra>"))

    # Add trace for rolling average
    fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df['Rolling_MW'], mode='markers',
                             name='7-day Rolling Average',
                             hovertemplate=
                             "<b>%{x}</b><br><br>" +
                             "Rolling MW: %{y}<br>" +
                             "<extra></extra>"))

    # Add markers for min and max values
    fig.add_trace(go.Scatter(x=[min_time, max_time],
                             y=[filtered_df.loc[filtered_df['DATETIME'] == min_time, 'HOURLY_CONSUMPTION_MW'].values[0],
                                filtered_df.loc[filtered_df['DATETIME'] == max_time, 'HOURLY_CONSUMPTION_MW'].values[0]],
                             mode='markers+text',
                             marker=dict(size=[10, 10]),
                             text=['Min', 'Max'],
                             textposition="top center",
                             name='Min/Max',
                             hovertemplate=
                             "<b>%{x}</b><br><br>" +
                             "Actual MW: %{y}<br>" +
                             "<extra></extra>"))

    # Add vertical lines for min and max values
    fig.add_shape(
        dict(type="line", x0=min_time, y0=0, x1=min_time, y1=filtered_df['HOURLY_CONSUMPTION_MW'].max(),
             line=dict(color="RoyalBlue", width=2)))
    fig.add_shape(
        dict(type="line", x0=max_time, y0=0, x1=max_time, y1=filtered_df['HOURLY_CONSUMPTION_MW'].max(),
             line=dict(color="RoyalBlue", width=2)))

    # Update layout
    fig.update_layout(title=f'Daily Energy Consumption for {country.capitalize()} over {time_range_name.capitalize()}',
                      xaxis_title='DATETIME',
                      yaxis_title='Energy Consumption (HOURLY_CONSUMPTION_MW)',
                      hovermode='x')

    fig.show()


# COMMAND ----------

# Define time ranges
time_ranges = {
    'decade':['2015-01-01','2023-01-01'],
    'year': ['2022-01-01', '2023-01-01'],
    'month': ['2023-01-01', '2023-02-01'],
    'week': ['2022-12-25', '2023-01-01']
}

create_plot('greece','month', time_ranges['month'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Box-and-Whisker Plot

# COMMAND ----------

import plotly.graph_objects as go

def create_box_plot(country):
    # Filter data for the specific country
    filtered_df = pandas_df[pandas_df['COUNTRY'] == country]

    # Create a box plot
    fig = go.Figure()

    # Add box trace
    fig.add_trace(go.Box(y=filtered_df['HOURLY_CONSUMPTION_MW'], name='HOURLY_CONSUMPTION_MW'))

    # Update layout
    fig.update_layout(title=f'Boxplot of Energy Consumption for {country.capitalize()}',
                      yaxis_title='Energy Consumption (HOURLY_CONSUMPTION_MW)')

    fig.show()

create_box_plot('greece')

# COMMAND ----------


