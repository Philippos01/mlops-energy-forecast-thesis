# Databricks notebook source
!pip install statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
from pyspark.sql.functions import count, when, isnull, col
from pyspark.sql import functions as F
import plotly.subplots as sp
import plotly.graph_objects as go

# COMMAND ----------

# MAGIC %md
# MAGIC ## Univariate Analysis

# COMMAND ----------

spark.sql('USE df_dev')
df = spark.read.table('final_consumption_countries_hourly')

# COMMAND ----------

# MAGIC %md
# MAGIC * Distribution of records across years, months, days and hours:

# COMMAND ----------

display(df.withColumn('year', F.year('DATETIME')).groupBy('year').count())
display(df.withColumn('month', F.month('DATETIME')).groupBy('month').count())
display(df.withColumn('day', F.dayofweek('DATETIME')).groupBy('day').count())
display(df.withColumn('hour', F.hour('DATETIME')).groupBy('hour').count())

# COMMAND ----------

# MAGIC %md
# MAGIC * Frequency of records for each country

# COMMAND ----------

df.groupBy('COUNTRY').count().show()


# COMMAND ----------

# MAGIC %md
# MAGIC ##  Bivariate Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC * Average hourly consumption per country

# COMMAND ----------

df.groupBy('COUNTRY').agg(F.avg('HOURLY_CONSUMPTION_MW').alias('avg_consumption')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC * Monthly consumption trends per country

# COMMAND ----------

df.withColumn('year', F.year('DATETIME')) \
  .withColumn('month', F.month('DATETIME')) \
  .groupBy('year', 'month', 'COUNTRY') \
  .agg(F.sum('HOURLY_CONSUMPTION_MW').alias('total_consumption')) \
  .orderBy('year', 'month') \
  .show()


# COMMAND ----------

# MAGIC %md
# MAGIC * Heatmap: Average hourly consumption for each country by hour of the day or by month of the year

# COMMAND ----------

import plotly.graph_objects as go

# Convert the DataFrame to a 2D list for Plotly
heatmap_data = df_heatmap.values.tolist()

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
    z=heatmap_data,
    x=df_heatmap.columns.tolist(),
    y=df_heatmap.index.tolist(),
    colorscale='RdBu_r', # you can change this to other color scales
))

# Set the layout
fig.update_layout(
    title='Average Hourly Consumption by Country and Hour of Day',
    xaxis_title='Hour of Day',
    yaxis_title='Country',
)

# Display the figure
fig.show()


# COMMAND ----------

pandas_df = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC The decompose_country function takes a DataFrame df and a country name as inputs. It performs a time series decomposition on the 'HOURLY_CONSUMPTION_MW' column of the DataFrame for the specified country.
# MAGIC
# MAGIC 1. It filters the DataFrame to include data only for the specified country.
# MAGIC 1. The data is sorted by date.
# MAGIC 1. The date column is set as the index.
# MAGIC 1. The data is resampled to a chosen frequency (monthly in this case).
# MAGIC 1. Any missing values are filled using forward filling.
# MAGIC 1. The seasonal decomposition is performed using an additive model.
# MAGIC 1. The trend, seasonality, and residuals components are extracted.
# MAGIC 1. Subplots are created for the original data, trend, seasonality, and residuals.
# MAGIC 1. Traces are added to the subplots to visualize the components.
# MAGIC 1. The plot layout is updated with appropriate dimensions and a title.
# MAGIC 1. The plot is displayed.
# MAGIC
# MAGIC By calling the decompose_country function with a DataFrame and a country name, the code generates a plot showing the original data, trend, seasonality, and residuals components of the time series for that country.

# COMMAND ----------

def decompose_country(df, country):
    # Filter data for the specified country
    df_country = df[df['COUNTRY'] == country]
    
    # Ensure the data is sorted by date
    df_country = df_country.sort_values('DATETIME')

    # Set the date as the index
    df_country.set_index('DATETIME', inplace=True)
    
    # Resample to hourly data, you can choose different frequency according to your data
    df_country = df_country.resample('M').asfreq()

    # Forward fill to handle the newly created NaNs
    df_country = df_country.bfill()

    # Perform the decomposition
    decomposition = seasonal_decompose(df_country['HOURLY_CONSUMPTION_MW'], model='additive')

    # Get the trend, seasonality and residuals
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Create subplots: 4 rows, 1 column
    fig = sp.make_subplots(rows=4, cols=1)

    # Add traces
    fig.add_trace(go.Scatter(x=df_country.index, y=df_country['HOURLY_CONSUMPTION_MW'], mode='lines', name='Original'), row=1, col=1)
    fig.add_trace(go.Scatter(x=trend.index, y=trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal, mode='lines', name='Seasonality'), row=3, col=1)
    fig.add_trace(go.Scatter(x=residual.index, y=residual, mode='lines', name='Residuals'), row=4, col=1)

    # Update layout
    fig.update_layout(height=800, width=1000, title_text="Decomposition for " + country, showlegend=True)

    # Render the plot
    fig.show()

decompose_country(pandas_df, 'greece')


# COMMAND ----------


