# Project Documentation: Streamlining Energy Consumption Forecasting using MLOps

![Databricks Badge](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![Azure Badge](https://img.shields.io/badge/Microsoft_Azure-0089D6?style=for-the-badge&logo=microsoft-azure&logoColor=white)
![MLflow Badge](https://img.shields.io/badge/MLflow-FF3621?style=for-the-badge&logo=mlflow&logoColor=white)
![Python Badge](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)


## 📊 Data Source

The data used in this project is sourced from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/). ENTSO-E stands for European Network of Transmission System Operators for Electricity. 
It is an organization that brings together 42 electricity transmission system operators (TSOs) from 35 countries across Europe. ENTSO-E plays a crucial role in coordinating TSOs and facilitating the European electricity market.
The platform provides authoritative, comprehensive, and transparent energy market data for European countries. 

For this project, hourly energy consumption data from 2015 to 2022 has been used. Additionally, some new data from 2023 has also been retrieved for testing purposes. The dataset includes hourly energy consumption data across 11 selected European countries (Belgium, Denmark, France, Germany, Greece, Italy, Luxembourg, Netherlands, Spain, Sweden, Switzerland).


# MLOps Pipeline

## 🛠️ Data Engineering

### Data Ingestion:
In the data ingestion stage, raw energy consumption data is collected from the ENTSO-E Transparency Platform. The data is downloaded and uploaded into Azure Databricks for processing.
It is important to ensure that data ingestion is efficient and reliable, as it forms the foundation for the subsequent steps in the pipeline.

### Data Transformation:
After ingestion, the data undergoes various transformations to convert it into a format that is suitable for analysis and modeling.
This includes converting timestamps to a standardized format, aggregating data, and reshaping datasets. 
In this project, hourly energy consumption data is used; however, it is possible to aggregate this data for different time windows (e.g., daily) based on the requirements.

### Data Quality Checks:

Quality checks are essential to ensure the integrity and completeness of the data. This includes handling missing values, identifying and rectifying any inconsistencies in the data, and ensuring that it meets the required standards for analysis. 

In this project, the primary tool utilized for this purpose is [Great Expectations](https://greatexpectations.io/), an open-source library for setting, validating, and documenting data expectations. 

Great Expectations was instrumental in defining expectations for data, which serve as automated assertions about data quality that are easy to implement and maintain. If any data does not meet these predefined expectations, the system alerts us, thereby ensuring that any decision made based on the data is as accurate as possible.

For an example of the data quality reports produced by Great Expectations in this project, see the links below:

#### Links to Data Quality report files:

- [**Data Quality Expectations**](https://philippos01.github.io/mlops-energy-forecast-thesis/MLOps%20Pipeline/Utils/Great%20Expectations/my_expectation_suite.html)
- [**Data Quality Validation**](https://philippos01.github.io/mlops-energy-forecast-thesis/MLOps%20Pipeline/Utils/Great%20Expectations/fde64798683368bcaf8fe113b0dd4b14.html)


## 🚀 Initial Deployment

This section describes the critical steps undertaken during the initial deployment phase of the MLOps pipeline. The pipeline consists of an exploratory data analysis, feature engineering, model training, and unit testing.

### 🕵️‍♂️ Exploratory Data Analysis (EDA):
Before diving into model training, it is essential to understand the characteristics of the data. Exploratory Data Analysis (EDA) involves summarizing the main features of the data, usually with visual methods. Through EDA, we can begin to uncover patterns, spot anomalies, and frame hypotheses for testing.

- **Univariate Analysis**: Involves the examination of single features or variables. For this project, the Univariate Analysis includes:
  * Distribution of records across years, months, days, and hours.
  * Frequency of records for each country.

- **Bivariate Analysis**: Investigates the relationship between two features or variables. In this project, the Bivariate Analysis includes:
  * Average hourly consumption per country.
  * Monthly consumption trends per country.

- **Visualizations**: Creating graphical representations of the data. For this project, specific visualizations include:
  * A heatmap for Average Hourly Consumption by Country and Hour of Day to observe patterns in energy consumption.
  * Decomposition plots for each country to examine original, trend, seasonality, and residuals in the time series data.

### 🧪 Feature Engineering:
After understanding the data through EDA, the next step is to prepare it for modeling. Feature engineering includes creating new features, transforming existing ones, and encoding categorical variables.

- **One-Hot Encoding of Countries**: This involves converting the categorical 'country' feature into numerical format, where each country is represented as a binary vector.

- **Feature Creation**: Generating new features that might improve the model's performance. For example, creating time-based features like the day of the week, month, year.

- **Primary Key Creation**: Creating a unique identifier for each record. This is essential for indexing and retrieving records efficiently from the database.

- **Saving Features to Databricks Feature Store**: After engineering, features are saved in Databricks Feature Store, which acts as a centralized repository for feature data, ensuring consistency across different models and deployments.

### 🤖 Model Training:
With the features prepared, we now proceed to model training. This step involves selecting an algorithm, training the model, and evaluating its performance.

- **Data Loading from Feature Store**: The features engineered previously are loaded from Databricks Feature Store.

- **Data Splitting**: The dataset is split into training and testing sets by ensuring the continuity of the data to correctly evaluate the model's performance on unseen data.

- **Model Creation and Training**: The algorithm is selected, and the model is trained using the training dataset.

- **Logging to Feature Store**: The trained model, along with its metrics and parameters and artifacts is logged in the Databricks Feature Store for versioning and reproducibility.

### 🧪 Unit Testing:
After the model is trained, it undergoes unit testing to ensure that it meets the required performance benchmarks.

- **Performance Testing**: The model is subjected to a set of tests to evaluate its performance. Metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are used.

- **Proceed to Staging Environment**: If the model passes performance tests, it is moved to a staging environment. This stage closely resembles the production environment and is used for final testing before the model is deployed for real-world use.


## 🔄 Daily Inference

This subsection outlines the daily inference procedure which is a crucial aspect of the MLOps pipeline. It ensures that the model continues to provide value by making predictions on new data.
The daily inference procedure comprises three key steps: feature engineering on new data, the inference process itself, and monitoring the predictions.

### 🧪 Feature Engineering on New Data:
To make predictions on new data, it's important to transform the data in a manner consistent with the training data. This involves applying the same transformations and encodings that were done during the initial deployment phase.

- **Data Transformation**: The new data is transformed to ensure it's in a compatible format for the model to make predictions. This includes handling any missing values, encoding categorical variables, and creating new features.

- **Saving Transformed Data**: Once the data is transformed, it's saved in a structured format that is easily retrievable. This structured data will be used for making predictions.

### 🎯 Daily Inference Procedure:
This is the process where the model uses the transformed new data to make predictions. These predictions can be used for various applications such as forecasting energy consumption.

- **Retrieving New Data**: The transformed new data is retrieved from the database.

- **Batch Scoring**: The model, using a batch scoring function in the feature store, makes predictions on the new data. Batch scoring is efficient for making predictions on large datasets.

- **Saving Predictions**: The predictions made by the model are saved back to the database. This data can be retrieved later for analysis and reporting.

### 📊 Daily Monitoring Procedure:
After the predictions are made and saved, it is critical to monitor how the model is performing on new data. This involves evaluating predictions and creating visualizations.

- **Retrieving Predicted Data**: The data that has been predicted by the model is retrieved from the database.

- **Evaluating Predictions**: The predictions are evaluated through various metrics to understand how well the model is performing.

- **Creating Visualizations**: Visualizations such as graphs and charts are created to help interpret the predictions. This can include trend analysis and distribution of predictions over time.

- **Reporting**: The results from the evaluation and visualizations are documented and reported. This reporting can be used for decision-making and planning.

By employing a systematic daily inference procedure, it ensures that the model remains functional and valuable in a real-world setting, while constantly monitoring its performance.

## 🔄 Model Retraining

One of the key aspects of maintaining a robust MLOps pipeline is to ensure that the deployed models remain efficient and accurate over time. The Model Retraining subsection focuses on the automated retraining of models on a regular basis, using the latest data.

### 🗓️ Scheduled Retraining:
Model retraining is scheduled to occur automatically at regular intervals - every 1, 3, or 6 months. This is essential as data patterns may evolve, and the model needs to adapt to these changes to maintain its accuracy and relevance.

- **Data Preparation**: The data saved during the daily inference, which includes the predicted values and their corresponding actual values, is used for retraining. This dataset is accumulated over the decided interval (1, 3, or 6 months).

- **Retraining Process**: The model is retrained using the accumulated data. This ensures that the model learns from the most recent data patterns and adapts its parameters accordingly.

### 📈 Performance Evaluation:
Post retraining, it's imperative to evaluate the model's performance to ascertain whether there’s an improvement in the predictions.

- **Tracking Progress**: The performance of the models over time is tracked. This includes monitoring the number of trainings and retrainings and how the metrics evolve with each iteration.

- **Comparative Analysis**: The retrained model, which is initially in the staging environment, is compared against the current production model. The evaluation metrics of both models are analyzed to determine if the retrained model shows improved performance.

- **Model Promotion**: If the retrained model in the staging environment outperforms the current production model, it is promoted to replace the production model. The model that was in production is archived for record-keeping.

- **Documentation**: All the steps, decisions, and metrics are documented for future reference and transparency.

By continuously monitoring and retraining the model, this process ensures that the model remains adaptive to changing data patterns and provides the most accurate and efficient predictions possible.


## 🚀 Deployment Strategy

In the context of MLOps, deployment is a critical phase where the machine learning model is integrated into a production environment, making it accessible for real-world forecasting. A robust deployment strategy ensures that the model is reliable, scalable, and efficient.

### MLflow and Databricks Integration:
- The model, post-training, is saved and registered in Feature Store within MLflow, a platform that manages the ML lifecycle, including experimentation, reproducibility, and deployment.

- MLflow is natively integrated within the Databricks workspace. This seamless integration is crucial as it allows for efficient management and tracking of models within a familiar ecosystem.

### Scalability and Performance:
- Databricks, known for its high-performance analytics engine, is particularly suited for handling large datasets and complex computations. By deploying the model within Databricks, we leverage its ability to scale effortlessly to meet data and computational demands.

# 🔄 Workflow Overview

In this project, we have established three main workflows that are integral to the systematic functioning and updating of the energy consumption forecasting system:

## 1.  Initial Deployment / Redeployment

This workflow encompasses all the steps necessary for the initial deployment of the model, as well as any subsequent redeployments. It includes data engineering, exploratory data analysis, feature engineering, model training, and performance evaluation. This workflow is initiated manually and ensures that the model is properly set up and integrated into the Azure Databricks and MLflow ecosystem.

## 2.  Daily Inference

The Daily Inference workflow is automated and triggered every day. Its purpose is to forecast the energy consumption for the next day. This workflow starts by retrieving new data from the database and processing it to be compatible with the model. Through the batch scoring function of the feature store, predictions are generated and subsequently saved back into the database for further analysis and utilization.

## 3. Model Retraining

The Model Retraining workflow is designed to ensure that the forecasting model remains up-to-date and incorporates the latest data for higher accuracy. This workflow is automatically triggered every three months. During this process, the model is retrained using newly collected data that has been saved during the Daily Inference workflow. After the retraining process, the model's performance is evaluated and compared to the current production model. If the retrained model exhibits improved performance, it replaces the existing production model, which is then archived.

These workflows are designed to work seamlessly together to provide an efficient, scalable, and up-to-date energy consumption forecasting system. Through automation and systematic processes, this setup ensures accuracy and sustainability in forecasting energy consumption across multiple European countries.


## Overall Architecture
![MLOps Architecture](MLOps%20Pipeline/Utils/Images/MLOps%20Architecture%20(1).png)
## 🎉 Conclusion

This project efficiently addresses the challenge of forecasting energy consumption across multiple European countries. By employing Azure Databricks and MLflow, it leverages a powerful and scalable environment for data processing and model deployment. Continuous monitoring and automatic retraining ensure that the model remains accurate and up-to-date. 
This solution offers immense value to utilities and grid operators in optimizing energy management and planning.
