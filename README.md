# Streamlining Energy Consumption Forecasting using MLOps

This project focuses on streamlining the process of forecasting energy consumption by employing Machine Learning Operations (MLOps). It integrates data engineering, machine learning algorithms, and automation to create a scalable and efficient forecasting system.

[![Generic badge](https://img.shields.io/badge/Status-Complete-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Databricks-Powered-blue.svg)](https://shields.io/)

## Table of Contents
- [Introduction](#-introduction)
- [Requirements](#️-requirements)
- [Setup & Installation](#️-setup--installation)
- [Aim of the Project](#-aim-of-the-project)
- [Results and Findings](#-results-and-findings)
- [Additional Documentation](#-additional-documentation)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)
- [Contributing](#️-contributing)
- [Contact](#-contact)
- [Citation](#-citation)

## 📌 Introduction
The core objective of this project is to develop and orchestrate an automated pipeline for forecasting energy consumption across eleven European countries, namely Belgium, Denmark, France, Germany, Greece, Italy, Luxembourg, Netherlands, Spain, Sweden, and Switzerland. The pipeline is specifically tailored for processing hourly energy consumption data.

This project is fully integrated within the Azure Cloud ecosystem and leverages the power and scalability of the Databricks platform. Utilizing these cutting-edge cloud technologies ensures that the pipeline is not only highly scalable but also incredibly efficient and reliable.

Forecasting energy consumption is pivotal for European countries, as it plays an instrumental role in ensuring energy sustainability, optimizing power generation and distribution, and facilitating informed decision-making. By producing reliable and timely forecasts, this project empowers energy providers and stakeholders with insights that can lead to cost reductions, enhanced operational efficiencies, and the promotion of sustainable energy practices.

The end goal is to establish a robust, scalable, and automated solution that provides precise forecasting of energy consumption. Through automating the forecasting process, we aim to keep up with the ever-evolving demands of the energy sector and contribute significantly to Europe’s broader economic and environmental objectives.


## 🛠️ Requirements

### Data Source
This project utilizes data from the ENTSO-E Transparency Platform, which provides comprehensive information on the European electricity market. To access the dataset, you will need to create an account on the ENTSO-E Transparency Platform. Once you have an account, you can access and download the dataset required for this project.

[Create an account on ENTSO-E Transparency Platform](https://keycloak-transparency.entsoe.eu/realms/tp/login-actions/registration?client_id=tp-web&tab_id=KZA2clTzsYo)

### Libraries and Dependencies
This project is dependent on several libraries and frameworks. It's important to ensure that all of the necessary libraries are installed to be able to run the code seamlessly.

You can install the required libraries using the <b>'requirements.txt'</b> file included in the repository. Run the following command:

```
pip install -r requirements.txt
```

### Azure and Databricks
As the project is fully integrated with the Azure Cloud and utilizes the Databricks platform, you will need to have:

* An active Azure subscription.
* A Databricks workspace set up within Azure.

## ⚙️ Setup & Installation

Follow these simplified steps to set up the project:

1. **Create Accounts**: Sign up for [Azure Cloud](https://azure.microsoft.com/), [Databricks](https://databricks.com/), and [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/).

2. **Clone the Repository**: Clone this repository to your machine or Azure virtual machine.

   ```sh
   git clone (https://github.com/Philippos01/mlops-energy-forecast-thesis.git)

3. **Install Requirements**: In the project directory, run:

4. **Set Up Databricks**: Log in to Databricks, create a new workspace, and import the project notebooks.

5. **Configure Azure**: In your Azure account, set up necessary resources and integrate Databricks.

6. **Access Dataset**: Log in to ENTSO-E Transparency Platform, get API keys, and configure them in the project.

7. **Run Notebooks**: In Databricks, open and run the notebooks.

8. **Monitor with MLflow**: Keep track of experiments using MLflow in Databricks.

9. **Deploy the Model**: After training, deploy the model as per the documentation.


## 🔍 Aim of the Project

## 🎯 Aim of the Project

The primary aim of this project is to create a robust and scalable solution for forecasting energy consumption across 11 European countries (Belgium, Denmark, France, Germany, Greece, Italy, Luxembourg, Netherlands, Spain, Sweden, Switzerland) on an hourly basis. The motivation behind forecasting energy consumption is to help utilities, grid operators, and consumers in these countries in better planning and managing energy resources which is crucial for sustainability and efficiency.

Here are the specific objectives that this project aims to accomplish:

- **Automate Data Acquisition**: The project automates the process of acquiring the relevant energy consumption data from the ENTSO-E Transparency Platform, which is an authoritative source for European energy market data.

- **Data Processing and Feature Engineering**: Processing the raw data to engineer features that are significant for forecasting models. This includes handling any missing or inconsistent data and engineering new features that might improve the performance of the forecasting models.

- **Model Building and Evaluation**: Building forecasting models using various machine learning algorithms. These models are then evaluated using relevant metrics to quantify their performance.

- **Scalability and Performance**: The solution is built on the Azure cloud and Databricks platform to ensure that it is scalable and can handle large volumes of data efficiently.

- **Deployment and Monitoring**: Once a suitable model is developed, it is deployed to make forecasts available for end-users. Additionally, the project incorporates monitoring tools to keep track of the model’s performance over time.

By achieving these objectives, the project contributes to the broader goal of optimizing energy consumption, which has substantial economic and environmental benefits.

For a comprehensive and in-depth analysis of the project's objectives and how it achieves them, please refer to the detailed documentation:

[📄 Read the Detailed Documentation](./documentation.md)

## 📈 Results and Findings

This section presents the results and findings obtained through the energy consumption forecasting pipeline. The results are categorized into explanatory analysis, average hourly consumption analysis, model comparison, and evaluation metrics for the deployed model.

### Explanatory Analysis

#### Daily Energy Consumption(e.g. Greece)

Explanatory data analysis is essential for understanding the patterns and trends in the dataset. Below is a plot illustrating daily energy consumption in Greece. The plot reveals seasonality and trends in energy consumption, which are crucial for accurate forecasting.

![Greece Daily Energy Consumption](https://github.com/Philippos01/mlops-energy-forecast-thesis/blob/main/MLOps%20Thesis%20Pipeline/Utils/Images/newplot.png)
*Daily Energy Consumption in Greece.*

### Average Hourly Consumption by Country and Hour of Day

The plot below provides insights into average hourly energy consumption by country and hour of day. This is crucial to understand which countries consume more energy at different times of the day and can guide resource allocation and energy production planning.

![Average Hourly Consumption by Country and Hour of Day](https://github.com/Philippos01/mlops-energy-forecast-thesis/blob/main/MLOps%20Thesis%20Pipeline/Utils/Images/newplot%20(1).png)
*Average Hourly Consumption by Country and Hour of Day.*

### Model Comparison: Daily Staging & Production Model Comparison for Greece

To evaluate and select the best model for forecasting, we compared the daily staging and production models. The plot below illustrates how closely each model's predictions match the actual energy consumption data(For the sake of the example we illustrate data from Greece for the 1st week of April)

![Daily Staging & Production Model Comparison for Greece](https://github.com/Philippos01/mlops-energy-forecast-thesis/blob/main/MLOps%20Thesis%20Pipeline/Utils/Images/newplot%20(5).png)
*Daily Staging & Production Model Comparison for Greece over one week.*

### Evaluation Metrics for Deployed Model

The current deployed model was evaluated based on various metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²). These metrics provide a quantitative understanding of the model's performance in forecasting energy consumption.

- **MSE**: 24742781.8
- **MAE**: 1859.5
- **RMSE**: 4974.2
- **R²**: 0.994
- **Training Time**: 134.2 sec

These findings and insights are instrumental for utility companies, policy-makers, and consumers in making informed decisions regarding energy consumption, production, and resource allocation.

For a more in-depth analysis of the results and findings, please refer to the detailed documentation.

[📄 Read the Detailed Documentation](./documentation.md)



## 📘 Additional Documentation
- [Link to additional documentation](path/to/additional/documentation)

## 🙏 Acknowledgments
- Acknowledge any individuals or organizations that contributed to this project.

## 📄 License
- Include licensing information here.

## 🖊️ Contributing
- Explain how others can contribute to this project. Detail the process for submitting pull requests or opening issues.

## 👥 Contact
- Provide contact information or links to your social media profiles.

## 🧾 Citation
- If applicable, provide a citation format for others who use this project in their research.

