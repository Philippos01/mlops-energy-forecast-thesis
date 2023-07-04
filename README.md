# Streamlining Energy Consumption Forecasting using MLOps

This project focuses on streamlining the process of forecasting energy consumption by employing Machine Learning Operations (MLOps). It integrates data engineering, machine learning algorithms, and automation to create a scalable and efficient forecasting system.

[![Generic badge](https://img.shields.io/badge/Status-Complete-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Databricks-Powered-blue.svg)](https://shields.io/)
[![made-with-azure](https://img.shields.io/badge/Made%20with-Azure-1f425f.svg)](https://azure.microsoft.com/)
[![made-with-databricks](https://img.shields.io/badge/Made%20with-Databricks-orange.svg)](https://www.databricks.com/)


## Table of Contents
- [Introduction](#-introduction)
- [Requirements](#️-requirements)
- [Setup & Installation](#️-setup--installation)
- [Aim of the Project](#-aim-of-the-project)
- [Results and Findings](#-results-and-findings)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)
- [Related Publication](#-related-publication)
- [Citation](#-citation)

## 📌 Introduction
The core objective of this project is to develop and orchestrate an automated pipeline for forecasting energy consumption across eleven European countries, namely Belgium, Denmark, France, Germany, Greece, Italy, Luxembourg, Netherlands, Spain, Sweden, and Switzerland. The pipeline is specifically tailored for processing hourly energy consumption data.

This project is fully integrated within the Azure Cloud ecosystem and leverages the power and scalability of the Databricks platform. Utilizing these cutting-edge cloud technologies ensures that the pipeline is not only highly scalable but also incredibly efficient and reliable.

Forecasting energy consumption is pivotal for European countries, as it plays an instrumental role in ensuring energy sustainability, optimizing power generation and distribution, and facilitating informed decision-making. By producing reliable and timely forecasts, this project empowers energy providers and stakeholders with insights that can lead to cost reductions, enhanced operational efficiencies, and the promotion of sustainable energy practices.

The end goal is to establish a robust, scalable, and automated solution that provides precise forecasting of energy consumption. Through automating the forecasting process, we aim to keep up with the ever-evolving demands of the energy sector and contribute significantly to Europe’s broader economic and environmental objectives.


## 🛠️ Requirements

### Data Source
This project utilizes data from the ENTSO-E Transparency Platform, which provides comprehensive information on the European electricity market. To access the dataset, you will need to create an account on the ENTSO-E Transparency Platform. Once you have an account, you can access and download the dataset required for this project.

[Create an account on ENTSO-E Transparency Platform](https://keycloak-transparency.entsoe.eu/realms/tp/protocol/openid-connect/auth?response_type=code&client_id=tp-web&redirect_uri=https%3A%2F%2Ftransparency.entsoe.eu%2Fsso%2Flogin&state=7135aea4-5563-4a24-9fae-727dcee13294&login=true&scope=openid)

### Libraries and Dependencies
This project is dependent on several libraries and frameworks. It's important to ensure that all of the necessary libraries are installed to be able to run the code seamlessly.

You can install the required libraries using the <b>'requirements.txt'</b> file included in the repository. Run the following command:

```
cd mlops-energy-forecast-thesis/MLOps Pipeline/Utils
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
   git clone https://github.com/Philippos01/mlops-energy-forecast-thesis.git

3. **Install Requirements**: Navigate to the project directory and install the required libraries using the requirements.txt file.
```
cd mlops-energy-forecast-thesis/MLOps Pipeline/Utils
pip install -r requirements.txt
```

4. **Set Up Databricks**: Log in to Databricks, and create a new workspace. Within the workspace, create a new cluster and make sure that it's running. Import the project notebooks into your workspace.
   
5. **Configure Azure**: In your Azure account, create a resource group. Within this resource group, create a Databricks workspace (if you haven't already during the Databricks setup) and configure the necessary resources such as storage accounts, networking, etc.
   
6. **Download and Import Dataset**: Log in to the ENTSO-E Transparency Platform and download the dataset. Import this dataset into Databricks.
   
7. **Run Notebooks** : In Databricks, open the notebooks and attach them to the cluster you created earlier. Run the cells in sequence, making sure to input your API keys when prompted.

8. **Monitor with MLflow**: You can keep track of experiments, parameters, metrics, and artifacts using MLflow in Databricks.

9. **Deploy the Model**: After training and evaluating the model, follow the instructions in the documentation to deploy it for forecasting.

10. **Schedule Notebooks**: Optionally, you can schedule the notebooks to run periodically to automate data retrieval and model training.

## 🎯 Aim of the Project

This project aims to implement a data-driven approach for forecasting energy consumption across 11 European countries (Belgium, Denmark, France, Germany, Greece, Italy, Luxembourg, Netherlands, Spain, Sweden, Switzerland) on an hourly basis using Azure Databricks. The technical steps encompassed in the project are as follows:

* **Data Acquisition** : Downlaod energy consumption data from the ENTSO-E Transparency Platform, an authoritative source for European energy market data. This project utilizes manual downloads and uploads to Databricks, but this process can be automated for future scalability.

* **Data Processing and Feature Engineering**: Handle any missing or inconsistent data and engineer new features that might improve the performance of the forecasting models. This involves processing the raw data to format it appropriately for machine learning models.

* **Model Building**: Develop forecasting models using machine learning algorithms such as XGBoost and LSTM (Long Short-Term Memory networks) to predict energy consumption patterns. The choice of algorithms is based on their proven performance in time-series forecasting tasks.

* **Model Evaluation**: Evaluate the performance of the forecasting models using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²). This helps in quantifying how well the models are performing.

* **Deployment and Monitoring**: Save the chosen model in Feature Store and make it available for inference within Databricks. Incorporate monitoring tools to track the model’s performance over time and ensure it stays within acceptable limits. This approach facilitates a seamless integration within the Databricks ecosystem, enabling easy access and utilization of the model for forecasting purposes.

* **Scalability and Performance**: Leverage the Azure cloud and Databricks platform to ensure that the implemented solution can handle large volumes of data efficiently. This enables the project to scale with the addition of new data or expansion to more countries.

By successfully implementing these technical steps, this project contributes to the larger goal of enabling better energy management and planning through data-driven insights and forecasts.

For a comprehensive and in-depth analysis of the project's objectives and how it achieves them, please refer to the detailed documentation:

[📄 Read the Detailed Documentation](./DOCUMENTATION.md)

## 📈 Results and Findings

This section presents the results and findings obtained through the energy consumption forecasting pipeline. The results are categorized into explanatory analysis, average hourly consumption analysis, model comparison, and evaluation metrics for the deployed model.

### Explanatory Analysis

#### Daily Energy Consumption(e.g. Greece)

Explanatory data analysis is essential for understanding the patterns and trends in the dataset. Below is a plot illustrating daily energy consumption in Greece. The plot reveals seasonality and trends in energy consumption, which are crucial for accurate forecasting.

![Greece Daily Energy Consumption](MLOps%20Pipeline/Utils/Images/newplot.png)
*Daily Energy Consumption in Greece.*

### Average Hourly Consumption by Country and Hour of Day

The plot below provides insights into average hourly energy consumption by country and hour of day. This is crucial to understand which countries consume more energy at different times of the day and can guide resource allocation and energy production planning.

![Average Hourly Consumption by Country and Hour of Day](MLOps%20Pipeline/Utils/Images/newplot%20(1).png)
*Average Hourly Consumption by Country and Hour of Day.*

### Model Comparison: Daily Staging & Production Model Comparison for Greece

To evaluate and select the best model for forecasting, we compared the daily staging and production models. The plot below illustrates how closely each model's predictions match the actual energy consumption data(For the sake of the example we illustrate data from Greece for the 1st week of April)

![Daily Staging & Production Model Comparison for Greece](MLOps%20Pipeline/Utils/Images/newplot%20(5).png)
*Daily Staging & Production Model Comparison for Greece over one week.*

### Evaluation Metrics for Deployed Model

The current deployed model was evaluated based on various metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²). These metrics provide a quantitative understanding of the model's performance in forecasting energy consumption.

- **MSE**: 24742781.8
- **MAE**: 1859.5
- **RMSE**: 4974.2
- **R²**: 0.994
- **Training Time**: 134.2 sec

These findings and insights are instrumental for utility companies, policy-makers, and consumers in making informed decisions regarding energy consumption, production, and resource allocation.

## 🙏 Acknowledgments

This project was conducted as part of my thesis at the Athens University of Economics and Business, Department of Management Science and Technology.

## 👥 Contact

If you have any questions or would like to discuss this project, feel free to reach out:

- LinkedIn: [LinkedIn](https://www.linkedin.com/in/fpriovolos/)
- Email: filippos.priovolos01@gmail.com

## 📝 Related Publication

This project is also the subject of a research paper that combines a theoretical and empirical approach. The paper dives into the details of the MLOps methodologies, techniques, and analysis involved in forecasting energy consumption with Azure Databricks 

- **Title**: "Streamlining MLOps for Energy Consumption Forecasting, A Case Study"
- **Authors**: Filippos Priovolos
If you use the content of this repository or the related paper in your research, please consider citing as shown in the citation section.


## 🧾 Citation

If you use this project in your research or want to refer to it, please attribute it as follows:

```bibtex
@misc{author2023energy,
  title={Streamlining MLOps for Energy Consumption Forecasting, A Case Study},
  author={Filippos Priovolos},
  year={2023},
}


