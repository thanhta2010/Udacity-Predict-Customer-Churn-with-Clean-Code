# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The objective of this project is to implement ML Models (Logistic Regression and Random Forest) to identify credit card customers that are most likely to churn. This completed project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). 
The module: churn_library.py and the testing file: churn_script_logging_and_tests.py both score above 9.9 using pylint clean code module

The dataset for this project, which was pulled from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code)

This project follows the following steps:
* Load the dataset from Kaggle which is over 10k records of credit card customer data
* Exploratory Data Analysis (EDA) : Explore data features and analyze correlations between features in data
* Feature Engineering : This step consist of: one-hot-encoding categorical features, standardization and resulting into 19 features
* Modeling :
  * Train two classification models (sklearn random forest and logistic regression)
  * Hyper-parameter tuning with l2 regularization: “lbfgs” for logistic regression classification model
  * Hyper-parameter tuning using GridSearchCV with 5-fold cross validation for Random Forest classification model
  * Model evaluation using ROC curves and store "ROC curves" graph for further evaluation
* Identify most important features influencing the predictions and store in the "feature importance" graph
* Save best models with their performance metrics

## Files and data description
The files and directories in the root directory are organized as the following:
* Project files
  * **churn_notebook.ipynb**: the notebook is provided to experiment, analyze the data, then build and evaluate ML models
  * **churn_library.py**: the codes in this library module is refactored following PEP8 coding and engineering best practices so it can become a production-ready module
  * **churn_script_logging_and_tests.py**: this testing file is used to test the library module: churn_library.py with unit testing, logging, and best coding practices

* Folders
  * **data** : stores bank_data.csv which is a 10k-records of credit card customer data
  * **images** 
    *  **eda** : stores histogram graphs generated when conducting the EDA process 
    *  **results**: stores the graphs of classification reports, ROC curves and features importance
  * **logs**: stores the log files generated when testing library module: churn_library.py
  * **model** stores the models in pkl format 
## Running Files
* This project can be run in python 3.6 or 3.8. The dependent modules are specified in requirements_py3.6.txt or requirements_py3.8.txt
* To run the project: python churn_library.py or run interactively cell by cell inside the notebook: churn_notebook.ipynb
* To test the functions in the library module: python churn_script_logging_and_tests.py
* All of the code executions (except in the notebook) are generated logs in log file: churn_library.log




