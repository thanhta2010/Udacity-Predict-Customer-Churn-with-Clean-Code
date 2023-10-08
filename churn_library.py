# library doc string
'''
This module contains functions to build ML model and store artifacts
for predicting customer churn following the PEP 8 guidelines.

Author : Thanh Ta

Date : October 1, 2023
'''

# import libraries
import os
import logging

from sklearn.metrics import plot_roc_curve, classification_report
#from sklearn.metrics import roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import normalize

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from utils import setup_logger

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

#logger = setup_logger('lib_logger', './logs/churn_library.log')

def import_data(path):
    '''
    returns dataframe for the csv found at path

    input:
            path: a path to the csv file
    output:
            csv_df: pandas dataframe
    '''

    # Read the csv file from the path and convert to pandas dataframe
    logging.info("Enter function: import_data")
    csv_df = pd.read_csv(path)
    logging.info("Return data frame: csv_df")
    return csv_df


def perform_eda(eda_df):
    '''
    perform eda on data_df and save figures to images folder
    input:
            data_df: pandas dataframe

    output:
            None
    '''

    # copy and modify data framme to include "Churn" column
    logging.info("Enter function: perform_eda")

    # Create a Churn column
    eda_df['Churn'] = eda_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))

    # Churn Histogram
    churn_hist = eda_df['Churn'].hist()
    logging.info("Save image: churn_hist.png")
    churn_hist.figure.savefig('./images/eda/churn_hist.png')

    # Customer Age Histogram
    customer_age_hist = eda_df['Customer_Age'].hist()
    logging.info("Save image: customer_age_hist.png")
    customer_age_hist.figure.savefig('./images/eda/customer_age_hist.png')

    # Marital Status Histogram
    marital_status_hist = eda_df.Marital_Status.value_counts(
        'normalize').plot(kind='bar')
    logging.info("Save image: marital_status_hist.png")
    marital_status_hist.figure.savefig('./images/eda/marital_status_hist.png')

    # Total Transaction Histogram
    total_transaction_hist = sns.histplot(
        eda_df['Total_Trans_Ct'], stat='density', kde=True)
    logging.info("Save image: total_transaction_hist.png")
    total_transaction_hist.figure.savefig(
        './images/eda/total_transaction_hist.png')

    # Heatmap
    heatmap = sns.heatmap(
        eda_df.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    logging.info("Save image: heatmap.png")
    heatmap.figure.savefig('./images/eda/heatmap.png')

    plt.close()


def encoder_helper(encoder_df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            encoder_df: pandas dataframe with new columns for
    '''

    # copy and modify data framme to include new columns with propotion of
    # churn for each category
    logging.info("Enter function: encoder_helper")

    # Category encoded columns
    for category in category_lst:
        category_churn_column_lst = []
        category_churn_groups = encoder_df.groupby(category).mean()[response]
        #category_churn_groups = data_df.groupby(category).mean()['Churn']
        for val in encoder_df[category]:
            category_churn_column_lst.append(category_churn_groups.loc[val])

        encoder_df[category + '_' + response] = category_churn_column_lst

    logging.info("Return data frame: encoder_df")
    return encoder_df


def perform_feature_engineering(data_df, response='Churn'):
    '''
    input:
              data_df: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables or index y column]

    output:
              X: X dataframe having feature columns
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    logging.info("Enter function: perform_feature_engineering")

    # Defining the categorical Columns
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    logging.info("Invoke function: encoder_helper")
    encoder_dframe = encoder_helper(data_df, cat_columns, response)

    # target column
    # y = data_df['Churn']
    y_target = data_df[response]

    # features want to keep
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    # create an empty data frame
    x_features = pd.DataFrame()

    # Feature columns
    x_features[keep_cols] = encoder_dframe[keep_cols]

    # train test split
    logging.info("Invoke function: train_test_split")
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_target, test_size=0.3, random_state=42)

    logging.info("return: X_train, X_test, y_train, y_test")
    return (x_features, x_train, x_test, y_train, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    logging.info("Enter function: classification_report_image")

    # Random Forest Classification Report for Train Data
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('                   '),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')

    # Random Forest Classification Report for Test Data
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    logging.info("Save image: Random_Forest_classification_report.png")
    plt.savefig('./images/results/Random_Forest_classification_report.png')
    plt.close()

    # Logistic Regression Classification Report for Train Data
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('                        '),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')

    # Logistic Regression Classification Report for Test Data
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    logging.info("Save image: Logistic_Regression_classification_report.png")
    plt.savefig('./images/results/Logistic_Regression_classification_report.png')
    plt.close()


def roc_curve_image(rfc_model, lr_model, x_test, y_test):
    '''
    create roc curve graph and save them
    input:
              rfc_model: model object random forest
              lr_model: model object logistic regression
              x_test: X testing data
              y_test: y testing data
    output:
              None
    '''

    logging.info("Enter function: roc_curve_image")
    # roc_curve plots
    plt.figure(figsize=(15, 8))
    axes = plt.gca()
    plot_roc_curve(
        rfc_model.best_estimator_,
        x_test,
        y_test,
        ax=axes,
        alpha=0.8)
    plot_roc_curve(lr_model, x_test, y_test, ax=axes, alpha=0.8)
    logging.info("Save image: rfc_lr_roc_curves.png")
    axes.figure.savefig('./images/results/rfc_lr_roc_curves.png')
    # plt.show()
    plt.close()


def shap_tree_explainer_image(rfc_model, x_test):
    """
    Input:
            rfc_model: The model of Random Forest
            x_test: The test pandas dataframe
    Output:
            None
    """

    logging.info("Enter function: shap_tree_explainer_image")
    explainer = shap.TreeExplainer(rfc_model.best_estimator_)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    logging.info("Save image: rfc_shap_tree_explainer.png")
    plt.savefig("./images/results/rfc_shap_tree_explainer.png")
    plt.close()


# Take out output_pth: path to store the figure and use the default path
def feature_importance_plot(rfc_model, x_data):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values

    output:
             None
    '''

    logging.info("Enter function: feature_importance_plot")
    # Calculate feature importances
    importances = rfc_model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    logging.info("Save image: rfc_feature_importance.png")
    plt.savefig("./images/results/rfc_feature_importance.png")
    plt.close()


def train_and_store_models(x_dataframe, x_train_data, x_test_data,
                           y_train_data, y_test_data):
    '''
    train, store model results: images + scores, and store models
    input:
              x_dataframe: Data frame of X feature values
              x_train_data: X training data
              x_test_data: X testing data
              y_train_data: y training data
              y_test_data: y testing data
    output:
              None
    '''

    logging.info("Enter function: train_and_store_models")
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    logging.info("Invoke function: lrc.fit")
    lrc.fit(x_train_data, y_train_data)

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    logging.info("Invoke function: GridSearchCV")
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    logging.info("Invoke function: cv_rfc.fit")
    cv_rfc.fit(x_train_data, y_train_data)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train_data)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test_data)

    y_train_preds_lr = lrc.predict(x_train_data)
    y_test_preds_lr = lrc.predict(x_test_data)

    # save model.
    logging.info("Save model: logistic_regression_model.pkl")
    joblib.dump(lrc, './models/logistic_regression_model.pkl')
    logging.info("Save model: random_forest_model.pkl")
    joblib.dump(cv_rfc.best_estimator_, './models/random_forest_model.pkl')

    logging.info("Invoke function: classification_report_image")
    classification_report_image(y_train_data,
                                y_test_data,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    logging.info("Invoke function: roc_curve_image")
    #roc_curve_image(rfc_model, lr_model, X_test, y_test)
    roc_curve_image(cv_rfc, lrc, x_test_data, y_test_data)

    # Skipping this function because getting warning:
    # This plugin does not support propagateSizeHints()
    # This plugin does not support raise()
    # The function runs forever
    #logging.info("Invoke function: shap_tree_explainer_image")
    #shap_tree_explainer_image(cv_rfc, X_test_data)

    logging.info("Invoke function: feature_importance_plot")
    feature_importance_plot(cv_rfc, x_dataframe)


if __name__ == "__main__":
    # Build Input data and run support functions in churn_library
    PATH = "./data/bank_data.csv"
    logging.info(
        "Invoke function: import_data to retrieve bank data from a file")
    print("Invoke function: import_data to retrieve bank data from a file")
    bank_data_df = import_data(PATH)

    logging.info("Invoke function: perform_eda on bank data frame")
    print("Invoke function: perform_eda on bank data frame")
    perform_eda(bank_data_df)

    logging.info(
        "Invoke function: perform_feature_engineering on bank data frame")
    print("Invoke function: perform_feature_engineering on bank data frame")
    # return (X, X_train, X_test, y_train, y_test)
    (feature_columns, feature_train, feature_test, target_train,
     target_test) = perform_feature_engineering(bank_data_df, response='Churn')

    logging.info("Invoke function: train_and_store_models")
    print("Invoke function: train_and_store_models")
    logging.info(
        "It will train and store LogisticRegression and RandomForest models, images + scores")
    train_and_store_models(feature_columns, feature_train, feature_test,
                           target_train, target_test)

    logging.info("churn_library module run as a main program completed !")
    print("churn_library module run as a main program completed !")
