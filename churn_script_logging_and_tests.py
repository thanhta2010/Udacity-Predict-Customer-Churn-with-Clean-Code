# testing function doc string
'''
This file contains the tests functions to test functions inside
churn_libary.py

Author : Thanh Ta

Date : October 1, 2023
'''

import os
import logging
import pandas as pd
import churn_library as cls

logging.basicConfig(
    filename='./logs/test_churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist
with the other test functions
    '''
    try:
        bank_data_df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as error:
        logging.error("Testing import_eda: The file wasn't found %s",
                      error)
        raise error

    try:
        assert bank_data_df.shape[0] > 0
        assert bank_data_df.shape[1] > 0
    except AssertionError as error:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise error


def test_eda():
    '''
    test perform eda function
    '''
    data_frame = cls.import_data("./data/bank_data.csv")

    try:
        cls.perform_eda(data_frame)
        logging.info("Testing perform_eda: SUCCESS")
    except (AttributeError, SyntaxError) as error:
        logging.info("Testing perform_eda: Input should be a dataframe %s",
                     error)
        raise error

    try:
        assert os.path.isfile('./images/eda/churn_hist.png')
        assert os.path.isfile('./images/eda/churn_hist.png')
        assert os.path.isfile('./images/eda/customer_age_hist.png')
        assert os.path.isfile('./images/eda/customer_age_hist.png')
        assert os.path.isfile('./images/eda/marital_status_hist.png')
        assert os.path.isfile('./images/eda/total_transaction_hist.png')
        assert os.path.isfile('./images/eda/heatmap.png')
    except AssertionError as error:
        logging.error(
            "Testing perform_eda: at least one of the image files does not save.")
        raise error


def test_encoder_helper():
    '''
    test encoder helper
    '''
    data_frame = cls.import_data("./data/bank_data.csv")
    cls.perform_eda(data_frame)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    response = "Churn"

    try:
        encoder_dframe = cls.encoder_helper(data_frame, cat_columns, response)
        logging.info("Testing encoder_helper: SUCCESS")
    except (KeyError, NameError) as error:
        logging.error(
            "Testing encoding_helper: cat_columns not successfully encoded ,%s",
            error)
        raise error

    try:
        assert sum(encoder_dframe.columns.str.contains(
            '_' + response)) == len(cat_columns)
    except AssertionError as error:
        logging.error(
            "Testing encoder_helper: the length of columns of encoder_dframe\
            not match with the length of cat_columns")
        raise error


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
#perform_feature_engineering(data_df, response='Churn')
    bank_data_df = cls.import_data("./data/bank_data.csv")
    cls.perform_eda(bank_data_df)
    response = "Churn"

    try:
        (feature_columns, feature_train, feature_test, target_train,
         target_test) = cls.perform_feature_engineering(bank_data_df, response)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except (KeyError, NameError) as error:
        logging.error(
            "Testing perform_feature_engineering: failed with error=%s",
            error)
    try:
        assert isinstance(feature_columns, pd.DataFrame)
        assert isinstance(feature_train, pd.DataFrame)
        assert isinstance(feature_test, pd.DataFrame)
        assert isinstance(target_train, pd.Series)
        assert isinstance(target_train, pd.Series)
        assert isinstance(target_test, pd.Series)
    except AssertionError as error:
        logging.error(
            "Testing perform_feature_engineering: The type of some dataframe \
             or series incorrect.")
        raise error


def test_train_and_store_models():
    '''
    test train_and_store_models
    '''
    bank_data_df = cls.import_data("./data/bank_data.csv")
    cls.perform_eda(bank_data_df)
    response = "Churn"
    (feature_columns, feature_train, feature_test, target_train,
     target_test) = cls.perform_feature_engineering(bank_data_df, response)

    try:
        cls.train_and_store_models(
            feature_columns,
            feature_train,
            feature_test,
            target_train,
            target_test)
        logging.info("Testing train_and_store_models: SUCCESS")
    except Exception as error:
        logging.error(
            "Testing train_and_store_models: Failed with error=%s",
            error)
        raise error

    try:
        assert os.path.isfile('./models/logistic_regression_model.pkl')
        assert os.path.isfile('./models/random_forest_model.pkl')
        assert os.path.isfile(
            './images/results/Random_Forest_classification_report.png')
        assert os.path.isfile(
            './images/results/Logistic_Regression_classification_report.png')
        assert os.path.isfile(
            './images/results/rfc_lr_roc_curves.png')
        assert os.path.isfile(
            './images/results/rfc_feature_importance.png')
    except AssertionError as error:
        logging.error(
            "Testing test_train_and_store_models: At least either model or image file not saved.")
        raise error


if __name__ == "__main__":
    logging.info(
        "Invoke function: test_import to retrieve bank data from a file")
    print(
        "Invoke function: test_import to retrieve bank data from a file")
    # test_import(cls.import_data)
    test_import()

    logging.info(
        "Invoke function: test_eda to test perform_eda")
    print(
        "Invoke function: test_eda to test perform_eda")
    # test_eda(cls.perform_eda)
    test_eda()

    logging.info(
        "Invoke function: test_encoder_helper to test encoder_helper")
    print(
        "Invoke function: test_encoder_helper to test encoder_helper")
    # test_encoder_helper(cls.encoder_helper)
    test_encoder_helper()

    logging.info(
        "Invoke function: test_perform_feature_engineering")
    print(
        "Invoke function: test_perform_feature_engineering")
    # test_perform_feature_engineering(cls.perform_feature_engineering)
    test_perform_feature_engineering()

    logging.info(
        "Invoke function: test_train_and_store_models")
    print(
        "Invoke function: test_train_and_store_models")
    # test_train_and_store_models(cls.train_and_store_models)
    test_train_and_store_models()

    logging.info(
        "Test functions in module churn_library completed !")
    print(
        "Test functions in module churn_library completed !")
