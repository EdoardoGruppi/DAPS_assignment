# Dedicated file to store global variables
# This file can help to reduce hard_coding and to change only once variables in different files and functions.

alpha_vantage_api_key = 'DWT0NVZMA5D0V0TW'
base_dir = './Datasets'
# Company signature used within the Nasdaq stock market
company = 'MSFT'
# Extended name of the company selected
company_extended = 'Microsoft'
# The date that defines the beginning of the observed period
starting_date = '2017-04-01'
# The date that defines the ending of the observed period. All the data obtained inside the period
# [starting_date, ending_date] is used to train and validate the model.
ending_date = '2020-04-30'
starting_test_period = '2020-05-01'
# The date that defines the ending of the entire period. This allows to collect also the data related to the
# tested period in order to make comparisons between the predictions made by the model and the true values.
ending_test_period = '2020-05-31'
