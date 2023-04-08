import pandas as pd
import numpy as np
import wrangle as wr
import matplotlib.pyplot as plt
import seaborn as sns

# modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score

import warnings
warnings.filterwarnings("ignore")


def get_target_and_features(train, val, test):
    x_cols = ['tax_value', 'fips', 'sale_date']
    split_df = [train, val, test]
    x_y = []
    for split in split_df:
        x_split = split.drop(columns=x_cols)
        y_split = pd.DataFrame(split.tax_value)
        x_y.append(x_split)
        x_y.append(y_split)
               
    return x_y[0], x_y[1], x_y[2], x_y[3], x_y[4], x_y[5]
        

    
def get_baseline(y_train, y_val):
    tax_value_pred_mean = y_train.tax_value.mean()
    y_train['tax_value_baseline'] = tax_value_pred_mean
    y_val['tax_value_baseline'] = tax_value_pred_mean

    
def get_distribution_and_corr(train):    
    corr = train.drop(columns=['sale_date', 'fips']).corr(method='spearman')
    cols = ['tax_value', 'bedrooms', 'bathrooms', 'area']
    sns.heatmap(corr,
       cmap='Purples',
       annot=True,
       mask=np.triu(corr))
    plt.show()
    plt.figure(figsize=(16, 3))
    for i, col in enumerate(cols):
        # i starts at 0, but plot nos should start at 1
        subplot_num = i+1
        # Create subplot.
        plt.subplot(1,4,subplot_num)
        # Title with column name.
        plt.title(col)
        # Display histogram for column.
        train[col].hist(bins=5)
        # Hide gridlines.
        plt.grid(False)
    
        

def stats_test(col, train):
    corr = train.drop(columns=['sale_date', 'fips']).corr(method='spearman')