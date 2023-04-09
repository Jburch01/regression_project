import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from env import get_db_url

from sklearn.model_selection import train_test_split
import sklearn.preprocessing




def get_zillow_data():
    '''
    Function will try to return ad database from csv file if file is local and in same directory.
    IF file doesn't exist it will create and store in same directory
    Otherwise will pull from codeup database.
    Must have credentials for codeup database.
    '''
    query = '''
        SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, fips, transactiondate
FROM properties_2017

LEFT JOIN propertylandusetype USING(propertylandusetypeid)

JOIN predictions_2017 using(parcelid)

WHERE (propertylandusetypeid  = 261
or
propertylandusetypeid  =  279)
and  transactiondate like %s;
        '''
    params = ('2017%',)
    try:
        csv_info = pd.read_csv('zillow.csv', index_col=0 )
        return csv_info
    except FileNotFoundError:
        url = get_db_url('zillow')
        info = pd.read_sql(query, url, params=params)
        info.to_csv("zillow.csv", index=True)
        return info
    
    
def remove_outliers(df, col_list, k=1.5):
    '''
    remove outliers from a dataframe based on a list of columns
    using the tukey method.
    returns a single dataframe with outliers removed
    '''
    col_qs = {}
    for col in col_list:
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
    for col in col_list:
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (k*iqr)
        upper_fence = col_qs[col][0.75] + (k*iqr)
        print(type(lower_fence))
        print(lower_fence)
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    return df


def prep_zillow():
    '''
    Funtion will return a zillow dataframe with the outliers and nulls dropped and the data cleaned
    '''
    df = get_zillow_data()
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                                   'bathroomcnt':'bathrooms', 
                                   'calculatedfinishedsquarefeet':'area',
                                   'taxvaluedollarcnt':'tax_value',
                                    'transactiondate': 'sale_date'})
    
    cols = [col for col in df.columns if col not in ['fips', 'year_built', 'sale_date']]
    
    df = remove_outliers(df, cols)
    return df



def wrangle_zillow():
    df = prep_zillow()
    train_val, test = train_test_split(df,
                                       train_size=0.8,
                                       random_state=706)
    train, validate = train_test_split(train_val,
                                       train_size=0.7,
                                       random_state=706)
    return train, validate, test



def scale_data(train, val, test):
    '''
    This function scales the data using the sklearn minmax scaler. Does not scale the target or categorical columns
    '''
    x_cols = [col for col in train.columns if col not in ['tax_value', 'fips', 'sale_date', 'bedrooms', 'bathrooms']]
    split = [train, val, test]
    scale_list= []
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train[x_cols])
    for cut in split:
        cut_copy = cut.copy()
        cut_copy[x_cols] = scaler.transform(cut_copy[x_cols])
        scale_list.append(cut_copy)

    
    return scale_list[0], scale_list[1], scale_list[2] 