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

from scipy.stats import pearsonr, spearmanr

import warnings
warnings.filterwarnings("ignore")

metric_df = {}
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
        

    
def get_baseline(y_train, y_val, s=0):
    '''
    Adds a baseline prediction for the y value using the mean
    '''
    tax_value_pred_mean = y_train.tax_value.mean()
    y_train['tax_value_baseline'] = tax_value_pred_mean
    y_val['tax_value_baseline'] = tax_value_pred_mean
    
    rmse_train_mu = mean_squared_error(y_train.tax_value,
                                y_train.tax_value_baseline) ** .5

    rmse_validate_mu = mean_squared_error(y_val.tax_value, y_val.tax_value_baseline) ** (0.5)
    if s == 0:
        print(f"""RMSE using Mean
        Train/In-Sample: {round(rmse_train_mu, 2)} 
        Validate/Out-of-Sample: {round(rmse_validate_mu, 2)}""")
    else:
        global metric_df
        metric_df = pd.DataFrame(data=[
        {'model': 'mean_baseline',
         'RMSE_train': rmse_train_mu,
         'RMSE_validate': rmse_validate_mu,
         'R2_validate': explained_variance_score(y_val.tax_value,
                                                 y_val.tax_value_baseline)
            }
        ]
        )




    
def lr(x_train, y_train, x_val, y_val, s=0):
    # MAKE THE THING: create the model object
    lm = LinearRegression()

    #1. FIT THE THING: fit the model to training data
    OLSmodel = lm.fit(x_train, y_train.tax_value)

    #2. USE THE THING: make a prediction
    y_train['tax_value_pred_lm'] = lm.predict(x_train)

    #3. Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm)**(1/2)
    
    # predict validate
    y_val['tax_value_pred_lm'] = lm.predict(x_val)


    # evaluate: RMSE
    rmse_validate = mean_squared_error(y_val.tax_value, y_val.tax_value_pred_lm)**(1/2)

    if s == 0:
        print(f"""RMSE for OLS using LinearRegression
        Training/In-Sample:  {rmse_train} 
        Validation/Out-of-Sample: {rmse_validate}""")
    else:
        global metric_df
        metric_df = metric_df.append(
    {'model': 'OLS Regressor',
     'RMSE_train': rmse_train,
     'RMSE_validate': rmse_validate,
     'R2_validate': explained_variance_score(y_val.tax_value,
                                             y_val.tax_value_pred_lm)
    }, ignore_index=True)



   
        
def lassolars(x_train, y_train, x_val, y_val, s=0):
    # MAKE THE THING: create the model object
    lars = LassoLars(alpha=0.01)

    #1. FIT THE THING: fit the model to training data
    # We must specify the column in y_train, since we have converted it to a dataframe from a series!
    lars.fit(x_train, y_train.tax_value)

    #2. USE THE THING: make a prediction
    y_train['tax_value_pred_lars'] = lars.predict(x_train)

    #3. Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lars) ** (1/2)

    #4. REPEAT STEPS 2-3

    # predict validate
    y_val['tax_value_pred_lars'] = lars.predict(x_val)

    # evaluate: RMSE
    rmse_validate = mean_squared_error(y_val.tax_value, y_val.tax_value_pred_lars) ** (1/2)
    
    if s == 0:
        print(f"""RMSE for Lasso + Lars
        _____________________
        Training/In-Sample: {rmse_train}, 
        Validation/Out-of-Sample:  {rmse_validate}
        Difference:  {rmse_validate - rmse_train}""")
        
    else:
        global metric_df
        metric_df = metric_df.append(
    {'model': 'Lars lasso',
     'RMSE_train': rmse_train,
     'RMSE_validate': rmse_validate,
     'R2_validate': explained_variance_score(y_val.tax_value,
                                             y_val.tax_value_pred_lars)
    }, ignore_index=True)
    
 




def tweedie(x_train, y_train, x_val, y_val, s=0):
    # MAKE THE THING: create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    #1. FIT THE THING: fit the model to training data
    # We must specify the column in y_train, since we have converted it to a dataframe from a series!
    glm.fit(x_train, y_train.tax_value)

    #2. USE THE THING: make a prediction
    y_train['tax_value_pred_glm'] = glm.predict(x_train)

    #3. Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_glm) ** (1/2)

    #4. REPEAT STEPS 2-3

    # predict validate
    y_val['tax_value_pred_glm'] = glm.predict(x_val)

    # evaluate: RMSE
    rmse_validate = mean_squared_error(y_val.tax_value, y_val.tax_value_pred_glm) ** (1/2)
    
    if s == 0:
        print(f"""RMSE for GLM using Tweedie, power=1 & alpha=0
        _____________________
        Training/In-Sample: {rmse_train}, 
        Validation/Out-of-Sample:  {rmse_validate}
        Difference:  {rmse_validate - rmse_train}""")
        
    else:
        global metric_df
        metric_df = metric_df.append(
    {'model': 'Tweedie Regressor',
     'RMSE_train': rmse_train,
     'RMSE_validate': rmse_validate,
     'R2_validate': explained_variance_score(y_val.tax_value,
                                             y_val.tax_value_pred_glm)
    }, ignore_index=True)
        
 
def get_poly(x_train, y_train, x_val, y_val, x_test, y_test, s=0):
    #1. Create the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2) #quadratic aka x-squared

    #1. Fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(x_train)

    #1. Transform X_validate_scaled & X_test_scaled 
    X_validate_degree2 = pf.fit_transform(x_val)
    X_test_degree2 = pf.fit_transform(x_test)
    #2.1 MAKE THE THING: create the model object
    lm2 = LinearRegression()

    #2.2 FIT THE THING: fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.tax_value)

    #3. USE THE THING: predict train
    y_train['tax_value_pred_lm2'] = lm2.predict(X_train_degree2)

    #4. Evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm2) ** (1/2)

    #4. REPEAT STEPS 3-4

    # predict validate
    y_val['tax_value_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_val.tax_value, y_val.tax_value_pred_lm2) ** 0.5
    if s == 0:
        print(f"""RMSE for Polynomial Model, degrees=2
        _____________________________________________
        Training/In-Sample:  {rmse_train} 
        Validation/Out-of-Sample:  {rmse_validate}""")
        
    else:
        global metric_df
        metric_df = metric_df.append({
    'model': 'quadratic', 
    'RMSE_train': rmse_train,
    'RMSE_validate': rmse_validate,
    'R2_validate':explained_variance_score(y_val.tax_value,
                                           y_val.tax_value_pred_lm2)
    }, ignore_index=True)

    
def compare_models(x_train, y_train, x_val, y_val, x_test, y_test):
    global metric_df
    get_baseline(y_train, y_val, s=1)
    lr(x_train, y_train, x_val, y_val, s=1)
    lassolars(x_train, y_train, x_val, y_val, s=1)
    tweedie(x_train, y_train, x_val, y_val, s=1)
    get_poly(x_train, y_train, x_val, y_val, x_test, y_test, s=1)
    
    return metric_df
    
    
    
    
def test_model(x_train, y_train, x_val, y_val, x_test, y_test):
    lars = LassoLars(alpha=0.01)

    #1. FIT THE THING: fit the model to training data
    # We must specify the column in y_train, since we have converted it to a dataframe from a series!
    lars.fit(x_train, y_train.tax_value)

    #2. USE THE THING: make a prediction
    y_train['tax_value_pred_lars'] = lars.predict(x_train)

    #3. Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lars) ** (1/2)

    #4. REPEAT STEPS 2-3

    # predict validate
    y_val['tax_value_pred_lars'] = lars.predict(x_val)

    # evaluate: RMSE
    rmse_validate = mean_squared_error(y_val.tax_value, y_val.tax_value_pred_lars) ** (1/2)
    
    y_test['tax_value_pred_lars'] = lars.predict(x_test)


    # evaluate: RMSE
    rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lars)**(1/2)
    R2_test = explained_variance_score(y_test.tax_value,
                                               y_test.tax_value_pred_lars)

    print(f"""RMSE for OLS using LinearRegression
    Test_rmse:  {rmse_test} 
    R2: {R2_test}""")
    
    
    
    
    
def get_distribution_and_corr(train):
    '''
    Gets a heat map correlations and the 
    distributions of the features including the target
    requires a df as a param
    '''
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
    
        

def stats_test(x, y, df):
    '''
    Does a stats test on x and y with df as a param
    prints the results of the test and wether or not
    the null hypothesis
    '''
    r, p = spearmanr(df[x], df[y])
    if p < 0.05:
        print(f'We can reject our null hypothesis with a p-value: {p}')
        print(f'The correlation is {r}')
    else:
        print('We failed ro reject the null hypothesis!')
        
        
        
def get_visual(df, x, y, plot_type):
    '''
    Gets a visual using the dataframe, x, y and plot type
    '''
    plot_type(data=df, x=x, y=y)
    plt.xlable=x
    plt.ylable=y
    plt.show()
    

def get_regplot(df, x ,y ):
    sns.regplot(data=df, x=x, y=y, line_kws={'color': 'red'})
    plt.show()