# -*- coding: utf-8 -*-

import numpy as np
import pandas
import scipy
import statsmodels.api as sm

"""
predictions(turnstile_weather) function takes in our pandas 
turnstile weather dataframe, and returns a set of predicted ridership values,
based on the other information in the dataframe.  

I am using this reference implementation: 
http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.linear_model.OLS.html

One of the advantages of the statsmodels implementation is that it gives you
easy access to the values of the coefficients theta. This can help you infer relationships 
between variables in the dataset.

Later I might experiment with polynomial terms as part of the input variables.  

The following links are useful: 
http://en.wikipedia.org/wiki/Ordinary_least_squares
http://en.wikipedia.org/w/index.php?title=Linear_least_squares_(mathematics)
http://en.wikipedia.org/wiki/Polynomial_regression
"""

def predictions(weather_turnstile): 
    '''
    I tried to implement the statsmodels package approach.  
    I used the same features as with gradient descent:  Hour, mintempi and precipi.  
    With gradient descent, R2 was about 0.46.  With the normal equation approach 
    with statsmodel.OLS, I keep getting R2 as 0.03 no matter what I do.  I must be 
    missing something, but it’s subtle because I don’t get an execution error traceback.  
    I’m giving up on this one.  
    
    WAIT I SOLVED IT.  :)  
    NEED TO ADD UNIT FEATURE TO X.  We did this in gradient descent.  
    This is basically adding it as a class via a set of dummy variables.
    This allows us to "pseudo-classify" the individual turnstiles as features.
    See pdf on dummy variables for a more thorough explanation.  
    '''
    Y = weather_turnstile['ENTRIESn_hourly']
    X = weather_turnstile[['precipi', 'Hour', 'mintempi']]
    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(weather_turnstile['UNIT'], prefix='unit')
    X = X.join(dummy_units)
    X = sm.add_constant(X)
    '''print X.head(10)
    print X.shape
    print Y.head(10)
    print Y.shape'''
    model = sm.OLS(Y,X)
    results = model.fit(method='qr')
    # print results.rsquared
    prediction = results.predict(exog=X)
    return prediction
