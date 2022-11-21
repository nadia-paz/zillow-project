import pandas as pd
import numpy as np


from sklearn.preprocessing import  PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, explained_variance_score

# linear regressions
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor

# non-linear regressions
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import wrangle as wr



############## GLOBAL VARIABLES ###########
seed = 2912 # random seed for random_states

# get zillow data
df = wr.get_zillow()

# remove unneeded columns and add dummy variables for county_name
wr.dummies(df)

# create 3 data sets that keep the values of the counties
la = df[(df.Orange == 0) & (df.Ventura == 0)] # LA county
ventura = df[df.Ventura == 1] # Ventura county
orange = df[df.Orange == 1] # Orange county

# split the data into 3 data sets and 3 target arrays
X_train, X_validate, X_test, y_train, y_validate, y_test = wr.full_split_zillow(df)

# get scaled X_train, X_validate, X_test sets
# standard scaler
X1, X2, X3 = wr.standard_scale_zillow(X_train, X_validate, X_test)
# quantile scaler
XQ1, XQ2, XQ3 = wr.scale_zillow_quantile(X_train, X_validate, X_test)

# get a baseline value = median of the train set's target
baseline = y_train.median()


# create dataframes to keep predictions of train and validate data sets
predictions_train = pd.DataFrame(y_train)
predictions_validate = pd.DataFrame(y_validate)
predictions_train['baseline'] = baseline
predictions_validate['baseline'] = baseline

###### GLOBAL EVALUATION VARS ##########
# calculate baseline RMSE score
RMSE_baseline = round(mean_squared_error(y_train, predictions_train.baseline) ** .5)
# DataFrame to keep model's evaluations
scores = pd.DataFrame(columns=['model_name', 'features', 'scaling',
                               'RMSE_train', 'R2_train', 'RMSE_validate', 
                               'R2_validate', 'RMSE_difference'])

# create a dictionary of regression models
models = {
    'Linear Regression': LinearRegression(),
    'Generalized Linear Model': TweedieRegressor(power=2, alpha = 0.5),
    'Gradient Boosting Regression': GradientBoostingRegressor(random_state=seed),
    'Decision Tree Regression': DecisionTreeRegressor(max_depth=4, random_state=seed),
    'Random Forest Regression':RandomForestRegressor(max_depth=4, random_state=seed),
    'LassoLars Regression':LassoLars(alpha=0.1)
    }

# create lists of features and save them in the dictionary
f1 = ['bedrooms', 'bathrooms', 'sq_feet'] # same as select_kbest with k=3
f2 = ['bedrooms', 'bathrooms']
f3 = ['bedrooms','bathrooms','sq_feet', 'pools']
f4 = ['bathrooms','sq_feet', 'pools']
f5 = ['bedrooms','bathrooms','sq_feet','house_age','pools','Orange','Ventura']
f6 = wr.select_kbest(X_train, y_train, 4)
f7 = wr.select_kbest(X_train, y_train, 2)
f8 = X_train.columns.tolist()

# create a dictionary with features
features = {
    'f1':f1,
    'f2':f2,
    'f3':f3,
    'f4':f4,
    'f5':f5,
    'f6':f6,
    'f7':f7,
    'f8':f8
}

############### EVALUATION FUNCTIONS #############

def regression_errors(y_actual, y_predicted):
    '''
    this function accepts 
    y: actual results/array
    yhat: predictions/array
    k: feature size/integer
    calculates regression scores based on the baseline being median
    returns RMSE and adjacted R2
    '''
    # root mean squared error score
    RMSE = mean_squared_error(y_actual, y_predicted) ** .5
    # adjucted R^2 score
    ADJR2 = explained_variance_score(y_actual, y_predicted)
    return round(RMSE), round(ADJR2, 2)

def regression_errors_median(y, yhat, k):
    '''
    this function accepts 
    y: actual results/array
    yhat: predictions/array
    k: feature size/integer
    calculates regression scores based on the baseline being median
    returns RMSE and adjacted R2
    '''
    # calculate predictions' residuals
    residual = yhat - y
    # calculate baseline's residuals
    residual_baseline = y - y.median()

    # sum of squared errors score
    SSE = (residual ** 2).sum()
    
    # total sum of squares score
    TSS = (residual_baseline ** 2).sum()
    
    # explained sum of squares score
    ESS = TSS - SSE
    
    # mean squared error score
    MSE = SSE/len(y)
    
    # root mean squared error score
    RMSE = MSE ** .5
    
    R2 = ESS / TSS
    
    ADJR2 = 1 - ( (1 - R2) * (len(y) - 1) / (len(y) - k - 1 )) 
    
    return round(RMSE), round(ADJR2, 2)

############### MODELING FUNCTIONS ###############

def run_model(X_train, X_validate, scaling):
    
    '''
    general function to run models with X_train and X_validate that were scaled
    '''

    for f in features:
        for key in models:
            # create a model
            model = models[key]
            # fit the model
            model.fit(X_train[features[f]], y_train)
            # predictions of the train set
            y_hat_train = model.predict(X_train[features[f]])
            # predictions of the validate set
            y_hat_validate = model.predict(X_validate[features[f]])
            # add train set predictions to the data frame
            predictions_train[key] = y_hat_train
            # add validate set predictions to the data frame
            predictions_validate[key] = y_hat_validate

            # calculate scores train set
            RMSE, R2 = regression_errors(y_train, y_hat_train)
            # calculate scores validation set
            RMSE_val, R2_val = regression_errors(y_validate, y_hat_validate)
            diff = np.abs(RMSE - RMSE_val)
            
            # add the score results to the scores Data Frame
            scores.loc[len(scores.index)] = [key, f, scaling, RMSE, R2, RMSE_val, R2_val, diff]

def run_model_standard():
    # runs regression models on the X_train scaled with StandardScaler()
    X1, X2, _ = wr.standard_scale_zillow(X_train, X_validate, X_test)
    run_model(X1, X2, 'standard')

def run_model_quantile():
    XQ1, XQ2, _ = wr.scale_zillow_quantile(X_train, X_validate, X_test)
    run_model(XQ1, XQ2, 'quantile')

def run_rfe():
    '''
    The function accepts the X_train data set, y_train array and k-number of features to select
    runs the RFE algorithm and returns the list of features to be selected for the modeling
    !RFE depends on the model.
    This function uses Linear regression
    '''
    # scale the data
    #X1, X2, _ = wr.standard_scale_zillow(X_train, X_validate, X_test)
    
    for key in models:
        # create a model
        model = models[key]
        
        # create a RFE feature selector
        rfe = RFE(model, n_features_to_select=4)
        rfe.fit(X1, y_train)
        
        # get the optimal features for every particular model
        f = X1.columns[rfe.get_support()].tolist()
        
        # fit the model with RFE features
        model.fit(X1[f], y_train)
        # predictions of the train set
        y_hat_train = model.predict(X1[f])
        # predictions of the validate set
        y_hat_validate = model.predict(X2[f])
        # add train set predictions to the data frame
        col_name = str(key)+'_rfe'
        predictions_train[col_name] = y_hat_train
        # add validate set predictions to the data frame
        predictions_validate[col_name] = y_hat_validate

        # calculate scores train set
        RMSE, R2 = regression_errors(y_train, y_hat_train)
        # calculate scores validation set
        RMSE_val, R2_val = regression_errors(y_validate, y_hat_validate)
        diff = np.abs(RMSE - RMSE_val)

        # add the score results to the scores Data Frame
        scores.loc[len(scores.index)] = [key, 'rfe', 'standard', RMSE, R2, RMSE_val, R2_val, diff]

def run_polynomial():

    
    for i in range(1,5):
        # features[f] gives an access to the list of features in the dictionary
        #length = len(features[f])
        # create a Polynomial feature transformer
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly.fit(X1.iloc[:, :i])
        # create a df with transformed features of the train set
        X1_poly = pd.DataFrame(
            poly.transform(X1.iloc[:, :i]),
            columns=poly.get_feature_names(X1.iloc[:, :i].columns),
            index=X1.index)
        X1_poly = pd.concat([X1_poly, X1.iloc[:, i:]], axis=1)
        #X1_poly = pd.concat([X1_poly, X1], axis=1)
        
        #display(X1_poly.head(1)) #testing the columns
        
        # create a df with transformed features for the validate set
        X2_poly = pd.DataFrame(
            poly.transform(X2.iloc[:, :i]),
            columns=poly.get_feature_names(X2.iloc[:, :i].columns),
            index=X2.index)
        X2_poly = pd.concat([X2_poly, X2.iloc[:, i:]], axis=1)
        #X2_poly = pd.concat([X2_poly, X2], axis=1)
                             
        feature_name = 'poly'+str(i)
        
        for key in models:
            # create a model
            model = models[key]
            # fit the model
            model.fit(X1_poly, y_train)
            # predictions of the train set
            y_hat_train = model.predict(X1_poly)
            # predictions of the validate set
            y_hat_validate = model.predict(X2_poly)
            # add train set predictions to the data frame
            predictions_train[key] = y_hat_train
            # add validate set predictions to the data frame
            predictions_validate[key] = y_hat_validate

            # calculate scores train set
            RMSE, R2 = regression_errors(y_train, y_hat_train)
            # calculate scores validation set
            RMSE_val, R2_val = regression_errors(y_validate, y_hat_validate)
            diff = np.abs(RMSE - RMSE_val)
            
            # add the score results to the scores Data Frame
            scores.loc[len(scores.index)] = [key, feature_name, 'standard', RMSE, R2, RMSE_val, R2_val, diff]
def run_single():
    # create a list ['bedrooms', 'bathrooms', 'sq_feet', 'lot_sqft', 'house_age']
    single_corr = X1.iloc[:, :-3].columns.tolist()

    # for every single feature in the list
    for f in single_corr:
        # create a linear regression model
        model = LinearRegression()
        # fit the model
        model.fit(X1[[f]], y_train)
        # predictions of the train set
        y_hat_train = model.predict(X1[[f]])
        # predictions of the validate set
        y_hat_validate = model.predict(X2[[f]])
        # add train set predictions to the data frame
        predictions_train['single'] = y_hat_train
        # add validate set predictions to the data frame
        predictions_validate['single'] = y_hat_validate

        # calculate scores train set
        RMSE, R2 = regression_errors(y_train, y_hat_train)
        # calculate scores validation set
        RMSE_val, R2_val = regression_errors(y_validate, y_hat_validate)
        diff = np.abs(RMSE - RMSE_val)

        # add the score results to the scores Data Frame
        scores.loc[len(scores.index)] = ['Single Linear Regression', f, 'standard', RMSE, R2, RMSE_val, R2_val, diff]

def run_all_models():
    '''
    the function runs all models and saves the results to csv file
    '''
    run_model_standard()
    run_model_quantile()
    run_rfe()
    run_polynomial()
    run_single()
    scores.to_csv('regression_results.csv')

############# SELECT AND RUN THE BEST MODEL #############

def select_best_model(scores):
    # select top 20 models based on the RMSE score of the train set
    top_20 = scores.sort_values(by='RMSE_train').head(20)
    # select top 5 models based on the RMSE score of the validate set
    top_5 = top_20.sort_values(by=['RMSE_validate']).head(5)
    # display top 5 models
    display(top_5)
    # select the best model with the smallest difference in the RMSE scores
    best_model = top_5.sort_values(by='RMSE_difference').head(1)
    return best_model

def run_best_model():
    '''
    the function runs the best model on the train, test and validate data sets 
    and returns scores in the data frame
    '''
    # create a data frame for test set results
    predictions_test = pd.DataFrame(y_test)
    predictions_test['baseline'] = baseline

    i = 2

    # create a Polynomial feature transformer
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly.fit(X1.iloc[:, :i])
    # create a df with transformed features of the train set
    X1_poly = pd.DataFrame(
        poly.transform(X1.iloc[:, :i]),
        columns=poly.get_feature_names(X1.iloc[:, :i].columns),
        index=X1.index)
    X1_poly = pd.concat([X1_poly, X1.iloc[:, i:]], axis=1)
    #X1_poly = pd.concat([X1_poly, X1], axis=1)

    #display(X1_poly.head(1)) #testing the columns

    # create a df with transformed features for the validate set
    X2_poly = pd.DataFrame(
        poly.transform(X2.iloc[:, :i]),
        columns=poly.get_feature_names(X2.iloc[:, :i].columns),
        index=X2.index)
    X2_poly = pd.concat([X2_poly, X2.iloc[:, i:]], axis=1)


    # create a df with transformed features for the test set
    X3_poly = pd.DataFrame(
        poly.transform(X3.iloc[:, :i]),
        columns=poly.get_feature_names(X3.iloc[:, :i].columns),
        index=X3.index)
    X3_poly = pd.concat([X3_poly, X3.iloc[:, i:]], axis=1)

    # create a Gradient Boosting Regression model
    model = GradientBoostingRegressor()
    # fit the model
    model.fit(X1_poly, y_train)
    # predictions of the train set
    y_hat_train = model.predict(X1_poly)
    # predictions of the validate set
    y_hat_validate = model.predict(X2_poly)
    # add train set predictions to the data frame
    y_hat_test = model.predict(X3_poly)
    predictions_test['predictions'] = y_hat_test

    # calculate scores train set
    RMSE_train, R2_train = regression_errors(y_train, y_hat_train)
    # calculate scores validation set
    RMSE_val, R2_val = regression_errors(y_validate, y_hat_validate)
    # calculate scores test set
    RMSE_test, R2_test = regression_errors(y_test, y_hat_test)
    RMSE_bl, _ = regression_errors(y_test, predictions_test.baseline)
    
    # save final score into a dictionary
    res = {
        'Features': 'poly: ' + str(X3.iloc[:, :i].columns.tolist()),
        'RMSE Baseline' : RMSE_bl,
        'RMSE Train Set': RMSE_train,
        'RMSE Validation Set':RMSE_val,
        'RMSE Test Set':RMSE_test,
        'R2 Train Set':R2_train,
        'R2 Validation Set':R2_val,
        'R2 Test':R2_test,
        'Beats a basline by:':str(f'{round((RMSE_bl - RMSE_test) / RMSE_bl * 100, 1)}%')
    }

    # add the score results to the scores Data Frame
    final_test = pd.DataFrame({'Gradient Bosting Regression': list(res.keys()), 'Scores': list(res.values())})

    return final_test