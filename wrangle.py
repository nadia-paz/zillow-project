import os
import pandas as pd
import numpy as np

# ignore warnings
#import warnings
#warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.linear_model import LinearRegression

from env import get_db_url

########## GLOBAL VARIABLES ##########

# random state seed
seed = 2912
# target varibles -> predicting home prices
target = 'home_value'

########## FUNCTIONS ##################

def get_zillow():
    '''
    this function calls other functions to:
    acquire zillow data
    rename columns
    transform data types and create new columns
    '''
    df = acquire_zillow()
    rename_columns(df)
    df = transform_columns(df)
    df = handle_outliers(df)
    df.drop_duplicates(inplace=True) # removes 131 rows
    return df

def acquire_zillow():
    '''
    acuires data from codeup data base
    returns a pandas dataframe with
    'Single Family Residential' properties of 2017
    from zillow
    '''
    
    filename = 'zillow.csv'
    sql = '''
    SELECT bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, \
        fips, lotsizesquarefeet, poolcnt, \
        yearbuilt, taxvaluedollarcnt
    FROM properties_2017
    JOIN predictions_2017 USING(parcelid)
    JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE propertylandusedesc='Single Family Residential' AND transactiondate LIKE '2017%%'
    '''

    url = get_db_url('zillow')
    
    # if csv file is available locally, read data from it
    if os.path.isfile(filename):
        df = pd.read_csv(filename) 
    
    # if *.csv file is not available locally, acquire data from SQL database
    # and write it as *.csv for future use
    else:
        # read the SQL query into a dataframe
        df =  pd.read_sql(sql, url)
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index_label = False)
        
    return df

def rename_columns(df):
    '''
    the function renames columns
    '''
    df.rename(columns={
    'bedroomcnt':'bedrooms',
    'bathroomcnt':'bathrooms',
    'calculatedfinishedsquarefeet':'sq_feet',
    'lotsizesquarefeet':'lot_sqft',
    'taxvaluedollarcnt':'home_value',
    'yearbuilt':'year_built',
    'poolcnt':'pools',
    }, inplace=True)

def transform_columns(df):
    '''
    the function makes a first step into preparation the data for the modeling
    it changes data types, replaces pools null values with 0 
    and drops the rest of null values
    creates new columns: house age, county_name, LA (1:0), Ventura(1:0)
    '''
    # transform fips to integer
    df['fips'] = df.loc[:, 'fips'].astype(int)
    #remove null values   
    #df = df.dropna()

    # add a new column with county names
    df['county_name'] = np.select([(df.fips == 6037), (df.fips == 6059), (df.fips == 6111)],
                             ['LA', 'Orange', 'Ventura'])
    # column to category data type
    df['county_name'] = df.loc[:, 'county_name'].astype('category', copy=False)

    # replace NaN with 0 in the pools columns
    df['pools'] = df.pools.replace({np.NAN:0})

    
    # drop null values
    df = df.dropna(axis=0)
    
    # create a column 'house_age'
    df['house_age'] = 2017 - df.year_built

    # change the type of bedrooms/sq_feet/home_value/pools to integer
    df.loc[:,'bedrooms'] = np.array(df['bedrooms'].values, dtype='uint8')
    df.loc[:,'bathrooms'] = np.array(df['bathrooms'].values, dtype='uint8')
    df.loc[:,'year_built'] = np.array(df['year_built'].values, dtype=int)
    df.loc[:,'sq_feet'] = np.array(df['sq_feet'].values, dtype=int)
    df.loc[:,'lot_sqft'] = np.array(df['lot_sqft'].values, dtype=int)
    df.loc[:,'home_value'] = np.array(df['home_value'].values, dtype=int)
    df.loc[:,'house_age'] = np.array(df['house_age'].values, dtype='uint8')
    df.loc[:,'pools'] = np.array(df['pools'].values, dtype='uint8')

    #rearange columns and drop 'fips'
    cols = ['bedrooms',
            'bathrooms',
            'sq_feet',
            'lot_sqft',
            'year_built',
            'house_age',
            'pools',
            'county_name',
            'home_value']
    
    df = df[cols]
    
    return df

def handle_outliers(df):

    '''
    the function removes outliers from the data set
    '''
    # remove sq_feet below quantile 0.99
    q_sqft = df.sq_feet.quantile(0.99)
    df = df[df.sq_feet < q_sqft]
    df = df[df.sq_feet >= 300]
    #df = df[df.sq_feet < 10_000]
    
    # remove bathrooms below quantile 0.99
    #q_bed = df.bedrooms.quantile(0.99)
    #df = df[df.bedrooms < q_bed]
    df = df[df.bedrooms < 7]

    # remove bedrooms below quantile 0.99
    #q_bath = df.bathrooms.quantile(0.99)
    #df = df[df.bathrooms < q_bath] 
    df = df[df.bathrooms < 7] 

    # remove bedrooms= 0 and bathrooms = 0, 69 rows
    df = df[df.bathrooms != 0]
    df = df[df.bedrooms != 0]

    # remove home values outliers
    # can not remvoe minimum outliers. way too many values
    #q = df.home_value.quantile(0.01)
    #df = df[df.home_value > q] 
    df = df[df.home_value < 2_000_000] # 519

    # remove lot_sqft below quantile 0.99
    #q_lot = df.lot_sqft.quantile(0.99)
    #df = df[df.lot_sqft < q_lot]

    return df

################## PRE-PROCESSING #############
def dummies(df):
    '''
    create dummy variables for LA and Ventura
    '''
    # create dummies for LA and Ventura
    df['Orange'] = np.where(df.county_name == 'Orange', 1, 0)
    df['Ventura'] = np.where(df.county_name == 'Ventura', 1, 0)
    df.drop(columns=['county_name', 'year_built'], inplace=True)

def full_split_zillow(df):
    '''
    the function accepts a zillow data frame a 
    '''
    train, validate, test = split_zillow(df)
    #save target column
    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]

    #remove target column from the sets
    train.drop(columns = target, inplace=True)
    validate.drop(columns = target, inplace=True)
    test.drop(columns = target, inplace=True)

    return train, validate, test, y_train, y_validate, y_test

################### SCALING FUNCTIONS #####################

def standard_scale_zillow(train, validate, test):
    '''
    accepts train, validate, test data sets
    scales the data in each of them
    returns transformed data sets
    '''

    col = ['bedrooms', 'bathrooms', 'sq_feet', 'lot_sqft', 'house_age']
    
    # create scalers
    scaler = StandardScaler()    
    #qt = QuantileTransformer(output_distribution='normal')
    scaler.fit(train[col])
    train[col] = scaler.transform(train[col])
    validate[col] = scaler.transform(validate[col])
    test[col] = scaler.transform(test[col])
    
    return train, validate, test

def scale_zillow_quantile(train, validate, test):
    '''
    accepts train, validate, test data sets
    scales the data in each of them
    returns transformed data sets
    '''
    #count_columns = ['bedroomcnt', 'bathroomcnt']
    
    #col = train.columns[1:-1]
    col = ['bedrooms', 'bathrooms', 'sq_feet', 'lot_sqft', 'house_age']
    
    # create scalers
    #min_max_scaler = MinMaxScaler()    
    qt = QuantileTransformer(output_distribution='normal')
    qt.fit(train[col])
    train[col] = qt.transform(train[col])
    validate[col] = qt.transform(validate[col])
    test[col] = qt.transform(test[col])
    
    return train, validate, test

############### FEATURE SELECTION FUCNTIONS ########
def select_kbest(X, y, k):
    '''
    the function accepts the X_train data set, y_train array and k-number of features to select
    runs the SelectKBest algorithm and returns the list of features to be selected for the modeling
    !KBest doesn't depend on the model
    '''
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X, y)
    return X.columns[kbest.get_support()].tolist()

def rfe(X, y, k):
    '''
    The function accepts the X_train data set, y_train array and k-number of features to select
    runs the RFE algorithm and returns the list of features to be selected for the modeling
    !RFE depends on the model.
    This function uses Linear regression
    '''
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    return X.columns[rfe.get_support()].tolist()

def rfe_model(X, y, model, k):
    '''
    The function accepts the X_train data set, y_train array,
    model (created with hyperparameters) and k-number of features to select
    runs the RFE algorithm and returns the list of features to be selected for the modeling
    '''
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    return X.columns[rfe.get_support()].tolist()

############### SPLIT FUCNTIONS ########
def split_zillow(df):
    '''
    This function takes in a dataframe and splits it into 3 data sets
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    #split_db class verision with random seed
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    return train, validate, test

def full_split3_zillow(train, validate, test, target):
    '''
    accepts train, validate, test data sets and the name of the target variable as a parameter
    splits the data frame into:
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    #train, validate, test = train_validate_test_split(df, target)

    #save target column
    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]

    #remove target column from the sets
    train.drop(columns = target, inplace=True)
    validate.drop(columns = target, inplace=True)
    test.drop(columns = target, inplace=True)

    return train, validate, test, y_train, y_validate, y_test