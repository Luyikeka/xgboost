import numpy as np
import pandas as pd
import os
import pytz
import statsmodels.api as sm

import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split# Split the data into training and testing sets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split# Split the data into training and testing sets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


params = {
    'booster':  'gbtree', 
    'objective': 'reg:squarederror', 
    'learning_rate': 0.1, # high learning rate to start... 
    'n_estimators': 500,
    'max_depth' : 7,
    'subsample' : 0.55,
    'min_child_weight' : 22,
    'gamma' : 0,
    'colsample_bytree' : 0.8,
    'eval_metric': 'rmse',
    'n_jobs':-1,
    'tree_method':'exact'
}
xgb_reg = xgb.XGBRegressor(**params, seed = 20)


param_grid = {
    'booster': ['gbtree'],
    'objective': ['reg:squarederror'],
    'learning_rate': [0.1],
    'n_estimators': [78],  # Equivalent to 'nrounds' in R
    'max_depth': [13],
    'subsample': [0.9],
    'colsample_bytree': [0.6],
    'seed': [1],
    'eval_metric': ['rmse'],
    'n_jobs': [-1],
    'tree_method': ['exact']
}


xgb_reg = xgb.XGBRegressor(**params)

###read data, data processing  
city = 'guangzhou'
path = '/home/yjzhang/ML/RFR_ozonetrend/data_pp/deseasonalized_fields-guangzhou(anthro3+MET+climate).csv'
df_ft = pd.read_csv(path)
dsn_daily_met = df_ft 

dsn_daily_met['deseasonalized_OFP'] = dsn_daily_met['deseasonalized_alkanes_OFP'] + dsn_daily_met['deseasonalized_alkenes_OFP'] + dsn_daily_met['deseasonalized_aromatics_OFP'] + \
                                      dsn_daily_met['deseasonalized_alkynes_OFP'] + dsn_daily_met['deseasonalized_OVOCs_OFP']
del dsn_daily_met['deseasonalized_alkanes_OFP']
del dsn_daily_met['deseasonalized_alkenes_OFP']
del dsn_daily_met['deseasonalized_aromatics_OFP']
del dsn_daily_met['deseasonalized_alkynes_OFP']
del dsn_daily_met['deseasonalized_OVOCs_OFP']
dsn_daily_met['date'] = pd.to_datetime(dsn_daily_met['date'], format='%Y-%m-%d')
years = ['2015','2016','2017','2018','2019',]

for year_for_test in years:
    print(year_for_test)
    test_dataset = dsn_daily_met[dsn_daily_met['date'].dt.year == int(year_for_test)]
    train_dataset = dsn_daily_met.drop(test_dataset.index)
    #--------------------------- fitting the model ---------------------------------------------
    cv = KFold(n_splits=4,random_state=None, shuffle=False) # 4 years left for training, 2015 to 2019 is 5-year, test is taken out for a year
    trainvalidationSplit = list(cv.split(train_dataset)).copy()  # you can see the index assigning to training and validation data
    # we can assign it following the tutorial from: https://stackoverflow.com/questions/46718017/k-fold-in-python


    date_for_train = dsn_daily_met.drop(test_dataset.index)
    date_for_train.index = np.arange(0,len(date_for_train.iloc[:,0]),1)
    date_for_train['year'] = pd.DatetimeIndex(date_for_train['date']).year

    
    
    len(date_for_train) == len(train_dataset)
    training_year = date_for_train['year'].unique()
    train_labels = train_dataset.pop('o3_raw')
    #train_labels = train_dataset.pop('deseasonalized_o3_raw')
    test_labels = test_dataset.pop('o3_raw')
    
    train_dataset.drop(columns=['Unnamed: 0'], inplace=True)
    test_dataset.drop(columns=['Unnamed: 0'], inplace=True)
    #train_dataset['Unnamed: 0'] = pd.to_numeric(train_dataset['Unnamed: 0'], errors='coerce')
    #test_dataset['Unnamed: 0'] = pd.to_numeric(test_dataset['Unnamed: 0'], errors='coerce')

    
    #test_labels = test_dataset.pop('deseasonalized_o3_raw')
    # assigning the index for cv split
    # because NA occurs in the dataset, we need to make sure that each fold contains one year of data
    del train_dataset['date']
    del test_dataset['date']
    for cvindex in np.arange(0,len(date_for_train['year'].unique()),1):
       # print(cvindex)
        validationYear = training_year[cvindex]
        validationix = np.array(date_for_train['year'][date_for_train['year'] == validationYear].index)
        trainingix = np.array(date_for_train['year'].drop(date_for_train['year'][date_for_train['year'] == validationYear].index).index)

        tu_obj = (trainingix,validationix)

        trainvalidationSplit[cvindex] = tu_obj
        
        # Perform grid search with only the specified parameters
        grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                                   cv=trainvalidationSplit, n_jobs=-1, verbose=2)

        grid_search.fit(train_dataset, train_labels)
    
        best_params = grid_search.best_params_
        
    
        final_model = xgb.XGBRegressor(**best_params)

    
    
    
        final_model.fit(train_dataset, train_labels)
    
        output = final_model.predict(train_dataset)
        #if 'date' in test_dataset.index.values.tolist(): del test_dataset['date']
        output_test = final_model.predict(test_dataset)

        #------------------------- making a dataframe -------------------------------------------------
        train_test = np.append(np.repeat("train",len(np.array(output))),np.repeat("test",len(np.array(output_test))))
        model_ozone = np.append(np.array(output),np.array(output_test))
        obs_ozone = np.append(np.array(train_labels),np.array(test_labels))
        inx = np.append(np.array(train_dataset.index),np.array(test_dataset.index)) # making a index, when we sample, the data is shuffled

        ozoneoutput = pd.DataFrame({'model_ozone':model_ozone,'obs_ozone':obs_ozone,"train_test":train_test,"inx":inx})



        ozoneoutput = ozoneoutput.sort_values(by=['inx'],ascending=True) # sort the index of the original data

        ozoneoutput.index = np.arange(0,len(ozoneoutput.index),1) # sort the index of the dataframe

        ozoneoutput['date'] =dsn_daily_met['date'] # attach date
        
        ozone_outputpath = '/home/yjzhang/ML/xgboost/'
        ozoneoutput.to_csv(
            
            
                ozone_outputpath+ city +"_"+str(year_for_test)+"_XGB_ozone_comparison.csv",
                header = True,
                sep = ",",index = False,encoding = 'UTF-8'
        
            
            
                )
    
    
        bestpara = pd.DataFrame(grid_search.best_params_.items())
        bestpara.columns = np.array(["para","value"])
    
    
        bestpara.to_csv(
            
            
                ozone_outputpath+city+"_"+str(year_for_test)+"_RFR_b.csv",
                header = True,
                sep = ",",index = False,encoding = 'UTF-8'
        
            
            
                )
    
    
        
        feature_list = train_dataset.columns.values
    
        importance = final_model.feature_importances_
    
        FItable = pd.DataFrame({'feature':feature_list,'score':importance})
    
        FItable.to_csv(
                ozone_outputpath+city+"_"+"testyear_"+str(year_for_test)+"_FItable.csv",
                header = True,
                sep = ",",index = False,encoding = 'UTF-8'
                )
    

# Function to just train and test (no tuning)
def run_model(xgb_reg,X_train,y_train,X_test,y_test):
    my_model = xgb_reg.fit(X_train,y_train)
    y_pred = xgb_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print('RMSE: ', rmse)
    print('R^2: ', r2)
    return y_pred,my_model

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBRegressor

# Define your hyperparameter grid for the grid search
param_grid = {
    'learning_rate': [0.1, 0.01], 
    'n_estimators': [100, 500],
    'max_depth': [7, 10],
    'subsample': [0.55, 0.8],
    'min_child_weight': [22, 30],
    'gamma': [0, 1],
    'colsample_bytree': [0.8, 1.0],
}

xgb_reg = XGBRegressor(booster='gbtree', objective='reg:squarederror', eval_metric='rmse', n_jobs=-1, tree_method='exact', seed=20)

# Read and process your data
path = '/home/yjzhang/ML/RFR_ozonetrend/deseasonalized_fields-tianjin(anthro3+MET+climate).csv'
df_ft = pd.read_csv(path)
dsn_daily_met = df_ft 
dsn_daily_met['date'] = pd.to_datetime(dsn_daily_met['date'], format='%Y-%m-%d')
years = ['2015', '2016', '2017', '2018', '2019']

for year_for_test in years:
    print(year_for_test)
    test_dataset = dsn_daily_met[dsn_daily_met['date'].dt.year == int(year_for_test)]
    train_dataset = dsn_daily_met.drop(test_dataset.index)

    # Perform grid search with only the specified parameters
    grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=4, n_jobs=-1, verbose=2)
    grid_search.fit(train_dataset, train_dataset['o3_raw'])  # Adjust the target variable accordingly

    best_params = grid_search.best_params_

    final_model = XGBRegressor(**best_params, booster='gbtree', objective='reg:squarederror', eval_metric='rmse', n_jobs=-1, tree_method='exact', seed=20)
    final_model.fit(train_dataset, train_dataset['o3_raw'])  # Adjust the target variable accordingly

    output = final_model.predict(train_dataset)
    output_test = final_model.predict(test_dataset)
    
    # Your code for feature-sample ratios can be added here
    
    # The rest of your code goes here

