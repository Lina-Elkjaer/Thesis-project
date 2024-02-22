#Preprocessing
""" Takes dict of predictor df's, outcome df, and admissions df and creates appropriate data-splits """
import pandas as pd
from sklearn.model_selection import train_test_split
#from pandas import DataFrame



def data_split(df_outcome:pd.DataFrame, predictor_dict:dict, df_admissions:pd.DataFrame, test_size = 0.3):
    
    #Creating datasplit
    X_train, X_test, y_train, y_test = train_test_split(df_outcome['ID'], df_outcome['value'], test_size = test_size, random_state=42, stratify = df_outcome['value'])
    
    #Useful prints
    print(f"Number of observations in trainingset:{sum(y_train)}")
    print(f"Percentage of positive class in trainingset: {sum(y_train)/len(y_train)*100}") #27.15% UTI
    print(f"Number of observations in testset:{sum(y_test)}") #152
    print(f"Percentage of positive class in testset: {sum(y_test)/len(y_test)*100}") #27.14% UTI

    
    # Creating a df for uti positive within the training set 
    train_data = pd.DataFrame({'ID' : X_train, 'value' : y_train})
    train_data_sub = train_data[train_data['value'] == 1]
    y_train_df = pd.merge(train_data_sub, df_outcome, on=['ID', 'value'], how='left')

    # Creating a df for uti positive within the test set 
    test_data = pd.DataFrame({'ID' : X_test, 'value' : y_test}) #NB 'value' = outcome
    test_data_sub = test_data[test_data['value'] == 1] 
    y_test_df = pd.merge(test_data_sub, df_outcome, on=['ID', 'value'], how='left')

    X_train_df = pd.DataFrame({'ID' : X_train})


    #Creating a train subset for each predictor
    predictor_dict_train = {}
    
    for key in predictor_dict:
        predictor_dict_train[key] = pd.merge(X_train_df, predictor_dict[key], on=['ID'], how='left')


    #Creating a test subset for each predictor
    X_test_df = pd.DataFrame({'ID' : X_test})

    predictor_dict_test = {}
    
    for key in predictor_dict:
        predictor_dict_test[key] = pd.merge(X_test_df, predictor_dict[key], on=['ID'], how='left')


    #Creating a train subset of df_admissions based on the ID's from X_train 
    ID_list_train = list(X_train)
    df_admissions_train = df_admissions[df_admissions['ID'].isin(ID_list_train)]

    #Creating a test subset of df_admissions based on the ID's from X_test
    ID_list_test = list(X_test)
    df_admissions_test = df_admissions[df_admissions['ID'].isin(ID_list_test)]

    return predictor_dict_train, predictor_dict_test, y_train_df, y_test_df, df_admissions_train, df_admissions_test




      






