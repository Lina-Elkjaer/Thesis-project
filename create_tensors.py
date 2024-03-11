import torch
import pandas as pd
import warnings
import math

warnings.filterwarnings("ignore", category=FutureWarning) # Doesnt really fix the issue that the function append will be deprecated later. however I have not been able to find solution.
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None # warning about overwriting, but I do it in purpose

def max_notes(X_df, df_admissions, look_back = 4):
    
    #Converting to date.time             
    df_admissions['date'] = pd.to_datetime(df_admissions['date'], format = "%Y-%m-%d %H:%M:%S")

    X_df['date'] = pd.to_datetime(X_df['date'], format = "%Y-%m-%d %H:%M:%S")

    #initiating max_length
    maximum_notes = 0

    ID_list = X_df['ID'].unique()

    for ID in ID_list:
        #creating subsets pr ID
        X_df_sub = X_df[X_df['ID'] == ID]
        df_admissions_sub = df_admissions[df_admissions['ID'] == ID]
    
        #Loop to split data into groups of 4 days 
        for i in range(len(df_admissions_sub['date'])):
            if i > (look_back-1):
                df_pred_days = df_admissions_sub[i-look_back: i] # -4 fordi vi starter ved i=0 
                date_list = []
                
                for date in X_df_sub['date']:

                    if  df_pred_days.iloc[0,0] <= date <  df_pred_days.iloc[-1,0]:
                        date_list.append(date)

                #subset containing 4 days of data for 1 patient 
                subset = X_df_sub[X_df_sub['date'].isin(date_list)]

                if len(subset) > maximum_notes:
                    maximum_notes = len(subset)
    
    return maximum_notes



def tensors(X_df, y_df, df_admissions, maximum_notes, predictor_dict, look_back = 4, look_ahead = 3):
    X_out_array = []
    y_out_array = []
    X_out_static = []
    X_out_dynamic = []
    
    #Converting to date.time  
    df_admissions['date'] = pd.to_datetime(df_admissions['date'], format = "%Y-%m-%d %H:%M:%S")
    X_df['date'] = pd.to_datetime(X_df['date'], format = "%Y-%m-%d %H:%M:%S")
    y_df['date'] = pd.to_datetime(y_df['date'], format = "%Y-%m-%d %H:%M:%S")

    #Arranging df as tensors
    for ID in X_df['ID'].unique():
        #creating necessary subsets
        X_df_sub = X_df[X_df['ID'] == ID]
        #df_tfidfvect_train_sub2 = df_tfidfvect_train_sub.drop(['ID','date'], axis=1)
        df_admissions_sub = df_admissions[df_admissions['ID'] == ID]
        if ID in y_df['ID'].values:
            #print(ID)
            uti_date = y_df.loc[y_df['ID'] == ID, 'date'].iloc[0]
        else: 
            uti_date = 'NULL'

        uti_pr_ID_list = []
        uti = 0

        #Loop to split data into groups of 4 days 
        for i in range(len(df_admissions_sub['date'])):
            #If uti_three_days_ago is positive, move on to next ID. Because we only want to exclude people who have had actually had a UTI, not people who will have it in two days.  
            #if uti_three_days_ago == 1:
                #continue
            if sum(uti_pr_ID_list) >= look_ahead:
                continue

            if i > (look_back-1):
                df_pred_days = df_admissions_sub[i-look_back: i] # -4 fordi vi starter ved i=0 
                #print(df_pred_days)
                date_list = []


                #if look_ahead date is in df use look_ahead date, else use last date in df
                if i+look_ahead < len(df_admissions_sub['date']):
                    look_forward = df_admissions_sub.iloc[i+look_ahead,0]
                else: 
                    look_forward = df_admissions_sub.iloc[-1,0]

                if uti_date != 'NULL' and look_forward > uti_date:
                    uti = 1
                else: 
                    uti = 0

                for date in X_df_sub['date']:
                    if  df_pred_days.iloc[0,0] <= date < df_pred_days.iloc[-1,0]:
                        date_list.append(date)

                #subset containing 4 days of data for 1 patient 
                subset = X_df_sub[X_df_sub['date'].isin(date_list)]

                #Creating padding for those who have too few notes
                if len(subset) < maximum_notes:
                    padding_length = maximum_notes - len(subset)
                
                    for i in range(padding_length):
                        subset = subset.append(pd.Series(0, index=subset.columns), ignore_index=True)
                        
                        ##Forsøg på at fikse deprecation warning, virker ikke!
                        #new_row = pd.Series(0, index=subset.columns)
                        #subset.loc[len(subset)] = new_row
                        #subset = pd.concat([subset, new_row], ignore_index=True)

                #Subset of only embeddings 
                subset2 = subset.drop(['ID','date'], axis=1)
                array_pr_ID = []
                #collecting all the static variables
                static_array_pr_ID = []
                #collecting all the dynamic variables
                dynamic_array_pr_ID = []

                for key, df in predictor_dict.items():
                    if "static" in key:
                        ID_sub = df[df['ID'] == ID]
                        static_array_pr_ID.append(ID_sub['value'].item())


                date_list2 = []

                #For dynamic dfs find the mean of the lookback period
                for key, df in predictor_dict.items():
                    if "dynamic" in key:
                        ID_sub = df[df['ID'] == ID]
                        ID_sub['date'] = pd.to_datetime(ID_sub['date'], format = "%Y-%m-%d %H:%M:%S")
                        for date in ID_sub['date']:
                            if  df_pred_days.iloc[0,0] <= date < df_pred_days.iloc[-1,0]:
                                date_list2.append(date)
                        ID_sub2 = ID_sub[ID_sub['date'].isin(date_list2)]
                        mean = ID_sub2['value'].mean()
                        

                        #If there is no score during the lookback period, take the latest score
                        if math.isnan(mean):
                            previous_scores = ID_sub[ID_sub['date'] < df_pred_days.iloc[0,0]]
                            #print(f'Previous scores: {previous_scores}')
                            if previous_scores.empty:
                                mean = ID_sub['value'].iloc[0] #if there is no score before or in the period, take the first score they have
                            else: 
                                latest_date = max(previous_scores['date'])
                                latest_score = ID_sub.loc[ID_sub['date'] == latest_date, 'value'].iloc[0]
                                mean = latest_score

                        #print(f'Mean: {mean}')
                        dynamic_array_pr_ID.append(mean)

                
                #Iterating over each row of ID-and-day-specific-embedding-df
                for index, row in subset2.iterrows():
                    value_list = []
                    #get every value in row and append to list 
                    for key, value in row.items():
                        value_list.append(value)
                    #append the list of all values to array
                    array_pr_ID.append(value_list)

                    #when all data within the given dates for one ID has been appended the ID_specifc_array append that to out_array
                    if len(array_pr_ID) == len(subset2):
                        X_out_array.append(array_pr_ID)
                        y_out_array.append(uti)
                        uti_pr_ID_list.append(uti)

                        X_out_static.append(static_array_pr_ID)
                        X_out_dynamic.append(dynamic_array_pr_ID)


    X_out_static_tensor = torch.Tensor(X_out_static)
    X_out_dynamic_tensor = torch.Tensor(X_out_dynamic)

    X_out_structured_tensor = torch.hstack((X_out_static_tensor, X_out_dynamic_tensor))

    #Make the out_arrays tensors (possible now because of the padding)
    X_out_tensor= torch.Tensor(X_out_array)
    y_out_tensor = torch.Tensor(y_out_array)


    return X_out_tensor, y_out_tensor, X_out_structured_tensor
                             


