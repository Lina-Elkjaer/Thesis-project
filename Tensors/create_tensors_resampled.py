import torch
import pandas as pd
import warnings
import math

warnings.filterwarnings("ignore", category=FutureWarning) # Doesnt really fix the issue that the function append will be deprecated later. however I have not been able to find solution.
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None # warning about overwriting, but I do it in purpose


def max_notes_resampled(X_df):
    #initiating max_length
    maximum_notes = 0
    ID_list = X_df['ID'].unique()

    for ID in ID_list:
        #creating subsets pr ID
        X_df_sub = X_df[X_df['ID'] == ID]
    
        if len(X_df_sub) > maximum_notes:
            maximum_notes = len(X_df_sub )
    
    return maximum_notes



def tensors_resampled(X_df, y_df, df_admissions, maximum_notes, predictor_dict):
    X_out_array = []
    y_out_array = []
    X_out_static = []
    X_out_dynamic = []
    n_notes_out = []
    
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
            uti = 1
        else: 
            uti = 0

        uti_pr_ID_list = []
      
        ######################   Initiating lists   ##########################
                
        #collecting text features
        array_pr_ID = []
        #collecting all the static variables
        static_array_pr_ID = []
        #collecting all the dynamic variables
        dynamic_array_pr_ID = []
        #collecting n_notes in each period for each patient
        #n_notes = []
                

        ##########################   PADDING    ###############################

        #Collecting n_notes in each period for each patient
        n_notes = len(X_df_sub) 
        if n_notes < 1: 
            continue
        #Creating padding for those who have too few notes
        if len(X_df_sub) < maximum_notes:
            padding_length = maximum_notes - len(X_df_sub)
                
            for i in range(padding_length):
                X_df_sub = X_df_sub.append(pd.Series(0, index=X_df_sub.columns), ignore_index=True)

                
        #Subset of only embeddings 
        subset2 = X_df_sub.drop(['ID','date'], axis=1) 



        ##########################   STATIC   ################################
        for key, df in predictor_dict.items():
            if "static" in key:
                ID_sub = df[df['ID'] == ID]
                static_array_pr_ID.append(ID_sub['value'].item())



        ##########################   DYNAMIC   ################################

        #For dynamic dfs find the mean of the lookback period
        for key, df in predictor_dict.items():
            if "dynamic" in key:
                ID_sub = df[df['ID'] == ID]
                mean = ID_sub['value'].mean()

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
                n_notes_out.append(n_notes)


    X_out_static_tensor = torch.Tensor(X_out_static)
    X_out_dynamic_tensor = torch.Tensor(X_out_dynamic)

    X_out_structured_tensor = torch.hstack((X_out_static_tensor, X_out_dynamic_tensor))

    n_notes_out_tensor = torch.LongTensor(n_notes_out) #must be long tensor to work as index

    #Make the out_arrays tensors (possible now because of the padding)
    X_out_tensor = torch.Tensor(X_out_array)
    y_out_tensor = torch.Tensor(y_out_array)


    return X_out_tensor, y_out_tensor, X_out_structured_tensor, n_notes_out_tensor
                             


