#XGB_pipeline

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV
from sklearn.utils import class_weight
import scipy.stats as stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import pandas as pd

def XGB_hyperparametersearch(X_train, y_train, categorical = True, no_specs = False ):
        
    xgb_model = XGBClassifier(objective="binary:logistic", random_state=42, enable_categorical = True, tree_method = 'hist')
    
    model_params = {'learning_rate' : stats.uniform(0.01, 0.1),
    'max_depth' : stats.randint(3, 15),
    'min_child_weight' : [ 1, 3, 5, 7 ],
    'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
    'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ], 
    'reg_alpha' :stats.uniform(1, 20),
    'reg_lambda': stats.uniform(1, 20),
    'subsample': stats.uniform(0.5, 0.5),
    'n_estimators':stats.randint(50, 200)
    }

    if categorical == True:
        X_train[["pred_df_brain_injury_static"]] = X_train[["pred_df_brain_injury_static"]].astype("category")

    #Computing class weights
    if no_specs == True:
        class_weights = class_weight.compute_sample_weight(class_weight='balanced', y = y_train)
    else:    
        class_weights = class_weight.compute_sample_weight(class_weight='balanced', y = y_train['outc_outcome_within_3_days_maximum_fallback_0_dichotomous'])

    #Hyperparameter search
    clf = RandomizedSearchCV(xgb_model, model_params, n_iter=100, cv=10, random_state=42, n_jobs = 42, scoring = "roc_auc", verbose = 3)

    clf.fit(X_train, y_train, sample_weight = class_weights)

    print("Best parameters set:")
    clf.best_estimator_

    print(f'Best ROC-AUC score: {clf.best_score_}') 

    best_params = clf.best_estimator_.get_params()

    for key, param in best_params.items():
        print(f'{key} = {param},')

    return best_params





def XGB_evaluation(xgb_model, X_train, y_train, X_test, y_test, categorical = True, no_specs = False):
    
    if categorical == True:
        X_train[["pred_df_brain_injury_static"]] = X_train[["pred_df_brain_injury_static"]].astype("category")
    
    #Computing class weights
    if no_specs == True:
        class_weights = class_weight.compute_sample_weight(class_weight='balanced', y = y_train)
    else:    
        class_weights = class_weight.compute_sample_weight(class_weight='balanced', y = y_train['outc_outcome_within_3_days_maximum_fallback_0_dichotomous'])

    #Fit
    xgb_model.fit(X_train, y_train, sample_weight = class_weights)
    
    cols_when_model_builds = xgb_model.get_booster().feature_names


    #Reordering the columns of the dataframe 
    X_test = X_test[cols_when_model_builds]
    
    if categorical == True:
        X_test[["pred_df_brain_injury_static"]] = X_test[["pred_df_brain_injury_static"]].astype("category")


    #Testing on actual new data

    # make predictions
    y_pred = xgb_model.predict(X_test)  
    predictions = [round(value) for value in y_pred]

    print('MODEL PERFORMANCE DURING TRAINING:')
    print(f'Accuracy: {cross_val_score(xgb_model, X_train, y_train, cv = 5, scoring = "accuracy")}')
    print(f'ROC-AUC: {cross_val_score(xgb_model, X_train, y_train, cv = 5, scoring = "roc_auc")}')
    print(f'F1-score: {cross_val_score(xgb_model, X_train, y_train, cv = 5, scoring = "f1")}')

    def perf_measure(y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                FP += 1
            if y_actual[i]==y_hat[i]==0:
                TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                FN += 1

        return(f"True Positives: {TP}. False Positives: {FP}. True negatives: {TN}. False negatives: {FN}.")


    y_true = []
    if no_specs == False:
        for i in y_test['outc_outcome_within_3_days_maximum_fallback_0_dichotomous']:
            y_true.append(int(i))
    else:
        for i in y_test:
            y_true.append(int(i))

    print('MODEL PERFORMANCE DURING TESTING:')
    print(perf_measure(y_true, predictions))
    
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(f'Precision = {average_precision_score(y_true, predictions)}')
    print(f'Recall = {recall_score(y_true, predictions)}')
    print(f'F1 score = {f1_score(y_true, predictions)}')
    print(f'ROC-AUC score = {roc_auc_score(y_true, predictions)}') #0.524 that is predictive ability is not better than random guessing


    feature_important = xgb_model.get_booster().get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    #print(keys)

    keys2 = []

    for i in keys: 
        if 'pred_tfidf' in i:
            i = i.removeprefix("pred_tfidf")
            i = i.split('_', 1)[0]
            i = "TF-IDF feature: " + i

        elif 'pred_df_' in i:
            i = i.removeprefix("pred_df_")
            i = i.split('_', 1)[0]
            if i == "acute":
                i = "Acute days"
            if i == "FIM":
                i = "FIM-score"
            if i == "brain":
                i = "Brain injury"
            if i == "age":
                i = "Age"
            if i == "female":
                i = "Sex"
        
        elif 'pred_sentence' in i:
            i = i.removeprefix("pred_sentence")
            i = i.split('_', 1)[0]
            i = "Sentence feature: " + i


        keys2.append(i)


    data = pd.DataFrame(data=values, index=keys2, columns=["score"]).sort_values(by = "score", ascending = False)
    data.nlargest(10, columns="score").plot(kind='barh', figsize = (7.5, 4.5), xlabel = "Feature importance") #20,10 før ## plot top 10 features

    return y_pred