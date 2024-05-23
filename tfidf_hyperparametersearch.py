
from xgboost import XGBClassifier
from sklearn.utils import class_weight
import optuna
from tfidf import tf_idf
from preparing_specs import preparing_specs
from sklearn.metrics import roc_auc_score
def tfidf_hyperparametersearch(predictor_dict_train, predictor_dict_test, df_admissions_train, df_admissions_test, y_train_df, y_test_df, categorical = True):
    
    def objective(trial):
        #Hyperparameterspaces to explore
        min_df = trial.suggest_int("min_df_trial", 0, 100, step = 10) 
        max_df = trial.suggest_float("max_df_trial", 0.50, 1.0, step = 0.1)
        max_features = trial.suggest_int("max_features_trial", 200, 600, step = 100)
        ngram_range_upper = trial.suggest_int("ngram_range_upper_trial", 1, 2)

        #Initialising
        df_tfidfvect_train, df_tfidfvect_test = tf_idf(predictor_dict_train["df_SFI_text"], predictor_dict_test["df_SFI_text"], min_df = min_df, max_df = max_df, max_features=max_features, ngram_range = (1, ngram_range_upper));
        
        #Preparing specs
        X_final_train_tfidf, y_final_train_tfidf =  preparing_specs(embeddings = df_tfidfvect_train,
                                                    predictor_dict = predictor_dict_train,
                                                    prediction_times = df_admissions_train, 
                                                    outcome_df = y_train_df, 
                                                    embedding_type ="tfidf",
                                                    lookbehind = 4, 
                                                    lookahead = 3)


        X_final_test_tfidf, y_final_test_tfidf =  preparing_specs(embeddings = df_tfidfvect_test,
                                                    predictor_dict = predictor_dict_test,
                                                    prediction_times = df_admissions_test, 
                                                    outcome_df = y_test_df, 
                                                    embedding_type ="tfidf",
                                                    lookbehind = 4, 
                                                    lookahead = 3)

        #Ensure brain injury variable is considered a categorical variable
        if categorical == True:
            X_final_train_tfidf[["pred_df_brain_injury_static"]] = X_final_train_tfidf[["pred_df_brain_injury_static"]].astype("category")

        #Computing class weights
        class_weights2 = class_weight.compute_sample_weight(class_weight='balanced', y = y_final_train_tfidf['outc_outcome_within_3_days_maximum_fallback_0_dichotomous'])

        #Initialise XGBoost
        xgb_model = XGBClassifier(objective="binary:logistic", random_state=42, enable_categorical = True, tree_method = 'hist')
        
        xgb_model.fit(X_final_train_tfidf, y_final_train_tfidf, sample_weight = class_weights2)

        #Ensure columns in test set are arranged correctly
        cols_when_model_builds = xgb_model.get_booster().feature_names
        X_final_test_tfidf = X_final_test_tfidf[cols_when_model_builds]

        #Ensure brain injury variable is considered a categorical variable in testset also
        if categorical == True:
            X_final_test_tfidf[["pred_df_brain_injury_static"]] = X_final_test_tfidf[["pred_df_brain_injury_static"]].astype("category")

        #Make predictions
        y_pred = xgb_model.predict(X_final_test_tfidf)  
        predictions = [round(value) for value in y_pred]


        #Creating list of true outcome values
        y_true = []

        for i in y_final_test_tfidf['outc_outcome_within_3_days_maximum_fallback_0_dichotomous']:
            y_true.append(int(i))

        #Evaluate based on ROC-AUC
        roc_auc = roc_auc_score(y_true, predictions)

        return roc_auc


    #Run study to find the tf-idf hyperparameters that maximize the ROC-AUC score of the XGBoost model
    study = optuna.create_study(direction = "maximize") 
    study.optimize(objective, n_trials = 50)  

    return study.best_params