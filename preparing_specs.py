#Preparing specs 

from timeseriesflattener.df_transforms import df_with_multiple_values_to_named_dataframes
from timeseriesflattener.aggregation_fns import mean
from timeseriesflattener.feature_specs.group_specs import PredictorGroupSpec
import numpy as np
import pandas as pd
# Specify how to aggregate the predictors and define the outcome
from timeseriesflattener.feature_specs.single_specs import OutcomeSpec, PredictorSpec, StaticSpec
from timeseriesflattener.aggregation_fns import maximum, mean

from timeseriesflattener import TimeseriesFlattener

def preparing_specs(embeddings, predictor_dict, prediction_times, outcome_df, embedding_type = "_unknown_embedding_type_", lookbehind = 4, lookahead = 3):
    
    prediction_times_df = prediction_times.reset_index(drop=True)

    # split the dataframe into a list of named dataframes with one value each
    embedded_dfs = df_with_multiple_values_to_named_dataframes(
        df=embeddings,
        entity_id_col_name="ID",
        timestamp_col_name="date",
        name_prefix = embedding_type,
    )

    emb_spec_batch = PredictorGroupSpec(
        named_dataframes=embedded_dfs,
        lookbehind_days=[lookbehind], #kan tilføje flere [7, 14] så for man en predictor som kigger 7 dage tilbage og en der kigger 14 dage tilbage
        fallback=[np.nan],
        aggregation_fns=[mean],
    ).create_combinations()

    #spec_list = []

    outcome_spec = OutcomeSpec(
        timeseries_df = outcome_df,
        lookahead_days=lookahead,
        fallback=0,
        aggregation_fn = maximum,
        feature_base_name="outcome",
        incident=False,
        )

    ts_flattener = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        entity_id_col_name="ID",
        timestamp_col_name="date",
        n_workers=40,
        drop_pred_times_with_insufficient_look_distance=True,
    )

    for key, df in predictor_dict.items():

        if "static" in key:
            spec = StaticSpec(
                timeseries_df = df,
                feature_base_name=key,
                prefix="pred",
            )

        elif "dynamic" in key:
            spec = PredictorSpec(
                timeseries_df = df,
                lookbehind_days = lookbehind,
                fallback=np.nan, 
                aggregation_fn=mean,
                feature_base_name=key,
            )
        
        elif "text" in key:
            continue

        else: 
            print(f"{key} has been skipped. All predictors in predictor_dict must have either text, static or dynamic in the name, depending on which type of predictor it is. Predictors that have neither, will be skipped")
            continue

        ts_flattener.add_spec([spec])


    print("Training specs will take a while")


    ts_flattener.add_spec([*emb_spec_batch, outcome_spec])
    
    df_train = ts_flattener.get_df()

    # find alle predictor columns (has prefix "pred_")
    predictor_columns_train =  [col for col in df_train if col.startswith('pred_')]
    # find outcome column (has prefix "outc_")
    outcome_column_train = [col for col in df_train if col.startswith('outc_')]

    X_final = df_train[predictor_columns_train]
    y_final = df_train[outcome_column_train]

    return X_final, y_final


