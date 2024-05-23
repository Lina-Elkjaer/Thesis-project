#Creating sentence embeddings

from sentence_transformers import SentenceTransformer
import pandas as pd


def sentence_embeddings(df_free_text:pd.DataFrame, transformer_model = 'encoder-large-v1'):
    
    #Load model
    sentence_model = SentenceTransformer(transformer_model)

    #Encode
    sentence_embeddings = sentence_model.encode(df_free_text['note'])

    #Create df
    sentence_embeddings_df = pd.DataFrame(sentence_embeddings)

    #Merge to get IDs and dates
    X_embeddings = pd.merge(df_free_text, sentence_embeddings_df, how='left', left_index=True, right_index=True)

    #Exclude the notes
    df_sentence = X_embeddings.loc[:, X_embeddings.columns != 'note']

    #Return embeddings df
    return df_sentence

    
