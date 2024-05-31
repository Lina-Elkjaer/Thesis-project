#tf-idf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.da.stop_words import STOP_WORDS

def tf_idf(X_train_SFI, X_test_SFI, **kwargs):
    
    #For removing numbers in notes
    X_train_SFI['note'] = X_train_SFI['note'].str.replace('\d+', '')
    X_test_SFI['note'] = X_test_SFI['note'].str.replace('\d+', '')


    #FIT
    danish_stopwords = list(STOP_WORDS)

    vectorizer = TfidfVectorizer(stop_words = danish_stopwords, **kwargs)

    X_train_temp = vectorizer.fit_transform(X_train_SFI['note'])
    
    print("Tokens", vectorizer.get_feature_names_out())

    tokens = vectorizer.get_feature_names_out()

    # Convert sparse matrix to DataFrame
    df_tfidfvect_train = pd.DataFrame(data = X_train_temp.toarray(), columns = tokens)


    #Transform
    X_test_temp = vectorizer.transform(X_test_SFI['note'])

    df_tfidfvect_test = pd.DataFrame(data = X_test_temp.toarray(), columns = tokens) #bruger tokens fra X_train


    #Adding date and ID to the train df
    df_tfidfvect_train['ID'] = X_train_SFI['ID']
    df_tfidfvect_train['date'] = X_train_SFI['date']

    #Adding date and ID to the test df
    df_tfidfvect_test['ID'] = X_test_SFI['ID']
    df_tfidfvect_test['date'] = X_test_SFI['date']

    return df_tfidfvect_train, df_tfidfvect_test


    