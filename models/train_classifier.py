import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re

import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin




def load_data(database_filepath):
    '''Load data from a .db file specified by database_filepath'''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Disaster',engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).astype(int)
    return X,Y, Y.columns


def tokenize(text):
    '''Customized tokenize function'''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    '''build a classification model using training data and sklearn pipeline
    The process contained using NLP pipelie: tokenize, count words, tfidf, multioutput Classifier and grid search using Cross Validation
    '''
    pipeline = Pipeline([
        ('vec', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
    'clf__estimator__n_estimators':[10,50]
    #'clf__max_depth':['3',None]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''Use test data to evaluate the model built
    Args:
        model: sklearn model trained with training data
        X_test: DataFrame for predictors of test data
        Y_test: DataFrame for labels of test data
        category_names: category names to be evaluated
    Returns:
        None
    Performance is printed out
    '''
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns = Y_test.columns,index = Y_test.index)
    for col in category_names:
        print(col+':')
        print(classification_report(Y_test[col],y_pred_df[col]))

def save_model(model, model_filepath):
    '''Saving trained model in pickle
    Args:
        model: trained sklearn model
        model_filepath: file path and name for saved pickle file
    Return:
        None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        #print(X_train.iloc[:3]) 
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()