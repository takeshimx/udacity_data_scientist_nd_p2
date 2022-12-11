import sys
import pandas as pd
from sqlalchemy import create_engine

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pickle

def load_data(database_filepath):
    """
    load_data
    This function loads data from a database
       and create 2 aliases for building a ML model.
    
    Inputs:
        database_filepath: database filepath
    
    Return:
        X: feature
        Y: target variable
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    tokenize
    This function returns list of cleaned words
    after 4 steps below, normalize, tokenize, remove stopwords, and stem / lemmatize.
    
    Inputs:
        text: text
    Return:
        lemmed: list of cleaned words
    """
    # normalize
    normalized_text = re.sub(r'[^a-zA-Z0-9]', " ", text.lower())
    
    # tokenize
    words = word_tokenize(normalized_text)
    
    # remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    
    # stem / lemmatize
    # reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    # reduce words to their root form using default pos
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed]
    
    return lemmed


def build_model():
    """
    build_model
    This function builds a ML model
    
    Return:
        model: tuned model
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    #'clf__estimator__bootstrap': True,
    #'vect__max_df': 0.5,
    'clf__estimator__n_estimators': [10],
    'clf__estimator__min_samples_split': [2],
    }

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=5, verbose=2, cv=2)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate_model
    This function evaluates a model.
    
    Inputs:
        model: ML model developed
        X_test: X for test
        Y_test: Y for test
        category_names: category names
    """
    y_pred = model.predict(X_test)
    
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    save_model
    This function is for saving a model and dumped as a pickle file.
    
    Inputs:
        model: ML model developed
        model_filepath: model filepath
    """
    file = open(model_filepath, 'wb')
    pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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
