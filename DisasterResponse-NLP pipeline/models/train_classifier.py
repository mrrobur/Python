# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    """
    Loads data from SQLlite. Drops Y columns where only one class occurs.
    :param database_filepath: Database file path
    :return:
        X - independent variables
        Y - dependent variables
        y_var - ordered column names (possible classes) of Y
    """
    # load data from database
    df = pd.read_sql_table('Messages_Categories', "sqlite:///"+database_filepath)
    drop_var = ['id','message','original','genre']

    y_var = list(set(df.columns) - set(drop_var) -set('message') )
    X = df['message'].values

    # remove Y values where only more than 1 classes
    y_unique = df[y_var].nunique()
    print("Dropping following columns as it has only one target class:\n", [i for i in y_unique[y_unique==1].index])
    Y = df[list(y_unique[y_unique>1].index)].values
    return X, Y, y_var

def tokenize(text):
    """
    Custom function that removes digits, URLs, stopwords, tokenize and lemmatize a sentence.
    :param text: A sentence to tokenize
    :return: Cleaned tokens
    """
    text_only = r'[^A-Za-z]'
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
  
    detected_urls = re.findall(url_regex,text)
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")
    
    stopw = stopwords.words('english') + ['urlplaceholder']
    text = re.sub(text_only," ",text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    tokens_wo_stopwords = [i for i in tokens if i not in stopw]
    
    clean_tokens = []
    for tok in tokens_wo_stopwords:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def score_model(Y_test, Y_pred, y_var, average_type=None):
    """
    Calculates precision, recall and f1 of model
    :param Y_test: Real value of dependent variable
    :param Y_pred: Predicted value of dependent variable
    :param y_var: list of Y column names
    :param average_type: type of average used in f1 score a calculation
    :return: pandas.Dataframe: scores for each Y class
    """
    results = dict()
    # classification_report is only printing a string which is unredible, using precision_recall_fscore_support instead
    for col in zip(Y_test.T, Y_pred.T, y_var):
        y_test, y_pred, var_name = col[0], col[1], col[2]
        class_report = precision_recall_fscore_support(y_test, y_pred, average=average_type)

        # precision, recall, f1, support = class_report
        d = {
            'precision': class_report[0],
            'recall': class_report[1],
            'f1-score': class_report[2],
            'support': class_report[3]
        }
        results[var_name] = d
    return pd.DataFrame.from_dict(results, orient='index').drop('support', axis=1)


def evaluate_model(model, X_test, Y_test, category_names, average_type=None):
    """
    Predicts and then evaluates model
    :param model: Trained model
    :param X_test: Independent variables from test set
    :param Y_test: Dependent variables from test set
    :param category_names: names of categories of dependent variable
    :param average_type: type of average used in f1 score a calculation
    :return: pandas.Dataframe: scores for each Y class
    """
    Y_pred = model.predict(X_test)
    return score_model(Y_test, Y_pred, category_names, average_type=average_type)

def f1_mean(Y_test, Y_pred, category_names):
    """
    Wrapper function that return only an average of f1-score for all predicted classes
    :param Y_test: Dependent variables from test set
    :param Y_pred: Indpendent variables from test set
    :param category_names: names of categories of dependent variable
    :return: float: average f1-score for all classes
    """
    return score_model(Y_test, Y_pred, category_names, 'macro').mean()['f1-score']


def build_model(category_names):
    """
    Build a whole pipeline for training the model and uses GridSearch to find optimal parameters
    :param category_names: names of categories of dependent variable
    :return: GridSearchCV object: best found model evaluated towards mean f1-score
    """
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
            ])

    parameters = {
        'vect__ngram_range': [(1,1), (1,2)],
        'clf__estimator__class_weight': [None,'balanced']
    }

    f1_mean_score = make_scorer(f1_mean, category_names=category_names, greater_is_better=True)
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=f1_mean_score, verbose=4, cv=4, n_jobs=1)
    return cv

def save_model(model, model_filepath):
    """
    Saves model to pickle in given path
    :param model: Trained model
    :param model_filepath: Destination and pickled file name
    :return: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    return None

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=32)
        
        print('Building model...')
        model = build_model(category_names)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        model_eval = evaluate_model(model, X_test, Y_test, category_names, 'macro')

        print("Specific scores:")
        print(model_eval)

        print('Scores distribution')
        print(model_eval.describe())

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