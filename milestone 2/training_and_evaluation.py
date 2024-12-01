# baseline_models_with_evaluation.py

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # For saving and loading models

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# Ignore warnings for clean output
import warnings
warnings.filterwarnings('ignore')

# Import your Evaluation class from evaluate.py
# Assuming evaluate.py is in the same directory
from evaluate import Evaluation

def parse_conllu_file(filepath, debug=False, num_samples=5):
    """
    Parses a CoNLL-U formatted file and extracts sentences and labels.
    If debug is True, prints the first num_samples sentences and labels.
    """
    sentences = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as file:
        current_sentence_tokens = []
        current_label = None
        for line in file:
            line = line.strip()
            if line.startswith('# label_sexist'):
                #Extract the label
                current_label = line.split('=')[1].strip()
            elif line == '':
                #End of the current sentence
                if current_sentence_tokens and current_label is not None:
                    sentence = ' '.join(current_sentence_tokens)
                    sentences.append(sentence)
                    labels.append(current_label)
                    current_sentence_tokens = []
                    current_label = None
            elif not line.startswith('#'):
                #Token line
                parts = line.split('\t')
                if len(parts) > 1:
                    token = parts[1]
                    current_sentence_tokens.append(token)
        #Catch any remaining sentence at the end of the file
        if current_sentence_tokens and current_label is not None:
            sentence = ' '.join(current_sentence_tokens)
            sentences.append(sentence)
            labels.append(current_label)
    df = pd.DataFrame({'text': sentences, 'label': labels})
    if debug:
        print(f"Parsed {len(df)} sentences from {filepath}")
        print(df.head(num_samples))
    return df

def load_and_prepare_data(train_path, test_path):
    """
    Loads data from CoNLL-U files and prepares training and test sets.
    """
    train_df = parse_conllu_file(train_path, debug=True)
    test_df = parse_conllu_file(test_path, debug=True)

    label_mapping = {'not sexist': 0, 'sexist': 1}
    train_df['label'] = train_df['label'].map(label_mapping)
    test_df['label'] = test_df['label'].map(label_mapping)

    return train_df, test_df

def preprocess_data(train_df, test_df):
    """
    Splits the data into features and labels.
    """
    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['label']
    return X_train, y_train, X_test, y_test

def save_model(model, vectorizer, model_name):
    """
    Saves the trained model and vectorizer to disk.
    """
    with open(f'{model_name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'{model_name}_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f'Model and vectorizer saved for {model_name}.')

def load_model(model_name):
    """
    Loads the trained model and vectorizer from disk.
    """
    with open(f'{model_name}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{model_name}_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print(f'Model and vectorizer loaded for {model_name}.')
    return model, vectorizer

def predict_input_text(model, vectorizer, text):
    """
    Predicts the label for a new input sentence.
    """
    processed_text = [text]
    text_vector = vectorizer.transform(processed_text)
    prediction = model.predict(text_vector)
    label = 'Sexist' if prediction[0] == 1 else 'Not Sexist'
    return label

def evaluate_model_with_evaluation_class(model, vectorizer, X_train, y_train, X_test, y_test, model_name):
    """
    Evaluates the model using the provided Evaluation class.
    """
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    evaluator = Evaluation(model)

    print(f"\nEvaluation for {model_name}:")
    evaluator.evaluate(X_train_vec, y_train, X_test_vec, y_test)

def logistic_regression_bow(X_train, y_train, X_test, y_test):
    """
    Logistic Regression with Bag-of-Words features.
    """
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    evaluate_model_with_evaluation_class(model, vectorizer, X_train, y_train, X_test, y_test, 'Logistic Regression')

    save_model(model, vectorizer, 'logistic_regression_bow')

def svm_tfidf(X_train, y_train, X_test, y_test):
    """
    Support Vector Machine with TF-IDF features.
    """
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(X_train_tfidf, y_train)

    # Evaluate using the Evaluation class
    evaluate_model_with_evaluation_class(model, vectorizer, X_train, y_train, X_test, y_test, 'SVM')

    # Save the model and vectorizer
    save_model(model, vectorizer, 'svm_tfidf')

def naive_bayes_bow(X_train, y_train, X_test, y_test):
    """
    Multinomial Naive Bayes with Bag-of-Words features.
    """
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate using the Evaluation class
    evaluate_model_with_evaluation_class(model, vectorizer, X_train, y_train, X_test, y_test, 'Naive Bayes')

    # Save the model and vectorizer
    save_model(model, vectorizer, 'naive_bayes_bow')


def regex_classifier(texts):
    """
    Rule-based classifier using regular expressions.
    """
    sexist_patterns = [
        r'\bmake me a sandwich\b',
        r'\bget back in the kitchen\b',
        r'\bwomen can\'t\b',
        r'\bgirls can\'t\b',
        r'\bwomen should\b',
        r'\bwomen are\b',
        r'\bgirls are\b',
        r'\bbitch(es)?\b',
        r'\bslut(s)?\b',
        r'\bwhore(s)?\b'
    ]

    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in sexist_patterns]

    def classify(text):
        for pattern in compiled_patterns:
            if pattern.search(text):
                return 1  # Sexist
        return 0  # Not Sexist

    return [classify(text) for text in texts]

def evaluate_rule_based_classifier(classifier_func, X_train, y_train, X_test, y_test, model_name):
    """
    Evaluates a rule-based classifier using the Evaluation class.
    """
    # Predictions
    y_train_pred = classifier_func(X_train)
    y_test_pred = classifier_func(X_test)

    #Create a dummy model that returns predictions (for compatibility with Evaluation class)
    class DummyModel:
        def predict(self, X):
            return np.array(classifier_func(X))

    dummy_model = DummyModel()

    #Transform X_train and X_test into dummy variables (needed but not used)
    X_train_dummy = np.zeros(len(X_train))
    X_test_dummy = np.zeros(len(X_test))

    #Create an instance of the Evaluation class
    evaluator = Evaluation(dummy_model)

    # Evaluate the model
    print(f"\nEvaluation for {model_name}:")
    evaluator.evaluate(X_train_dummy, y_train, X_test_dummy, y_test, plot_confusion=True)

def run_baseline_models(train_df, test_df):
    """
    Runs all baseline models, evaluates them, and saves the trained models.
    """
    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)

    # Logistic Regression with Bag-of-Words
    logistic_regression_bow(X_train, y_train, X_test, y_test)

    # SVM with TF-IDF
    svm_tfidf(X_train, y_train, X_test, y_test)

    # Naive Bayes with Bag-of-Words
    naive_bayes_bow(X_train, y_train, X_test, y_test)

def test_new_input_sentence():
    """
    Loads saved models and tests them on new input sentences until the user types 'exit' or 'quit'.
    """
    # Load models and vectorizers once
    lr_model, lr_vectorizer = load_model('logistic_regression_bow')
    svm_model, svm_vectorizer = load_model('svm_tfidf')
    nb_model, nb_vectorizer = load_model('naive_bayes_bow')

    print("\nEnter sentences to classify. Type 'exit' or 'quit' to stop.")
    while True:
        input_sentence = input("\nEnter a sentence: ").strip()
        if input_sentence.lower() in ('exit', 'quit'):
            print("Exiting the classification loop.")
            break
        if not input_sentence:
            print("Empty input. Please enter a valid sentence.")
            continue

        #Logistic Regression Model
        lr_prediction = predict_input_text(lr_model, lr_vectorizer, input_sentence)
        print(f"Logistic Regression Prediction: {lr_prediction}")

        #SVM Model
        svm_prediction = predict_input_text(svm_model, svm_vectorizer, input_sentence)
        print(f"SVM Prediction: {svm_prediction}")

        #Naive Bayes Model
        nb_prediction = predict_input_text(nb_model, nb_vectorizer, input_sentence)
        print(f"Naive Bayes Prediction: {nb_prediction}")




if __name__ == '__main__':
    train_path = '/data/processed/train_sexism_dataset_conllu.conllu'
    test_path = '/data/processed/test_sexism_dataset_conllu.conllu'


    train_df, test_df = load_and_prepare_data(train_path,test_path)

    run_baseline_models(train_df, test_df)

    test_new_input_sentence()
