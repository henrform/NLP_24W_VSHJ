import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from import_preprocess import convert_labels_to_int, convert_labels_to_string
from evaluate import Evaluation

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from import_preprocess import convert_labels_to_int, convert_labels_to_string

def get_all_predictions(dic_models, sample_x, sample_x_bow, sample_y):
    """
    Get predictions for all models on a single sample.
    """
    if sample_y not in ('sexist', 'not sexist'):
        if sample_y in (0, 1):
            sample_y = "sexist" if sample_y == 1 else "not sexist"
        else:          
            sample_y = convert_labels_to_string(sample_y)
    
    prediction = {'true': sample_y}
    for name, model in dic_models.items():
        if name == 'Majority Class':
            prediction[name] = model.predict([sample_x])
            
        elif name in ('Naive Bayes', 'Logistic Regression', 'XGBoost (BOW)'):
            prediction[name] = model.predict(sample_x_bow)
            
        elif name == 'LSTM':
            pred = model.predict([sample_x])
            pred = "sexist" if pred[0][0] > 0.5 else "not sexist"
            prediction[name] = pred
            
        else:
            raise ValueError(f"Model {name} not recognized")
    return prediction

class ClassificationModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def evaluate(self, X_dict, y_true_list, plot_confusion=True, model_name=""):
        """
        Evaluate the model on train+dev and test sets.
        Metrics: accuracy, balanced accuracy, precision, recall.
        """
        y_pred_list = [self.predict(X) for X in X_dict.values()]
        
        if model_name == "LSTM":
            y_pred_list = [convert_labels_to_string(y) for y in y_pred_list]
            y_true_list = [convert_labels_to_string(y) for y in y_true_list]

        dataset_names = []
        print("#" * 40 + "\n")
        results = []
        for i, name in enumerate(X_dict.keys()):
            dataset_names.append(name)
            print(f"Metrics for {name}")
            acc, bal_acc, prec, rec = self._calculate_print_metrics(y_true_list[i], y_pred_list[i])
            results.append([model_name, name, acc, bal_acc, prec, rec])
            print("\n" + "#" * 40 + "\n")

        if plot_confusion:
            self._plot_confusion_matrices(y_true_list, y_pred_list, dataset_names, model_name)
            
        return results

    def _calculate_print_metrics(self, y_true, y_pred):
        """
        Calculate and print accuracy, balanced accuracy, precision and recall.
        """
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0.0, pos_label='sexist')
        recall = recall_score(y_true, y_pred, zero_division=0.0, pos_label='sexist')

        print(f"accuracy: {accuracy:.4f}")
        print(f"balanced accuracy: {balanced_accuracy:.4f}")
        print(f"precision: {precision:.4f}")
        print(f"recall: {recall:.4f}")
        return accuracy, balanced_accuracy, precision, recall

    def _plot_confusion_matrices(self, y_true_list, y_pred_list, dataset_names, model_name):
        """
        Plot confusion matrices for train+dev and test sets.
        """
        num_datasets = len(dataset_names)
        fig, axes = plt.subplots(1, num_datasets, figsize=(4*num_datasets, 4))

        for i, pair in enumerate(zip(dataset_names, y_true_list, y_pred_list)):
            name, y_true, y_pred = pair
            cm = confusion_matrix(y_true, y_pred)

            plt.subplot(1, num_datasets, i + 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=["0 - not sexist", "1 - sexist"], yticklabels=["0 - not sexist", "1 - sexist"])
            plt.title(f'Confusion matrix: {name}')
            plt.xlabel('predicted')
            plt.ylabel('actual')

        plt.suptitle(model_name)
        plt.tight_layout()
        plt.show()


class XGBoostClassifier(ClassificationModel):
    def __init__(self, max_depth=10, learning_rate=0.1, n_estimators=100, verbosity=1, objective='binary:logistic',
                 eval_metric='logloss'):
        """
        'max_depth' - maximum depth of each tree
        'learning_rate' - controlling the contribution of each tree to the final model
        'n_estimators' - number of boosting rounds (trees) to be trained

        Objective function to be minimized is binary cross-entropy.
        """
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.objective = objective
        self.eval_metric = eval_metric
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        # convert the input data to DMatrix, which is the internal data structure XGBoost uses
        y_train = convert_labels_to_int(y_train)
        y_val = convert_labels_to_int(y_val)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        ddev = xgb.DMatrix(X_val, label=y_val)

        evals = [(dtrain, 'train'), (ddev, 'eval')]

        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'verbosity': self.verbosity,
        }

        # train the model, incorporate early stopping too
        self.model = xgb.train(params, dtrain, num_boost_round=self.n_estimators, evals=evals, early_stopping_rounds=10)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        dtest = xgb.DMatrix(X)
        y_pred = self.model.predict(dtest)

        # since this is binary classification, we round the predictions to get 0 or 1
        y_pred = (y_pred > 0.5).astype(int)
        return convert_labels_to_string(y_pred)


class MajorityClassClassifier(ClassificationModel):
    def __init__(self):
        self.majority_class = None

    def train(self, X_train, y_train, X_val, y_val):
        """
        Determines the majority class from the training labels by manually counting.
        No need for a separate development set, therefore use concatenation.
        """
        count_not_sexist = 0
        count_sexist = 0

        for label in y_train:
            if label == "not sexist":
                count_not_sexist += 1
            elif label == "sexist":
                count_sexist += 1

        if count_sexist > count_not_sexist:
            self.majority_class = "sexist"
        else:
            self.majority_class = "not sexist"

    def predict(self, X):
        """
        Predict the majority class for all input samples, no matter the content.
        """
        if self.majority_class is None:
            raise ValueError("Model must be trained before prediction")

        return [self.majority_class] * len(X)


class NaiveBayesClassifier(ClassificationModel):
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X)


class LogisticRegression(ClassificationModel):
    def __init__(self, vectorizer=None):
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        self.model = linear_model.LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X)


class LSTM_Model(ClassificationModel):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def fit_tokenizer(self, X_train):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(X_train)
    
    def prepare_X(self, X):
        seq = self.tokenizer.texts_to_sequences(X)
        padded = pad_sequences(seq, padding='post')
        return padded
        
    def initialize_model(self, vocab_size):
        model = Sequential()
        model.add(Embedding(input_dim = vocab_size, output_dim = 64))
        model.add(LSTM(32))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, X_train, y_train, X_dev, y_dev):
        self.fit_tokenizer(X_train)
        vocab_size = len(self.tokenizer.word_index) + 1
        
        X_train = self.prepare_X(X_train)
        X_dev = self.prepare_X(X_dev)
        y_train = np.array(y_train)
        y_dev = np.array(y_dev)
        
        self.model = self.initialize_model(vocab_size)
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_dev, y_dev))
        
    def predict(self, X):
        X = self.prepare_X(X)
        probs = self.model.predict(X)
        y_pred = (probs > 0.5).astype(int)
        return y_pred