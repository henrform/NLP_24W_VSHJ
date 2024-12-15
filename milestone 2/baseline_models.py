import re
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from import_preprocess import convert_labels_to_int, convert_labels_to_string

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from collections import Counter
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
        Evaluate the model on train and validatino sets.
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
            self._plot_confusion_matrices(y_true_list, y_pred_list, dataset_names, model_name, ax=None, plot=True)
            
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

    def _plot_confusion_matrices(self, y_true_list, y_pred_list, dataset_names, model_name, ax=None, plot=True):
        """
        Plot confusion matrices for train and validation sets.
        """
        num_datasets = len(dataset_names)
        if ax is None:
            fig, ax = plt.subplots(1, num_datasets, figsize=(4*num_datasets, 4))
            ax = [ax] if num_datasets == 1 else ax

        for i, pair in enumerate(zip(dataset_names, y_true_list, y_pred_list)):
            name, y_true, y_pred = pair
            cm = confusion_matrix(y_true, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=["0 - not sexist", "1 - sexist"], yticklabels=["0 - not sexist", "1 - sexist"], ax=ax[i])
            ax[i].set_title(f'Confusion matrix: {name}')
            ax[i].set_xlabel('predicted')
            ax[i].set_ylabel('actual')

        plt.suptitle(model_name)
        plt.tight_layout()
        if plot:
            plt.show()
        else:
            return ax

    def qualitative_analysis_top_tokens(self, X_original_tokenized, X_representation, y, set_type="train", model_name="", plot=False, top_n=25):
        """
        X_original_tokenized - original lists of tokens
        X_representation - BOW/TF-IDF features (if model should use them us an input)

        For each of the 4 corners of confision matrix (TN, FP, FN, TP) find 'top_n' most 
        frequently occuring tokens. Plot bar charts if 'plot' set.  
        """
        y_pred = self.predict(X_representation)

        if model_name == "LSTM":
            y_pred = convert_labels_to_string(y_pred)
            y = convert_labels_to_string(y) 

        true_positives = []
        false_negatives = []
        false_positives = []
        true_negatives = []

        for tokens, true_label, pred_label in zip(X_original_tokenized, y, y_pred):
            if true_label == "sexist" and pred_label == "sexist":
                true_positives.append(tokens)
            elif true_label == "sexist" and pred_label == "not sexist":
                false_negatives.append(tokens)
            elif true_label == "not sexist" and pred_label == "sexist":
                false_positives.append(tokens)
            elif true_label == "not sexist" and pred_label == "not sexist":
                true_negatives.append(tokens)

        top_tn_tokens = self._get_top_tokens(true_negatives, n=top_n)
        top_fp_tokens = self._get_top_tokens(false_positives, n=top_n)
        top_fn_tokens = self._get_top_tokens(false_negatives, n=top_n)
        top_tp_tokens = self._get_top_tokens(true_positives, n=top_n)

        if plot:
            self._plot_top_token(top_tn_tokens, top_fp_tokens, top_fn_tokens, top_tp_tokens, set_type=set_type, model_name=model_name)
        else:
            return top_tn_tokens, top_fp_tokens, top_fn_tokens, top_tp_tokens

        
    def _get_top_tokens(self, tokenized_sentences, n=25):
        all_tokens = [token for sentence in tokenized_sentences for token in sentence]  # flatten the list of lists
        token_counts = Counter(all_tokens)
        return token_counts.most_common(n)
    
    def _plot_top_token(self, top_tn_tokens, top_fp_tokens, top_fn_tokens, top_tp_tokens, set_type, model_name):
        data = {
            "true negatives": top_tn_tokens,
            "false positives": top_fp_tokens,
            "false negatives": top_fn_tokens,
            "true positives": top_tp_tokens
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(6,10))
        axes = axes.flatten()

        for ax, (title, tokens) in zip(axes, data.items()):
            if not tokens:  
                ax.set_title(title, fontsize=12)
                ax.text(0.5, 0.5, 'no tokens', ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue
            tokens, counts = zip(*tokens)  
            ax.barh(tokens, counts, color='steelblue')  
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("count")  
            ax.invert_yaxis()
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=10)
        
        plt.suptitle(f"{model_name}\nset: {set_type}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  
        plt.show()
        
    
    def get_context_for_token(self, token, input, y, X, S, no_examples=5, return_contexts=False):
        """
        Find the context for a given token in the dataset.
        """
        predictions = self.predict(input)
        contexts = []
        processed_tokens = []
        y_true = []
        y_pred = []
        for tokens, label, prediction, sentence in zip(X, y, predictions, S):
            if token in tokens:
                contexts.append((sentence, label))
                processed_tokens.append(tokens)
                y_true.append(label)
                y_pred.append(prediction)
        
        all_tokens = [word for tokens in processed_tokens for word in tokens if word != token]
        all_tokens = [word for word in all_tokens if len(word) > 2]
        token_counts = Counter(all_tokens)
        top_tokens = token_counts.most_common(10)
        tokens, counts = zip(*top_tokens)
        tokens = tokens[::-1]
        counts = counts[::-1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self._plot_confusion_matrices([y_true], [y_pred], [""], model_name=f"Token: {token}", ax=[ax1], plot=False)
        ax2.barh(tokens, counts, color='steelblue')
        ax2.set_title(f"Most common tokens in the context of '{token}'")
        plt.tight_layout()
        plt.show()
        
        print("Examples:")
        for context, label in contexts[:no_examples]:
            label = "sexist" if label == 1 or label == "sexist" else "not sexist"
            print(f"Label: {label}")
            print(context)
            print()
        
        if return_contexts:
            return contexts 

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
    

class RuleBasedClassifier(ClassificationModel):
    def __init__(self, patterns):
        """
        'patterns' - list of patterns that we consider as hate speech/sexism
        """
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def classify(self, text):
        for pattern in self.patterns:
            if pattern.search(text):
                return "sexist" 
        return "not sexist"
    
    def train(self, S_train):
        pass
    
    def predict(self, S_test):
        return [self.classify(text) for text in S_test]


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