import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from import_preprocess import convert_labels_to_int, convert_labels_to_string
from evaluate import Evaluation

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


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
        # y_train_dev_pred = self.predict(X_train_dev)
        # y_test_pred = self.predict(X_test)

        dataset_names = []
        print("#" * 40 + "\n")
        for i, name in enumerate(X_dict.keys()):
            dataset_names.append(name)
            print(f"Metrics for {name}")
            self._calculate_print_metrics(y_true_list[i], y_pred_list[i])
            print("\n" + "#" * 40 + "\n")

        # print("Metrics for TRAIN+DEV set")
        # self.calculate_print_metrics(y_train_dev, y_train_dev_pred)
        # print("#" * 40)
        # print("\nMetrics for TEST set")
        # self.calculate_print_metrics(y_test, y_test_pred)

        if plot_confusion:
            self._plot_confusion_matrices(y_true_list, y_pred_list, dataset_names, model_name)

    def _calculate_print_metrics(self, y_true, y_pred):
        """
        Calculate and print accuracy, balanced accuracy, precision and recall.
        """
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0.0, pos_label='sexist')
        # precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, zero_division=0.0, pos_label='sexist')
        # recall = recall_score(y_true, y_pred)

        print(f"accuracy: {accuracy:.4f}")
        print(f"balanced accuracy: {balanced_accuracy:.4f}")
        print(f"precision: {precision:.4f}")
        print(f"recall: {recall:.4f}")

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
        model = linear_model.LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X)