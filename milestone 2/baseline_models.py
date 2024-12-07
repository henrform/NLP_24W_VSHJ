import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from import_preprocess import convert_labels_to_int, convert_labels_to_string


class XGBoostClassifier:
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


class MajorityClassClassifier:
    def __init__(self):
        self.majority_class = None

    def train(self, y):
        """
        Determines the majority class from the training labels by manually counting.
        No need for a separate development set, therefore use concatenation.
        """
        count_not_sexist = 0
        count_sexist = 0

        for label in y:
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


class NaiveBayesClassifier:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        self.model = MultinomialNB()
        self.model.fit(X, y)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X)


class LogisticRegression:
    def __init__(self, vectorizer=None):
        pass

    def train(self, X_train, y_train, X_val, y_val):
        pass

    def predict(self, X):
        pass