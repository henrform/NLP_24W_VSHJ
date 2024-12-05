import xgboost as xgb

class XGBoostClassifier:
    def __init__(self, max_depth=10, learning_rate=0.1, n_estimators=100, verbosity=1, objective='binary:logistic', eval_metric='logloss'):
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
    
    def train(self, X_train, y_train, X_dev, y_dev):
        # convert the input data to DMatrix, which is the internal data structure XGBoost uses
        dtrain = xgb.DMatrix(X_train, label=y_train)
        ddev = xgb.DMatrix(X_dev, label=y_dev)
        
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
    
    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        y_pred = self.model.predict(dtest)
        
        # since this is binary classification, we round the predictions to get 0 or 1
        y_pred = (y_pred > 0.5).astype(int)
        return y_pred
    
