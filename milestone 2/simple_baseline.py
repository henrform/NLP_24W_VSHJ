class MajorityClassClassifier:
    def __init__(self):
        self.majority_class = None

    def fit(self, X_train_dev, y_train_dev):
        """
        Determines the majority class from the training labels by manually counting.
        No need for a separate development set, therefore use concatenation.
        """
        count_0 = 0
        count_1 = 0
        
        for label in y_train_dev:
            if label == "not sexist":
                count_0 += 1
            elif label == "sexist":
                count_1 += 1
        
        if count_1 > count_0:
            self.majority_class = "sexist"
        else:
            self.majority_class = "not sexist"

    def predict(self, X_test):
        """
        Predict the majority class for all input samples, no matter the content.
        """
        if self.majority_class is None:
            raise ValueError("The classifier hasn't been fitted yet.")
        
        return [self.majority_class] * len(X_test)



# class RegexClassifier: # TO BE CONTINUED...