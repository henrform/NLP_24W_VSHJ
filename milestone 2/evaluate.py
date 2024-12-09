from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Evaluation:
    def __init__(self, model):
        """
        The 'model' should be an instance of a model class and must already be trained.
        """
        self.model = model

    def evaluate(self, X_train_dev, y_train_dev, X_test, y_test, plot_confusion=True, model_name=""):
        """
        Evaluate the model on train+dev and test sets.
        Metrics: accuracy, balanced accuracy, precision, recall.
        """
        y_train_dev_pred = self.model.predict(X_train_dev)
        y_test_pred = self.model.predict(X_test)

        print("Metrics for TRAIN+DEV set")
        self.calculate_print_metrics(y_train_dev, y_train_dev_pred)
        print("#"*40)
        print("\nMetrics for TEST set")
        self.calculate_print_metrics(y_test, y_test_pred)

        if plot_confusion:
            self.plot_confusion_matrices(y_train_dev, y_train_dev_pred, y_test, y_test_pred, model_name)


    def calculate_print_metrics(self, y_true, y_pred):
        """
        Calculate and print accuracy, balanced accuracy, precision and recall.
        """
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label='sexist')
        # precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, pos_label='sexist')
        # recall = recall_score(y_true, y_pred)

        print(f"accuracy: {accuracy:.4f}")
        print(f"balanced accuracy: {balanced_accuracy:.4f}")
        print(f"precision: {precision:.4f}")
        print(f"recall: {recall:.4f}")

    def plot_confusion_matrices(self, y_train_dev, y_train_dev_pred, y_test, y_test_pred, model_name):
        """
        Plot confusion matrices for train+dev and test sets.
        """
        cm_train_dev = confusion_matrix(y_train_dev, y_train_dev_pred) # conf matrix for train+dev
        cm_test = confusion_matrix(y_test, y_test_pred) # conf matrix for test

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        sns.heatmap(cm_train_dev, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
                    xticklabels=["0 - not sexist", "1 - sexist"], yticklabels=["0 - not sexist", "1 - sexist"])
        axes[0].set_title(f'Confusion matrix: train+dev \n model: {model_name}')
        axes[0].set_xlabel('predicted')
        axes[0].set_ylabel('actual')

        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],
                    xticklabels=["0 - not sexist", "1 - sexist"], yticklabels=["0 - not sexist", "1 - sexist"])
        axes[1].set_title(f'Confusion matrix: test \n model: {model_name}')
        axes[1].set_xlabel('predicted')
        axes[1].set_ylabel('actual')

        plt.tight_layout()
        plt.show()
