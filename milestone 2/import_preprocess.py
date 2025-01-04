import conllu
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import vstack
import numpy as np
import ast
from collections import Counter

def convert_labels_to_string(y):
    """
    1 -> 'sexist, 0 -> 'not sexist'
    """
    return ["sexist" if label == 1 else "not sexist" for label in y]


def convert_labels_to_int(y):
    """
    'sexist' -> 1, 'not sexist' -> 0
    """
    return [1 if label == "sexist" else 0 for label in y]


class ImportPreprocess:
    def __init__(self, folder_path="../data/processed/"):
        self.folder_path = folder_path
        self.X_train, self.y_train, self.S_train, self.y_train_multi = None, None, None, None
        self.X_val, self.y_val, self.S_val, self.y_val_multi = None, None, None, None
        self.X_test, self.y_test, self.S_test, self.y_test_multi = None, None, None, None
        self.X_train_balanced, self.y_train_balanced, self.y_train_multi_balanced = None, None, None

    def import_train_val_test(self):
        """
        Importing and parsing the train, val and test datasets from CoNLL-U format
        (formed in milestone 1).

        Updates:
        X_train, y_train, X_val, y_val, X_test, y_test
            - X_* contains tokenized sentences
            - y_* contains the labels ('sexist'/'not sexist')
        S_train, S_val, S_test
            - S_* contains original sentences
        y_train_multi, y_val_multi, y_test_multi
            - y_*_multi includes annotations provided by 3 labelers (non-aggregated)
        """
        datasets = {}
        for dataset_type in ['train', 'dev', 'test']:
            file_path = f"{self.folder_path}{dataset_type}_sexism_dataset_conllu.conllu"
            with open(file_path, encoding='ISO-8859-1') as f:
                data = conllu.parse(f.read())
            
            X = []
            y = []
            S = []
            y_multi = []
            for tokenlist in data:
                y.append(tokenlist.metadata['label_sexist'])
                X.append([token['form'] for token in tokenlist])
                S.append(tokenlist.metadata['text'])
                y_multi.append(ast.literal_eval(tokenlist.metadata["multi_label"]))
            
            datasets[dataset_type] = {'X': X, 'y': y, 'S': S, 'y_multi': y_multi}
        
        self.X_train, self.y_train, self.S_train, self.y_train_multi = datasets['train']['X'], datasets['train']['y'], datasets['train']['S'], datasets['train']['y_multi']
        self.X_val, self.y_val, self.S_val, self.y_val_multi = datasets['dev']['X'], datasets['dev']['y'], datasets['dev']['S'], datasets['dev']['y_multi']
        self.X_test, self.y_test, self.S_test, self.y_test_multi = datasets['test']['X'], datasets['test']['y'], datasets['test']['S'], datasets['test']['y_multi']

    def concatenate_train_val(self):
        return self.X_train + self.X_val, self.y_train + self.y_val
    
    def create_balanced_dataset(self, n_samples=5000):
        """
        Create a balanced dataset by sampling n_samples from each class.
        """
        conversion_needed = False
        if self.y_train[0] != 0 and self.y_train[0] != 1:
            y = convert_labels_to_int(self.y_train)
            conversion_needed = True
        y = np.array(y)
        
        class_0_indices = np.where(y == 0)[0]
        class_1_indices = np.where(y == 1)[0]
        
        replace_0 = n_samples > len(class_0_indices)
        replace_1 = n_samples > len(class_1_indices)
         
        sampled_class_0_indices = np.random.choice(class_0_indices, size=n_samples, replace=replace_0)
        sampled_class_1_indices = np.random.choice(class_1_indices, size=n_samples, replace=replace_1)

        balanced_indices = np.concatenate([sampled_class_0_indices, sampled_class_1_indices])
        np.random.shuffle(balanced_indices) 

        X_balanced = [self.X_train[i] for i in balanced_indices]
        y_balanced = [y[i] for i in balanced_indices]
        y_multi_balanced = [self.y_train_multi[i] for i in balanced_indices]
        
        if conversion_needed:
            y_balanced = convert_labels_to_string(y_balanced)

        self.X_train_balanced, self.y_train_balanced, self.y_train_multi_balanced = X_balanced, y_balanced, y_multi_balanced
        
    def apply_aggregation(self, aggregation_type='majority voting'):
        """
        Aggregate y_train_multi, y_val_multi, y_test_multi based on the specified aggregarion_type.
        """
        def aggregate_labels(y_multi, aggregation_type):
            """
            Aggregates labels of 3 annotators.
            aggregation_type: method for aggregating labels 
                - 'majority voting': assigns the label that occurs most frequently among the 3 annotators
                - 'at least one sexist': assigns 'sexist' if at least 1 annotator labeled it as 'sexist';
                                         otherwise, assigns 'not sexist'
            """
            if aggregation_type not in {'majority voting', 'at least one sexist'}:
                raise ValueError("Invalid aggregation_type. Choose 'majority voting' or 'at least one sexist'.")

            aggregated_labels = []
            for labels in y_multi:
                if aggregation_type == 'majority voting':
                    # count the frequency of each label and choose the most common one
                    label_counts = Counter(labels)
                    aggregated_labels.append(label_counts.most_common(1)[0][0])
                elif aggregation_type == 'at least one sexist':
                    # assign 'sexist' if any annotator labeled it as such
                    if 'sexist' in labels:
                        aggregated_labels.append('sexist')
                    else:
                        aggregated_labels.append('not sexist')

            return aggregated_labels

        # apply aggregation to each dataset
        y_train_agg = aggregate_labels(self.y_train_multi, aggregation_type)
        y_train_balanced_agg = aggregate_labels(self.y_train_multi_balanced, aggregation_type) if self.y_train_balanced else None
        y_val_agg = aggregate_labels(self.y_val_multi, aggregation_type)
        y_test_agg = aggregate_labels(self.y_test_multi, aggregation_type)

        return y_train_agg, y_train_balanced_agg, y_val_agg, y_test_agg

    
    def create_bow_representation(self, max_features=3000, balanced=False):
        """
        Transform tokenized sentences into bag of words (BoW) representation.
        'CountVectorizer' is applied, but without any preprocessing (already done in milestone 1).
        Due to high number of unique tokens, only the 'max_features' most frequent tokens are considered.
        If working with balanced dataset ('balanced'), 'CountVectorizer' should be fit with it. 
        (It's important for determining 'max_features' most occuring tokens)
        """
        vectorizer = CountVectorizer(analyzer=lambda x: x, token_pattern=None, max_features=max_features)
        if balanced:
            X_train_bow = vectorizer.fit_transform(self.X_train_balanced)
        else:
            X_train_bow = vectorizer.fit_transform(self.X_train)

        X_val_bow = vectorizer.transform(self.X_val)
        X_test_bow = vectorizer.transform(self.X_test)

        X_train_val_bow = vstack([X_train_bow, X_val_bow]) # for final evaluation purpose

        return X_train_bow, X_val_bow, X_test_bow, X_train_val_bow, vectorizer.get_feature_names_out()


    def create_tfidf_representation(self, max_features=3000, balanced=False):
        """
        Transform tokenized sentences into TF-IDF representation.
        Same explanation as in 'create_bow_representation'.
        """
        vectorizer = TfidfVectorizer(analyzer=lambda x: x, token_pattern=None, max_features=max_features)
        if balanced:
            X_train_tfidf = vectorizer.fit_transform(self.X_train_balanced)
        else:
            X_train_tfidf = vectorizer.fit_transform(self.X_train)

        X_val_tfidf = vectorizer.transform(self.X_val)
        X_test_tfidf = vectorizer.transform(self.X_test)

        X_train_val_tfidf = vstack([X_train_tfidf, X_val_tfidf])  # for final evaluation purpose

        return X_train_tfidf, X_val_tfidf, X_test_tfidf, X_train_val_tfidf, vectorizer.get_feature_names_out()    