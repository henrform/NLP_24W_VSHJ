import conllu
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import vstack
import numpy as np

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
        self.X_train, self.y_train, self.S_train = None, None, None
        self.X_val, self.y_val, self.S_val = None, None, None
        self.X_test, self.y_test, self.S_test = None, None, None
        self.X_train_balanced, self.y_train_balanced = None, None

    def import_train_val_test(self):
        """
        Importing and parsing the train, val and test datasets from CoNLL-U format
        (formed in milestone 1).

        Updates:
        X_train, y_train, X_val, y_val, X_test, y_test
            - X_* contains tokenized sentences
            - y_* contains the labels ('sexist'/'not sexist')
        """
        datasets = {}
        for dataset_type in ['train', 'dev', 'test']:
            file_path = f"{self.folder_path}{dataset_type}_sexism_dataset_conllu.conllu"
            with open(file_path, encoding='ISO-8859-1') as f:
                data = conllu.parse(f.read())
            
            X = []
            y = []
            S = []
            for tokenlist in data:
                y.append(tokenlist.metadata['label_sexist'])
                X.append([token['form'] for token in tokenlist])
                S.append(tokenlist.metadata['text'])
            
            datasets[dataset_type] = {'X': X, 'y': y, 'S': S}
        
        self.X_train, self.y_train, self.S_train = datasets['train']['X'], datasets['train']['y'], datasets['train']['S']
        self.X_val, self.y_val, self.S_val = datasets['dev']['X'], datasets['dev']['y'], datasets['dev']['S']
        self.X_test, self.y_test, self.S_test = datasets['test']['X'], datasets['test']['y'], datasets['test']['S']

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
        
        if conversion_needed:
            y_balanced = convert_labels_to_string(y_balanced)

        self.X_train_balanced, self.y_train_balanced = X_balanced, y_balanced
        
    
    def create_bow_representation(self, max_features=300, balanced=False):
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
        
        # X_balanced, _ = self.create_balanced_dataset(self.X_train, self.y_train, n_samples=5000)
        # X_balanced_bow = vectorizer.transform(X_balanced)

        X_train_val_bow = vstack([X_train_bow, X_val_bow]) # for final evaluation purpose

        # return X_train_bow, X_val_bow, X_test_bow, X_balanced_bow, X_train_val_bow, vectorizer.get_feature_names_out()
        return X_train_bow, X_val_bow, X_test_bow, X_train_val_bow, vectorizer.get_feature_names_out()


    def create_tfidf_representation(self, max_features=300, balanced=False):
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

    