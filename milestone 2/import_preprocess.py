import conllu
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import vstack


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
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None

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
            for tokenlist in data:
                y.append(tokenlist.metadata['label_sexist'])
                X.append([token['form'] for token in tokenlist])
            
            datasets[dataset_type] = {'X': X, 'y': y}
        
        self.X_train, self.y_train = datasets['train']['X'], datasets['train']['y']
        self.X_val, self.y_val = datasets['dev']['X'], datasets['dev']['y']
        self.X_test, self.y_test = datasets['test']['X'], datasets['test']['y']

    def concatenate_train_val(self):
        return self.X_train + self.X_val, self.y_train + self.y_val
    
    def create_bow_representation(self, max_features=300):
        """
        Transform tokenized sentences into bag of words (BoW) representation.
        'CountVectorizer' is applied, but without any preprocessing (already done in milestone 1).
        Due to high number of unique tokens, only the 'max_features' most frequent tokens are considered.
        """
        vectorizer = CountVectorizer(analyzer=lambda x: x, token_pattern=None, max_features=max_features)
        X_train_bow = vectorizer.fit_transform(self.X_train)
        X_val_bow = vectorizer.transform(self.X_val)
        X_test_bow = vectorizer.transform(self.X_test)

        X_train_val_bow = vstack([X_train_bow, X_val_bow]) # for final evaluation purpose

        return X_train_bow, X_val_bow, X_test_bow, X_train_val_bow, vectorizer.get_feature_names_out()
    
    def pad_token_sequences(self):
        raise NotImplementedError()