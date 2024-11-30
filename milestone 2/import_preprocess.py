import conllu

class ImportPreprocess:
    def __init__(self, folder_path="../data/processed/"):
        self.folder_path = folder_path
        self.X_train, self.y_train = None, None
        self.X_dev, self.y_dev = None, None
        self.X_test, self.y_test = None, None

    def import_train_dev_test(self):
        """
        Importing and parsing the train, dev and test datasets from CoNLL-U format 
        (formed in milestone 1).

        Returns:
        X_train, y_train, X_dev, y_dev, X_test, y_test
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
        self.X_dev, self.y_dev = datasets['dev']['X'], datasets['dev']['y']
        self.X_test, self.y_test = datasets['test']['X'], datasets['test']['y']
    
    def convert_class_labels(self):
        """
        'sexist' -> 1, 'not sexist' -> 0
        """
        self.y_train = [1 if label == "sexist" else 0 for label in self.y_train]
        self.y_dev = [1 if label == "sexist" else 0 for label in self.y_dev]
        self.y_test = [1 if label == "sexist" else 0 for label in self.y_test]

    def concatenate_train_dev(self):
        return self.X_train + self.X_dev, self.y_train + self.y_dev

    def pad_token_sequences(self):
        raise NotImplementedError()