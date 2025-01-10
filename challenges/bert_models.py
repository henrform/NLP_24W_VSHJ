import sys
import os

milestone_2_path = os.path.abspath("../milestone 2")
sys.path.append(milestone_2_path)

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from baseline_models import ClassificationModel
from import_preprocess import convert_labels_to_string, convert_labels_to_int

class HateBERTModel(ClassificationModel):
    def __init__(self):
        print("Loading pretrained model and tokenizer...")
        self.model = BertForSequenceClassification.from_pretrained("GroNLP/hateBERT", num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained("GroNLP/hateBERT")
        print("Pretrained weights loaded successfully.")
            
    def prepare_X(self, X):
        """
        Generates encoded padded tensors from the input data.
        Each sentence, initially represented as a list of tokens, is encoded by converting each token 
        into its corresponding ID from the tokenizer's vocabulary. 
        Padding is applied to ensure all sequences in the batch have the same length.
        """
        X_text = [' '.join(tokens) for tokens in X]
        encoded = [self.tokenizer.encode(text) for text in X_text]
        padded = pad_sequence([torch.tensor(seq) for seq in encoded], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return padded
    
    def prepare_dataset(self, X, y=None):
        """Create a PyTorch dataset from the input data"""
        class TextDataset(Dataset):
            def __init__(self, X, y=None):
                self.X = X
                self.y = y
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                if self.y is not None:
                    return self.X[idx], self.y[idx]
                return self.X[idx]
        return TextDataset(X, y)
    
    
    def train(self, X_train, y_train, X_dev, y_dev, epochs=3, batch_size=32, lr=5e-5):
        X_train = self.prepare_X(X_train)
        X_dev = self.prepare_X(X_dev)

        y_train = convert_labels_to_int(y_train) # 0/1
        y_dev = convert_labels_to_int(y_dev)
        
        train_dataset = self.prepare_dataset(X_train, y_train)
        dev_dataset = self.prepare_dataset(X_dev, y_dev)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
        
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define optimizer and loss function
        optimizer = AdamW(self.model.parameters(), lr=lr)
        # loss_fn = torch.nn.CrossEntropyLoss()
        
        print("Training started.")
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad() # clear the gradients
                X_batch, y_batch = batch
                # X_batch = X_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
                # y_batch = torch.tensor(y_batch).to('cuda' if torch.cuda.is_available() else 'cpu')
                
                outputs = self.model(X_batch, labels=y_batch) # computes loss too
                loss = outputs.loss # cross-entropy loss
                loss.backward() # backpropagation after each batch
                optimizer.step() # update model's parameters
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}/{epochs}, Cross-entropy Loss: {total_loss:.4f}")
            
            # Validation
            self.model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in dev_loader:
                    X_batch, y_batch = batch
                    # X_batch = X_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
                    outputs = self.model(X_batch)
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y_batch.numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            print(f"Validation Accuracy: {acc:.4f}")
    
    def predict(self, X):
        X = self.prepare_X(X)
        dataset = self.prepare_dataset(X)
        data_loader = DataLoader(dataset, batch_size=32)
        
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                # batch = batch.to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = self.model(batch)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)
        
        return convert_labels_to_string(predictions) # "sexist"/"not sexist"