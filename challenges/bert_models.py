import sys
import os

milestone_2_path = os.path.abspath("../milestone 2")
sys.path.append(milestone_2_path)

import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, DebertaV2Tokenizer, DebertaForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from baseline_models import ClassificationModel
from import_preprocess import convert_labels_to_string, convert_labels_to_int


class BERTModel(ClassificationModel):
    def __init__(self, model_name):
        """
        Initialize the model with a specified transformer architecture: 
        "DeBERTa", "RoBERTa", "HateBERT" or "DistilBERT".
        """
        print(f"Loading pretrained model and tokenizer: {model_name}...")

        model_dict = {
            "DeBERTa": (DebertaForSequenceClassification, DebertaV2Tokenizer, "microsoft/deberta-v3-base"),
            "RoBERTa": (RobertaForSequenceClassification, RobertaTokenizer, "roberta-base"),
            "HateBERT": (BertForSequenceClassification, BertTokenizer, "GroNLP/hateBERT"),
            "DistilBERT": (DistilBertForSequenceClassification, DistilBertTokenizer, "distilbert-base-uncased")
        }
        
        if model_name not in model_dict:
            raise ValueError(f"Model {model_name} not supported. Choose from: 'DeBERTa', 'RoBERTa', 'HateBERT' or 'DistilBERT'.")
        
        self.model_name = model_name
        model_class, tokenizer_class, model_path = model_dict[model_name]
        
        # load model and tokenizer 
        self.model = model_class.from_pretrained(model_path, num_labels=2)
        self.tokenizer = tokenizer_class.from_pretrained(model_path)
        
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
    
    
    def train(self, X_train, y_train, X_dev, y_dev, epochs=10, batch_size=32, lr=5e-5, patience=10, plot_training_curve=True):
        """
        Train the model with the provided training and validation datasets.
        Monitor cross-entropy loss for both sets and trigger early stopping if no improvement 
        is observed on the validation set for 'patience' consecutive epochs.
        Set 'plot_training_curve' to True to visualize the training and validation loss curves.
        """
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
        training_loss = []
        validation_loss = []
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad() # clear the gradients
                X_batch, y_batch = batch
                # X_batch = X_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
                # y_batch = torch.tensor(y_batch).to('cuda' if torch.cuda.is_available() else 'cpu')
                
                outputs = self.model(X_batch, labels=y_batch) # computes loss too
                loss = outputs.loss # cross-entropy loss
                loss.backward() # backpropagation after each batch
                optimizer.step() # update model's parameters
                
                total_train_loss += loss.item()
            
            print(f"Epoch {epoch + 1}/{epochs}, Cross-entropy Loss: {total_train_loss:.4f}")
            avg_train_loss = total_train_loss / len(train_loader) # loss per batch
            training_loss.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            all_preds = []
            all_labels = []
            total_dev_loss = 0
            with torch.no_grad():
                for batch in dev_loader:
                    X_batch, y_batch = batch
                    # X_batch = X_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
                    outputs = self.model(X_batch, labels=y_batch)
                    total_dev_loss += outputs.loss.item()

                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y_batch.numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            print(f"Validation Loss: {total_dev_loss:.4f}, Validation Accuracy: {acc:.4f}")
            avg_dev_loss = total_dev_loss / len(dev_loader) # loss per batch
            validation_loss.append(avg_dev_loss)

            # early stopping logic
            if validation_loss[-1] < best_val_loss:
                best_val_loss = validation_loss[-1]
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # plot training and validation curves if requested
        if plot_training_curve:
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(training_loss) + 1), training_loss, label='training loss', marker='o', color='steelblue')
            plt.plot(range(1, len(validation_loss) + 1), validation_loss, label='validation loss', marker='o', color='orange')
            plt.xlabel('epoch')
            plt.ylabel('cross-entropy loss per batch')
            plt.title('Training and validation loss curves')
            plt.legend()
            plt.grid(linewidth=0.5)
            plt.show()
    
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

        