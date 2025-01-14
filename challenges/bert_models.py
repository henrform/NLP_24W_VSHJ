import sys
import os

milestone_2_path = os.path.abspath("../milestone 2")
sys.path.append(milestone_2_path)

import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, DebertaV2Tokenizer, DebertaForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset as HFDataset # for HF integration
from baseline_models import ClassificationModel
from import_preprocess import convert_labels_to_string, convert_labels_to_int


class BERTModel(ClassificationModel):
    def __init__(self, model_name, load_path=None):
        """
        Initialize the model with a specified transformer architecture: 
        "DeBERTa", "RoBERTa", "HateBERT" or "DistilBERT".

        Initialize the model with pretrained weights.
        If 'load_path' is specified, load the model weights from the checkpoint. 
        """
        model_dict = {
            "DeBERTa": (DebertaForSequenceClassification, DebertaV2Tokenizer, "microsoft/deberta-v3-base"),
            "RoBERTa": (RobertaForSequenceClassification, RobertaTokenizer, "cardiffnlp/twitter-roberta-base-sentiment-latest"),
            "HateBERT": (BertForSequenceClassification, BertTokenizer, "GroNLP/hateBERT"),
            "DistilBERT": (DistilBertForSequenceClassification, DistilBertTokenizer, "distilbert-base-uncased")
        }
        
        if model_name not in model_dict:
            raise ValueError(f"Model {model_name} not supported. Choose from: 'DeBERTa', 'RoBERTa', 'HateBERT' or 'DistilBERT'.")
        
        self.model_name = model_name
        model_class, tokenizer_class, model_path = model_dict[model_name]

        # instantiate model
        print(f"Loading pretrained weights: {model_name}...")
        self.model = model_class.from_pretrained(model_path, num_labels=2, ignore_mismatched_sizes=True)
        print("Pretrained weights loaded successfully.")
        
        # load from the checkpoint
        if load_path:
            print(f"Loading model weights from {load_path}...")
            self.load_model(load_path)

        print("Loading tokenizer...")
        self.tokenizer = tokenizer_class.from_pretrained(model_path)
        print("Tokenizer loaded sucessfully.")
        
        self.best_model = self.model # ensure predictions use the best model from training, not the final one

    def save_model(self, save_path):
        """
        Save the model to the specified path.
        During training, we're monitoring validation loss and preserving best model's weights.
        Same tokenization process should be used for predictions later and we're ensuring it during instantiation.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_path = os.path.join(save_path, f"best_{self.model_name}.pth")

        # save model state 
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {save_path}.")
    
    def load_model(self, load_path):
        """
        Load the model from the specified path.
        """
        model_path = os.path.join(load_path, f"best_{self.model_name}.pth")

        # load model state
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print(f"Model loaded from {load_path}.")
            
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
    
    def prepare_dataset_hf(self, X, y=None):
        """
        Create a Hugging Face Dataset from the input data.

        Example: 
        X[0] = ['shes', 'keeper', 'winking_face'], y[0] = 'not sexist'
        encoded_dataset[0] = {'text': ['shes', 'keeper', 'winking_face'],
                              'label': 0, # 'not sexist'
                              'input_ids': [101, 2016, 2015, 10684, 16837, 2075, 1035, 2227, 102, 0, 0, ..., 0], 
                              'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ..., 0],
                              'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ..., 0]}
        'input_ids':
            numerical IDs corresponding to the tokens, based on the tokenizer's vocabulary
            [101] - special [CLS] token (used as a classification marker)
            [102] - special [SEP] token (used as a sequence boundary)
        'attention_mask':
            indicates which tokens should be attended
            1 - the token is part of the actual input
            0 - the token is padding and should be ignored 
        """
        data = {"text": X}  # X is a list of lists of tokens
        if y is not None:
            data["label"] = convert_labels_to_int(y) # 0/1

        # Hugging Face Dataset
        dataset = HFDataset.from_dict(data)

        # encode the dataset
        def encode_function(examples):
            return self.tokenizer(
                examples["text"],  # pre-tokenized input as lists
                truncation=True,
                max_length=512,
                padding="max_length",
                is_split_into_words=True  # tell the tokenizer input is pre-tokenized
            )

        encoded_dataset = dataset.map(encode_function, batched=True)

        return encoded_dataset
    
    def train(self, X_train, y_train, X_dev, y_dev, epochs=10, batch_size=32, lr=5e-5, patience=10, plot_training_curve=True, save_path="best_model"):
        """
        Train the model with the provided training and validation datasets.
        Monitor cross-entropy loss for both sets and trigger early stopping if no improvement 
        is observed on the validation set for 'patience' consecutive epochs.
        Save the best-performing model during training to the 'save_path'.
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
                X_batch = X_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
                y_batch = torch.tensor(y_batch).to('cuda' if torch.cuda.is_available() else 'cpu')
                
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
                    X_batch = X_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
                    y_batch = y_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
                    outputs = self.model(X_batch, labels=y_batch)
                    total_dev_loss += outputs.loss.item()

                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y_batch.cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            print(f"Validation Loss: {total_dev_loss:.4f}, Validation Accuracy: {acc:.4f}")
            avg_dev_loss = total_dev_loss / len(dev_loader) # loss per batch
            validation_loss.append(avg_dev_loss)

            # early stopping logic; save the model if better than current best
            if validation_loss[-1] < best_val_loss:
                best_val_loss = validation_loss[-1]
                patience_counter = 0
                print(f"New best model found! Saving to {save_path}.")
                self.best_model = self.model
                self.save_model(save_path)
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
                batch = batch.to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = self.model(batch)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)
        
        return convert_labels_to_string(predictions) # "sexist"/"not sexist"

    def train_hugging_face_api(self, X_train, y_train, X_dev, y_dev, epochs=10, batch_size=32, lr=5e-5, patience=10):
        """
        This uses Hugging Face's built-in training loop and optimizers for model training.
        """
        train_dataset = self.prepare_dataset_hf(X_train, y_train)
        val_dataset = self.prepare_dataset_hf(X_dev, y_dev)

        # define metrics for evaluation
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = torch.argmax(torch.tensor(logits), dim=1).numpy()
            return {'accuracy': accuracy_score(labels, predictions)}

        # training arguments
        training_args = TrainingArguments(
            output_dir=f'./results/{self.model_name}', # directory for model checkpoints
            evaluation_strategy="epoch",     # evaluate every epoch
            save_strategy="epoch",           # save model every epoch
            save_total_limit=1,              # only save the last best model
            load_best_model_at_end=True,     # load the best model at the end of training
            metric_for_best_model="eval_loss",  # track validation loss to find the best model
            greater_is_better=False,         # lower loss is better (cross-entropy loss automatically)
            num_train_epochs=epochs,         # number of epochs
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.01,               # weight decay for regularization
            logging_dir=f'./logs/{self.model_name}', # directory for logs
            logging_steps=10,
        )

        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=patience)

        # Trainer
        trainer = Trainer(
            model=self.model,               # the pre-trained model
            args=training_args,             # training arguments
            train_dataset=train_dataset,    # training dataset
            eval_dataset=val_dataset,       # validation dataset
            tokenizer=self.tokenizer,       # tokenizer
            compute_metrics=compute_metrics,      # evaluation metrics
            callbacks=[early_stopping_callback],  # add early stopping callback
        )

        # training
        print("Training using Hugging Face Trainer API...")
        trainer.train()

        # evaluate the model
        eval_results = trainer.evaluate()
        print(f"Evaluation Results: {eval_results}")

        