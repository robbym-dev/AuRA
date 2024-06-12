import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

class CustomBERTModel(nn.Module):
    """
    A custom model class that encapsulates a pre-trained BERT model and a classifier layer.
    This model is designed to handle various pre-trained models and adapt their embeddings
    for a classification task with a specified number of labels.
    """
    def __init__(self, number_of_labels: int, model_choice: str):
        """
        Initializes the CustomBERTModel with a specific model and number of labels.
        
        Args:
            number_of_labels (int): The number of labels for the classification task.
            model_choice (str): The pre-trained model identifier from Hugging Face's model hub.
        """
        super(CustomBERTModel, self).__init__()
        self.model_choice = model_choice

        # Conditional initialization based on the model_choice to handle different model configurations
        if model_choice in ["mosaicml/mpt-7b-instruct", "mosaicml/mpt-7b"]:
            config = AutoConfig.from_pretrained(model_choice, trust_remote_code=True)
            config.attn_config['attn_impl'] = 'triton'
            config.max_seq_len = 2048
            model_encoding = AutoModelForCausalLM.from_pretrained(
                model_choice,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_auth_token=True
            )
            embedding_size = 4096
            self.encoderModel = model_encoding.transformer
        elif model_choice in ['mosaicml/mpt-1b-redpajama-200b']:
            model_encoding = AutoModel.from_pretrained(model_choice, trust_remote_code=True)
            embedding_size = 2048
            self.encoderModel = model_encoding
        elif model_choice in ["google/t5-large-lm-adapt", "google/t5-xl-lm-adapt"]:
            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 1024
            self.encoderModel = model_encoding
        elif model_choice in ["roberta-large", "microsoft/deberta-v3-large"]:
            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 1024
            self.encoderModel = model_encoding
        elif model_choice in ["microsoft/deberta-v2-xlarge", "microsoft/deberta-v2-xxlarge"]:
            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 1536
            self.encoderModel = model_encoding
        else:
            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 768
            self.encoderModel = model_encoding

        # Classifier layers that map the embedding size to the number of labels
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.Linear(256, number_of_labels)
        )
        self.embedding_size = embedding_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None, decoder_input_ids: torch.Tensor = None):
        """
        Defines the forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input IDs from the tokenizer.
            attention_mask (torch.Tensor): Attention mask to avoid focusing on padding tokens.
            labels (torch.Tensor, optional): Labels for the input data.
            decoder_input_ids (torch.Tensor, optional): Decoder input IDs for models that require them.
        
        Returns:
            torch.Tensor: The output logits from the classifier or the model's direct outputs.
        """
        if self.model_choice in ["google/t5-xl-lm-adapt", "google/t5-large-lm-adapt"]:
            total_output = self.encoderModel(input_ids=input_ids, attention_mask=attention_mask)
            return total_output['logits']
        else:
            total_output = self.encoderModel(input_ids, attention_mask=attention_mask)
            sequence_output = total_output['last_hidden_state']
            last_hidden_state_formatted = sequence_output[:, 0, :].view(-1, self.embedding_size)
            linear2_output = self.classifier(last_hidden_state_formatted)
            return linear2_output
    
    def save_pretrained(self, save_directory):
        """
        Saves the model and its configuration to a specified directory.
        
        Args:
            save_directory (str): The directory where the model and configuration will be saved.
        """
        os.makedirs(save_directory, exist_ok=True)
        # Save the model
        model_path = os.path.join(save_directory, "model_checkpoint.pt")
        torch.save(self.state_dict(), model_path)
        # Save the configuration
        config = AutoConfig.from_pretrained(self.model_choice)
        config.save_pretrained(save_directory)
        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory, model_choice, number_of_labels):
        """
        Class method to load a pretrained model from a specified directory.
        
        Args:
            load_directory (str): The directory from which to load the model.
            model_choice (str): The pre-trained model identifier.
            number_of_labels (int): The number of labels for the classification task.
        
        Returns:
            CustomBERTModel: The loaded model instance.
        """
        config = AutoConfig.from_pretrained(model_choice)
        model = cls(number_of_labels, model_choice)
        model_path = os.path.join(load_directory, "model_checkpoint.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # Remove the classifier weights from the state_dict
        state_dict.pop('classifier.1.weight', None)
        state_dict.pop('classifier.1.bias', None)

        # Load the remaining state_dict
        model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded from {load_directory}")

        return model

class QueryDocumentDataset(Dataset):
    """
    A custom Dataset class for handling query-document pairs with labels.
    """
    def __init__(self, queries, documents, labels, tokenizer, max_length=512):
        """
        Initializes the QueryDocumentDataset with queries, documents, labels, and a tokenizer.
        
        Args:
            queries (list): A list of query strings.
            documents (list): A list of document strings corresponding to the queries.
            labels (list): A list of integer labels corresponding to the query-document pairs.
            tokenizer (AutoTokenizer): A tokenizer instance for tokenizing the text.
            max_length (int): The maximum length of the tokenized output.
        """
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure labels are valid integers
        self.labels = [int(label) for label in labels if not pd.isna(label)]

    def __len__(self):
        """
        Returns the number of items in the dataset.
        
        Returns:
            int: The number of queries.
        """
        return len(self.queries)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset by index.
        
        Args:
            idx (int): The index of the item.
        
        Returns:
            dict: A dictionary containing the tokenized query, document, and the label.
        """
        query = self.queries[idx]
        document = self.documents[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            query, document,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

def prepare_data_for_training(data_path, tokenizer, max_length=512, label_column="ARES_Answer_Relevance_Prediction"):
    """
    Prepares a dataset for training from a specified data file.
    
    Args:
        data_path (str): The path to the data file.
        tokenizer (AutoTokenizer): The tokenizer to use for processing text.
        max_length (int): The maximum token length for the tokenizer.
        label_column (str): The name of the column containing the labels.
    
    Returns:
        QueryDocumentDataset: The prepared dataset.
    """
    data = pd.read_csv(data_path, sep='\t')
    data = data.dropna(subset=[label_column])  # Drop rows with NaN values in the label column
    queries = data['Query'].tolist()
    documents = data['Document'].tolist()
    labels = data[label_column].tolist()
    
    dataset = QueryDocumentDataset(queries, documents, labels, tokenizer, max_length)
    return dataset

class EarlyStopping:
    """
    A utility class for implementing early stopping during model training.
    """
    def __init__(self, patience=3, verbose=False, delta=0):
        """
        Initializes the EarlyStopping instance.
        
        Args:
            patience (int): The number of epochs to wait before stopping after no improvement.
            verbose (bool): If True, prints detailed messages.
            delta (float): The minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        """
        Call method that checks if early stopping conditions are met.
        
        Args:
            val_loss (float): The validation loss of the current epoch.
            model (nn.Module): The model being trained.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves the model checkpoint if the validation loss has decreased.
        
        Args:
            val_loss (float): The current validation loss.
            model (nn.Module): The model to save.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss

def train_model(model, train_dataloader, val_dataloader, num_epochs, learning_rate, device, patience=3):
    """
    Trains the model using the provided data loaders and training parameters.
    
    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): The DataLoader for training data.
        val_dataloader (DataLoader): The DataLoader for validation data.
        num_epochs (int): The total number of epochs to train.
        learning_rate (float): The learning rate for the optimizer.
        device (torch.device): The device to train the model on.
        patience (int): The patience for early stopping.
    
    Returns:
        nn.Module: The trained model.
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = loss_fn(outputs, batch['labels'])
            loss.backward()

            optimizer.step
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss}")

        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                loss = loss_fn(outputs, batch['labels'])
                total_val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                correct_predictions += torch.sum(predictions == batch['labels'])

        avg_val_loss = total_val_loss / len(val_dataloader)
        accuracy = correct_predictions.double() / len(val_dataloader.dataset)
        print(f"Average Validation Loss: {avg_val_loss}, Accuracy: {accuracy}")

        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model

def save_model(model, tokenizer, model_dir):
    """
    Saves the trained model and tokenizer to a specified directory.
    
    Args:
        model (nn.Module): The model to save.
        tokenizer (AutoTokenizer): The tokenizer to save.
        model_dir (str): The directory where the model and tokenizer will be saved.
    """
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
def prepare_classifier_data_from_files(aura_config, llm_names):
    """
    Prepares classifier data from files specified in the aura_config for the given llm_names.
    
    Args:
        aura_config (dict): Configuration containing paths and other settings.
        llm_names (list): List of names of language models.
    
    Returns:
        dict: A dictionary containing documents, queries, answers, and ARES results.
    """
    documents = []
    queries = []
    answers = {llm_name: [] for llm_name in llm_names}

    docs_df = pd.read_csv(os.path.join(aura_config["LLM_prediction_folder_directory"], "best_retriever_docs.tsv"), sep='\t')
    queries = docs_df['Query'].tolist()
    documents = docs_df['Document'].tolist()

    for llm_name in llm_names:
        results_file = os.path.join(aura_config["LLM_prediction_folder_directory"], f"{llm_name}_results.tsv")
        results_df = pd.read_csv(results_file, sep='\t')
        answers[llm_name] = results_df['Answer'].tolist()

    ares_results = load_ares_predictions(aura_config, llm_names)

    classifier_data = {
        "documents": documents,
        "queries": queries,
        "answers": answers,
        "ares_results": ares_results
    }

    return classifier_data

def load_ares_predictions(aura_config, llm_names):
    """
    Loads ARES predictions for the specified llm_names from the aura_config.
    
    Args:
        aura_config (dict): Configuration containing paths and other settings.
        llm_names (list): List of names of language models.
    
    Returns:
        dict: A dictionary containing ARES predictions for each llm_name.
    """
    ares_results = {}
    for llm_name in llm_names:
        prediction_file_path = os.path.join(aura_config["LLM_prediction_folder_directory"], f"{llm_name}_ares_predictions.tsv")
        if not os.path.exists(prediction_file_path):
            raise FileNotFoundError(f"Prediction file for {llm_name} not found: {prediction_file_path}")
        ares_results[llm_name] = pd.read_csv(prediction_file_path, sep='\t')["ARES_Answer_Relevance_Prediction"].tolist()
    return ares_results