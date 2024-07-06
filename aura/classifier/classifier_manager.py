# classifier_manager.py

import os
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .llm_classifier import CustomBERTModel, prepare_data_for_training, train_model, save_model

def train_llm_classifier(config):
    # Load and preprocess data
    classifier_data = prepare_classifier_data_from_files(config, config["llm_names"])
    
    # Combine data into a single DataFrame
    data = []
    for i in range(len(classifier_data["queries"])):
        for llm_name in config["llm_names"]:
            data.append({
                "Query": classifier_data["queries"][i],
                "Document": classifier_data["documents"][i],
                "Answer": classifier_data["answers"][llm_name][i],
                "ARES_Score": classifier_data["ares_results"][llm_name][i],
                "LLM": llm_name
            })
    
    df = pd.DataFrame(data)
    
    # Determine the best LLM for each query-document pair
    df['Best_LLM'] = df.groupby(["Query", "Document"])['ARES_Score'].transform(lambda x: x.idxmax())
    df = df[df['LLM'] == df['Best_LLM']].reset_index(drop=True)
    
    # Create labels based on the LLM names
    llm_to_label = {llm: i for i, llm in enumerate(config["llm_names"])}
    df['label'] = df['Best_LLM'].map(llm_to_label)
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(df, test_size=config["validation_split"], random_state=42)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["model_choice"])
    model = CustomBERTModel(number_of_labels=len(config["llm_names"]), model_choice=config["model_choice"])
    
    # Prepare datasets
    train_dataset = prepare_data_for_training(train_data, tokenizer, config["max_length"], label_column="label")
    val_dataset = prepare_data_for_training(val_data, tokenizer, config["max_length"], label_column="label")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    
    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        config["num_epochs"], 
        config["learning_rate"], 
        device, 
        config["gradient_accumulation_multiplier"]
    )
    
    # Save the model
    save_model(trained_model, tokenizer, config["model_dir"])
    print(f"Model saved to {config['model_dir']}")

    # return trained_model, tokenizer

def prepare_classifier_data_from_files(aura_config, llm_names):
    """
    Prepares classifier data from files.

    Args:
        aura_config (dict): Configuration dictionary containing necessary parameters.
        llm_names (list): List of LLM names.

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
    Loads ARES predictions for the given LLM names.

    Args:
        aura_config (dict): Configuration dictionary containing necessary parameters.
        llm_names (list): List of LLM names.

    Returns:
        dict: A dictionary with LLM names as keys and their ARES predictions as values.
    """
    ares_results = {}
    for llm_name in llm_names:
        prediction_file_path = os.path.join(aura_config["LLM_prediction_folder_directory"], f"{llm_name}_ares_predictions.tsv")
        if not os.path.exists(prediction_file_path):
            raise FileNotFoundError(f"Prediction file for {llm_name} not found: {prediction_file_path}")
        ares_results[llm_name] = pd.read_csv(prediction_file_path, sep='\t')["ARES_Answer_Relevance_Prediction"].tolist()
    return ares_results