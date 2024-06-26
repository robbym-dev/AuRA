# AuRA



# To run AuRA Pipeline, run following:
```python
aura_config = {
    "documents": "<file path to documents>",
    "queries": "<file path to queries>",
    "few_shot_examples_file_path": "<file path to few shot examples>",
    "rag_type": "question_answering",
    "checkpoints": ["<file path to context relevance checkpoint>", "<file path to abnswer relevance checkpoint>"],
    "gold_label_path": "<file path to gold labels>",
    "LLM_prediction_folder_directory": "<file path to save LLM predictions>",
    "classifier_training_data": "<file path to classifier training data>",
    "model_dir": "<file path to save model checkpoint>",
    "model_choice": "microsoft/deberta-v3-large",
    "classifier_validation_data": "<file path to classifier validation data>",
    "top_k": 1, 
    "max_queries": "<max number of queries>", # OPTIONAL
    "num_epochs": 10, # OPTIONAL
    "patience_value": 3, # OPTIONAL
    "learning_rate": 5e-6, # OPTIONAL
    "assigned_batch_size": 1, # OPTIONAL
    "gradient_accumulation_multiplier": 32, # OPTIONAL
}

run_aura_pipeline(aura_config)
```