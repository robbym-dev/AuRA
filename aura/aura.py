# aura.py

from typing import List, Union, get_origin, get_args
from aura.retriever.manager import DocumentRetrievalManager
from aura.classifier.classifier_manager import train_llm_classifier
from aura.routing.llm_routing_manager import query_and_evaluate_llms, run_llm_routing_pipeline


class AURA:
    REQUIRED_BUT_HAS_DEFAULT = object()

    config_spec = {
        "retriever": {
            "ground_truth_path": (str, None),
            "LLM_prediction_folder_directory": (str, None),
            "few_shot_examples_file_path": (str, None),
            "rag_type": (str, None),
            "checkpoint": (str, None),
            "gold_label_path": (str, None),
            "cache_dir": (str, "./ares_cache"),
            "top_k": (int, 3),
            "rank_constant": (int, "None"),  # New RRF parameter
            "window_size": (int, "None"),   # New RRF parameter
            "documents": (str, "None"),
            "queries": (str, "None")
        },
        "llm_generation": { 
            "LLM_prediction_folder_directory": (str, None), 
            "few_shot_examples_file_path": (str, None), 
            "checkpoint": (str, None),
            "rag_type": (str, None), 
            "gold_label_path": (str, None),
            "debug_mode": (bool, True)
        },
        "classifier": {
            "LLM_prediction_folder_directory": (str, None),
            "model_dir": (str, None),
            "model_choice": (str, "microsoft/deberta-v3-large"),  # Optional, with default
            "num_epochs": (int, 10),  # Optional, with default
            "learning_rate": (float, 5e-6),  # Optional, with default
            "patience_value": (int, 3),  # Optional, with default
            "assigned_batch_size": (int, 1),  # Optional, with default
            "gradient_accumulation_multiplier": (int, 32),  # Optional, with default
            "max_length": (int, 1024), # Optional, with default
            "validation_split": (float, 0.2),
            "llm_names": (list, ["GPT-4", "Claude Opus", "Llama"]),
        },
        "llm_routing": {
            "queries": (str, None), 
            "documents": (str, None),
            "LLM_prediction_folder_directory": (str, None),
            "model_dir": (str, None), 
            "checkpoints": (list, None),
            "model_choice": (str,  "microsoft/deberta-v3-large"), 
            "llm_names": (list, ["GPT-4", "Claude Opus", "Llama"]),
            "top_k": (int, 1)  # Optional, with default
        }
    }

    def __init__(self, retriever_config={}, llm_generation_config={}, classifier_config={}, llm_routing_config={}):
        self.retriever_config = self.prepare_config("retriever", retriever_config)
        self.llm_generation_config = self.prepare_config("llm_generation", llm_generation_config)
        self.classifier_config = self.prepare_config("classifier", classifier_config)
        self.llm_routing_config = self.prepare_config("llm_routing", llm_routing_config)

    def prepare_config(self, component_name, user_config):
        if not user_config:
            return {}
        component = self.config_spec[component_name]
        prepared_config = {}
        for param, (param_type, default) in component.items():
            if param in user_config:
                value = user_config[param]
                # Check if the param_type is a generic type
                origin_type = get_origin(param_type)
                if origin_type:
                    # If it is a generic type, we need to check the origin and arguments
                    if not isinstance(value, origin_type):
                        raise TypeError(f"Parameter '{param}' for {component_name} is expected to be of type {param_type}, received {type(value)} instead.")
                    # Check if value has the correct subtypes
                    param_args = get_args(param_type)
                    if param_args and not all(isinstance(v, param_args[0]) for v in value):
                        raise TypeError(f"Parameter '{param}' for {component_name} is expected to have elements of type {param_args[0]}, received elements of type {type(value[0])} instead.")
                else:
                    # Regular type checking
                    if not isinstance(value, param_type):
                        raise TypeError(f"Parameter '{param}' for {component_name} is expected to be of type {param_type.__name__}, received {type(value).__name__} instead.")
                prepared_config[param] = value
            elif default is not None:
                prepared_config[param] = default
            else:
                raise ValueError(f"Missing required parameter '{param}' for {component_name}.")
        return prepared_config

    def run_aura_pipeline(self):
        print("Starting the AuRA pipeline...")
        
        # Document Retrieval
        retrieval_manager = DocumentRetrievalManager(self.retriever_config)
        retrieval_results = retrieval_manager.run()
        
        # LLM Generation and Evaluation
        best_llm_name = query_and_evaluate_llms(self.llm_generation_config, retrieval_results['best_documents_file'])
        print(f"Best LLM: {best_llm_name}")
        
        # Classifier Training
        train_llm_classifier(self.classifier_config)
        
        # LLM Routing
        routing_results = run_llm_routing_pipeline(self.llm_routing_config)
        
        return {
            **retrieval_results,
            "best_llm": best_llm_name,
            **routing_results
        }