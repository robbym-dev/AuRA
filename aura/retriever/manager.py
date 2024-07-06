# aura/retriever/manager.py

import os
import pandas as pd
from .retriever import BM25, SentenceBERT_Retriever, MiniLM_Retriever, BM25_Elastic_Jina
from typing import Dict, Tuple
from tqdm import tqdm
from ares import ARES

class DocumentRetrievalManager:
    def __init__(self, config, retriever_name):
        """
        Initializes the DocumentRetrievalManager with the given configuration and retriever name.

        Args:
            config (dict): Configuration settings for the retriever.
            retriever_name (str): The name of the retriever to be used.
        """
        self.config = config
        self.retriever_name = retriever_name
        self.retriever = self.initialize_retriever()

    def initialize_retriever(self):
        """
        Initializes the appropriate retriever based on the retriever name.

        Returns:
            Retriever: An instance of the selected retriever class.

        Raises:
            ValueError: If the retriever name is not recognized.
        """
        retriever_classes = {
            'bm25_elastic_jina': BM25_Elastic_Jina,  # Add the new retriever class
            'bm25': BM25,
            'sentencebert': SentenceBERT_Retriever,
            'minilm': MiniLM_Retriever
        }

        retriever_class = retriever_classes.get(self.retriever_name.lower())
        if not retriever_class:
            raise ValueError(f"Retriever {self.retriever_name} not found")

        print(f"Initializing {self.retriever_name} retriever...")
        retriever = retriever_class(self.config)
        retriever.initialize_retriever()
        return retriever

    def search(self, query):
        """
        Searches for the given query using the initialized retriever.

        Args:
            query (str): The query string to search for.

        Returns:
            str: The top search result for the given query.
        """
        return self.retriever.search([query])[0]

def find_best_retriever(config: Dict) -> Tuple[str, str]:
    """
    Finds the best retriever based on the given configuration.

    Args:
        config (Dict): Configuration dictionary for the retriever.

    Returns:
        Tuple[str, str]: A tuple containing the name of the best retriever and the path to its document file.
    """
    queries_df = pd.read_csv(config["queries"], sep='\t')
    queries = queries_df['Query'].tolist()
    
    best_retriever_docs_file = os.path.join(config["LLM_prediction_folder_directory"], "best_retriever_docs.tsv")
    
    if os.path.exists(best_retriever_docs_file):
        print(f"Best retriever docs file already exists: {best_retriever_docs_file}")
        return "pre-evaluated", best_retriever_docs_file
    
    print(f"Conducting retrieval on {len(queries)} queries...")

    retrievers = {
        "bm25_elastic_jina": BM25_Elastic_Jina(config),
        "bm25": BM25(config),
        "sentence_bert": SentenceBERT_Retriever(config),
        "minilm": MiniLM_Retriever(config)
    }

    best_retriever_name = None
    best_retriever_score = float('-inf')

    for name, retriever in retrievers.items():
        print(f"Evaluating {name} retriever...")
        retriever.initialize_retriever()
        documents = retriever.search(queries)

        results_data = []
        for query, doc_list in zip(queries, documents):
            for doc in doc_list:
                results_data.append({"Query": query, "Document": doc})

        breakpoint()
        retrieved_docs_file = f"retrieved_docs_{name}.csv"
        pd.DataFrame(results_data).to_csv(retrieved_docs_file, index=False, sep='\t')
        
        # Print statement to confirm saving
        print(f"Retrieved documents for {name} saved to {retrieved_docs_file}")

        # Evaluate retriever using ARES
        ppi_config = {
            "evaluation_datasets": [retrieved_docs_file],
            "few_shot_examples_filepaths": [config["few_shot_examples_file_path"]],
            "checkpoints": [config["checkpoint"]],
            "rag_type": config["rag_type"],
            "labels": ["Context_Relevance_Label"],
            "gold_label_paths": [config["gold_label_path"]],
        }
        ares = ARES(ppi=ppi_config)
        results = ares.evaluate_RAG()
        print(f"Results for {name} retriever: {results}")

        if results:
            first_result = results[0]
            confidence_interval = first_result.get("ARES_Confidence_Interval", [0, 0])
            print(f"Confidence Interval for {name}: {confidence_interval}")
            score = confidence_interval[0]

            if score > best_retriever_score:
                best_retriever_name = name
                best_retriever_score = score

    print(f"Best retriever: {best_retriever_name} with score: {best_retriever_score}")

    # Use the best retriever to generate the final document set
    best_retriever = retrievers[best_retriever_name]
    documents = best_retriever.search(queries)
    results_data = []
    for query, doc_list in zip(queries, documents):
        for doc in doc_list:
            results_data.append({"Query": query, "Document": doc})
    pd.DataFrame(results_data).to_csv(best_retriever_docs_file, index=False, sep='\t')

    return best_retriever_name, best_retriever_docs_file