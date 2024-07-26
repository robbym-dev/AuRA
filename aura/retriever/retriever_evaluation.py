# retriever_evaluation.py

import pandas as pd
import numpy as np
from typing import Dict, List
from tqdm import tqdm
from prettytable import PrettyTable

class RetrieverEvaluator:
    def __init__(self, nq_data_path: str, config: Dict):
        self.nq_data = pd.read_csv(nq_data_path, sep='\t')
        self.config = config
        self.top_k = config.get('top_k', 3)
        self.prepare_data()

    def prepare_data(self):
        # Filter out rows where Context_Relevance_Label is empty or 'N/A'
        self.filtered_data = self.nq_data[
            (self.nq_data['Context_Relevance_Label'] != 'N/A') & 
            (self.nq_data['Context_Relevance_Label'].notna()) &
            (self.nq_data['Context_Relevance_Label'] != '')
        ].copy()
        
        # Convert Context_Relevance_Label to int
        self.filtered_data['Context_Relevance_Label'] = self.filtered_data['Context_Relevance_Label'].astype(int)
        
        # Create queries dataset
        self.queries = self.filtered_data[['Query']].drop_duplicates()
        
        # Create ground truth dataset
        self.ground_truth = self.filtered_data[['Query', 'Document', 'Context_Relevance_Label']]

    def get_queries(self):
        return self.queries['Query'].tolist()

    def evaluate_retrievers(self, retriever_results: Dict[str, List[List[str]]]) -> Dict[str, Dict[str, float]]:
        results = {}
        for retriever_name, retrieved_docs in retriever_results.items():
            results[retriever_name] = self.evaluate_retriever(retriever_name, retrieved_docs)
        return results

    def evaluate_retriever(self, retriever_name: str, retrieved_docs: List[List[str]]) -> Dict[str, float]:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0

        for query, docs in tqdm(zip(self.queries['Query'], retrieved_docs), total=len(self.queries), desc=f"Evaluating {retriever_name}"):
            ground_truth = self.ground_truth[self.ground_truth['Query'] == query]
            relevant_docs = set(ground_truth[ground_truth['Context_Relevance_Label'] == 1]['Document'].values)
            irrelevant_docs = set(ground_truth[ground_truth['Context_Relevance_Label'] == 0]['Document'].values)
            
            retrieved = set(docs[:self.top_k])
            
            true_positives += len(relevant_docs.intersection(retrieved))
            false_positives += len(retrieved.intersection(irrelevant_docs))
            false_negatives += len(relevant_docs - retrieved)
            true_negatives += len(irrelevant_docs - retrieved)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

        return {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Accuracy": accuracy,
            "True Positives": true_positives,
            "False Positives": false_positives,
            "False Negatives": false_negatives,
            "True Negatives": true_negatives
        }

    def print_results(self, results: Dict[str, Dict[str, float]]):
        table = PrettyTable()
        table.field_names = ["Retriever", "Precision", "Recall", "F1 Score", "Accuracy", "TP", "FP", "FN", "TN"]
        
        for retriever, metrics in results.items():
            table.add_row([
                retriever,
                f"{metrics['Precision']:.4f}",
                f"{metrics['Recall']:.4f}",
                f"{metrics['F1 Score']:.4f}",
                f"{metrics['Accuracy']:.4f}",
                metrics['True Positives'],
                metrics['False Positives'],
                metrics['False Negatives'],
                metrics['True Negatives']
            ])
        
        print("\nRetriever Evaluation Results:")
        print(table)