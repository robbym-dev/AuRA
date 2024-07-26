import os
import pandas as pd
from typing import Dict
import random

class GroundTruthEvaluator:
    def __init__(self, ground_truth_file: str, retriever_results_dir: str):
        self.ground_truth = self._load_ground_truth(ground_truth_file)
        self.retriever_results = self._load_retriever_results(retriever_results_dir)
        
    def _load_ground_truth(self, file_path: str) -> pd.DataFrame:
        print(f"Loading ground truth file: {file_path}")
        df = pd.read_csv(file_path, sep='\t')
        df = df[['Query', 'Document', 'Context_Relevance_Label']]
        df = df[df['Context_Relevance_Label'] == 1]
        return df
    
    def _load_retriever_results(self, directory: str) -> Dict[str, pd.DataFrame]:
        results = {}
        print(f"Loading retriever results from directory: {directory}")
        for filename in os.listdir(directory):
            if filename.endswith('_results.tsv'):
                retriever_name = filename.replace('_results.tsv', '')
                file_path = os.path.join(directory, filename)
                print(f"Loading results for {retriever_name} from file: {file_path}")
                df = pd.read_csv(file_path, sep='\t')
                results[retriever_name] = df
        return results

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        evaluation_results = {}
        
        for retriever_name, retriever_df in self.retriever_results.items():
            print(f"Evaluating {retriever_name}...")
            
            total_precision = 0
            total_recall = 0
            total_queries = len(retriever_df['Query'].unique())
            total_documents = len(retriever_df['Document'].unique())
            
            print(f"Evaluating {retriever_name} over {total_queries} unique queries and {total_documents} unique documents...")
            
            inaccurate_queries = []
            wrong_docs_count = 0
            no_docs_count = 0
            
            for query in retriever_df['Query'].unique():
                retrieved_docs = set(retriever_df[retriever_df['Query'] == query]['Document'])
                relevant_docs = set(self.ground_truth[self.ground_truth['Query'] == query]['Document'])
                
                true_positives = len(retrieved_docs.intersection(relevant_docs))
                precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
                recall = true_positives / len(relevant_docs) if relevant_docs else 0
                
                total_precision += precision
                total_recall += recall
                
                if true_positives == 0:
                    if "No Relevant Information Available" in retrieved_docs:
                        no_docs_count += 1
                    else:
                        wrong_docs_count += 1
                    inaccurate_queries.append((query, retrieved_docs, relevant_docs))
            
            avg_precision = total_precision / total_queries if total_queries > 0 else 0
            avg_recall = total_recall / total_queries if total_queries > 0 else 0
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            evaluation_results[retriever_name] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': f1_score,
                'inaccurate_queries': inaccurate_queries,
                'wrong_docs_count': wrong_docs_count,
                'no_docs_count': no_docs_count
            }
            
            print(f"{retriever_name}: Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {f1_score:.4f}")
            print(f"Incorrect documents: {wrong_docs_count}, No documents found: {no_docs_count}")
            
            # Print 5 random inaccurate queries only for AuRA
            if retriever_name == 'AuRA':
                sample_size = min(5, len(inaccurate_queries))
                random_inaccurate_queries = random.sample(inaccurate_queries, sample_size)
                print(f"\n5 random queries with completely inaccurate retrieval for AuRA:")
                for query, retrieved, relevant in random_inaccurate_queries:
                    print(f"- Query: {query}")
                    print(f"  Retrieved: {retrieved}")
                    print(f"  Relevant: {relevant}")
                    print()
        
        return evaluation_results
    
    def print_results(self, results: Dict[str, Dict[str, float]]):
        print("\nEvaluation Results:")
        print("-------------------")
        for retriever, metrics in results.items():
            print(f"{retriever}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  Incorrect documents: {metrics['wrong_docs_count']}")
            print(f"  No documents found: {metrics['no_docs_count']}")