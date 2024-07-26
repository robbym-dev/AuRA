import os
import pandas as pd
from typing import Dict, List, Union, Set
import logging
from collections import defaultdict
from .retriever import BM25, SentenceBERT_Retriever, MiniLM_Retriever, BM25_Elastic_Jina, ReciprocalRankFusionRetriever, AuRADynamicRetriever
from ares import ARES
from .ground_truth_scoring import GroundTruthEvaluator
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentRetrievalManager:
    def __init__(self, config: Dict):
        self.config = config
        self.top_k = config.get('top_k', 3)
        self.output_dir = config.get('LLM_prediction_folder_directory', './retriever_results')
        os.makedirs(self.output_dir, exist_ok=True)
        self.ground_truth_path = config.get('ground_truth_path')
        self.queries_count = config.get('queries_count', 1000)
        self.documents = None
        self.queries = None
        self.retrievers = {
            'BM25': BM25(config),
            'SentenceBERT': SentenceBERT_Retriever(config),
            'MiniLM': MiniLM_Retriever(config),
            'BM25_Elastic_Jina': BM25_Elastic_Jina(config)
        }
        self.rrf_retriever = ReciprocalRankFusionRetriever(config, list(self.retrievers.values()))
        self.ground_truth_evaluator = None
    
    def load_data(self):
        df = pd.read_csv(self.ground_truth_path, sep='\t')
        
        # Filter out rows where Context_Relevance_Label is NA
        valid_df = df.dropna(subset=['Context_Relevance_Label'])
        
        # Filter out documents with 50 or fewer words
        valid_df['word_count'] = valid_df['Document'].apply(lambda x: len(str(x).split()))
        valid_df = valid_df[valid_df['word_count'] > 50]
        
        # Load all unique documents from the filtered dataset
        self.documents = valid_df['Document'].unique().tolist()
        
        # Load only queries with valid ground truth and associated with valid documents, limited to queries_count
        valid_queries = valid_df['Query'].unique()
        self.queries = valid_queries[:self.queries_count]
        
        # Store valid query-document pairs for reference
        self.valid_pairs = valid_df[['Query', 'Document']].values.tolist()
        
        # Filter out queries associated with documents having fewer than 50 words
        self.queries = [query for query in self.queries if query in valid_df['Query'].values]
        
        logging.info(f"Loaded {len(self.documents)} unique documents and {len(self.queries)} queries with valid ground truth.")
        logging.info(f"Total valid query-document pairs: {len(self.valid_pairs)}")
        
    def run(self) -> Dict:
        self.load_data()
        existing_files = self._check_existing_files()
        
        results = self._load_existing_results(existing_files)
        missing_retrievers = set(self.retrievers.keys()) - set(existing_files)
        
        if missing_retrievers:
            new_results = self._run_retrievers(self.queries, missing_retrievers)
            results.update(new_results)
            self._save_results(self.queries, new_results)

        # Run RRF retriever only if it doesn't exist
        if 'RRF' not in existing_files:
            rrf_results = self.rrf_retriever.search(self.queries)
            results['RRF'] = {query: docs for query, docs in zip(self.queries, rrf_results)}
            self._save_results(self.queries, {'RRF': results['RRF']})

        results_files = {name: os.path.join(self.output_dir, f"{name}_ares_scores.tsv") for name in results.keys()}
        
        # Run ARES evaluation for all retrievers including RRF
        self._check_and_run_ares_evaluation(results)
        
        # Initialize and run AuRA Dynamic Retriever
        aura_retriever = AuRADynamicRetriever(self.output_dir, self.top_k)
        aura_output_file = os.path.join(self.output_dir, "AuRA_results.tsv")
        aura_retriever.save_results(aura_output_file)
        results['AuRA'] = self._load_existing_results(['AuRA'])['AuRA']
        results_files['AuRA'] = aura_output_file

        # Run ARES evaluation for AuRA
        # self._check_and_run_ares_evaluation({'AuRA': results['AuRA']})

        # Initialize and run Ground Truth Evaluator for available retriever results
        expected_retrievers = set(['BM25', 'SentenceBERT', 'MiniLM', 'BM25_Elastic_Jina', 'RRF', 'AuRA'])
        available_retrievers = set(results.keys())
        missing_retrievers = expected_retrievers - available_retrievers

        if missing_retrievers:
            logging.warning(f"The following retrievers are missing: {', '.join(missing_retrievers)}. Proceeding with evaluation for available retrievers.")

        self.ground_truth_evaluator = GroundTruthEvaluator(self.ground_truth_path, self.output_dir)
        evaluation_scores = self.ground_truth_evaluator.evaluate()
        self.ground_truth_evaluator.print_results(evaluation_scores)

        return {
            "results_files": results_files,
            "evaluation_scores": evaluation_scores,
            "missing_retrievers": list(missing_retrievers)
        }

    def _check_existing_files(self) -> List[str]:
        retriever_names = ['BM25', 'SentenceBERT', 'MiniLM', 'BM25_Elastic_Jina', 'RRF', 'AuRA']
        existing_files = []
        for name in retriever_names:
            results_file = os.path.join(self.output_dir, f"{name}_results.tsv")
            if os.path.exists(results_file):
                existing_files.append(name)
        return existing_files

    def _load_existing_results(self, existing_files: List[str]) -> Dict[str, Dict[str, str]]:
        results = {}
        for name in existing_files:
            file_path = os.path.join(self.output_dir, f"{name}_results.tsv")
            df = pd.read_csv(file_path, sep='\t')
            df = df[df['Query'].isin(self.queries)]  # Filter results to match current queries
            df['word_count'] = df['Document'].apply(lambda x: len(str(x).split()))
            df = df[df['word_count'] > 50]  # Filter out documents with 50 or fewer words
            results[name] = df.groupby('Query')['Document'].apply(list).to_dict()
        return results

    def _run_retrievers(self, queries: List[str], missing_retrievers: Set[str]) -> Dict[str, Dict[str, str]]:
        results = {}
        for name in missing_retrievers:
            print(f"\nRunning {name} retriever...")
            retriever = self.retrievers[name]
            retriever.initialize_retriever(self.documents)
            retriever_results = retriever.search(queries)
            results[name] = {query: docs for query, docs in zip(queries, retriever_results)}
        return results
    
    def _save_results(self, queries: List[str], results: Dict[str, Union[Dict[str, List[str]], List[Dict[str, str]]]]):
        for name, retriever_results in results.items():
            output_file = os.path.join(self.output_dir, f"{name}_results.tsv")
            
            # Check if results are in dictionary format
            if isinstance(retriever_results, dict):
                df = pd.DataFrame([(query, doc) for query, docs in retriever_results.items() for doc in docs], columns=['Query', 'Document'])
            else:  # For AuRA results
                df = pd.DataFrame(retriever_results)
            
            # Log the initial results
            logging.info(f"Initial results for {name}: {df.shape[0]} rows")
            missing_queries_initial = set(queries) - set(df['Query'].unique())
            if missing_queries_initial:
                logging.warning(f"Initial missing queries for {name}: {missing_queries_initial}")

            # Ensure only results for current queries are saved
            df = df[df['Query'].isin(queries)]
            
            # Add word count and filter
            df['word_count'] = df['Document'].apply(lambda x: len(str(x).split()))
            df = df[df['word_count'] > 50]  # Filter out documents with 50 or fewer words

            # Log the filtered results
            logging.info(f"Filtered results for {name}: {df.shape[0]} rows after filtering by word count")
            missing_queries_filtered = set(queries) - set(df['Query'].unique())
            if missing_queries_filtered:
                logging.warning(f"Filtered missing queries for {name}: {missing_queries_filtered}")

            # Save the DataFrame to a TSV file
            df.to_csv(output_file, sep='\t', index=False)
            logging.info(f"Results for {name} saved to {output_file}")

    def _check_and_run_ares_evaluation(self, results: Dict[str, Dict[str, List[str]]]):
        for name in results.keys():
            input_file = os.path.join(self.output_dir, f"{name}_results.tsv")
            output_file = os.path.join(self.output_dir, f"{name}_ares_scores.tsv")
            
            if name == "AuRA":
                print(f"Skipping ARES evaluation for AuRA...")
                continue
            
            if not os.path.exists(output_file):
                print(f"Running ARES evaluation for {name}...")
                ppi_config = {
                    "evaluation_datasets": [input_file],
                    "few_shot_examples_filepaths": [self.config["few_shot_examples_file_path"]],
                    "checkpoints": [self.config["checkpoint"]],
                    "rag_type": self.config["rag_type"],
                    "labels": ["Context_Relevance_Label"],
                    "gold_label_paths": [self.config["gold_label_path"]],
                    "prediction_filepaths": [output_file]
                }
                ARES(ppi=ppi_config).evaluate_RAG()
            else:
                print(f"ARES evaluation for {name} already exists. Skipping...")