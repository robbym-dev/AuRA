# aura/retriever/retriever.py

import pandas as pd
from collections import Counter
from datasets import Dataset
from typing import List, Dict
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers 
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('detailed_logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class Retriever:
    def __init__(self, config):
        self.config = config
        self.top_k = config.get("top_k", 3)
        self.documents = None
        self.document_index = None

    def initialize_retriever(self, documents):
        self.documents = documents
        print(f"Initializing {self.__class__.__name__} retriever...")
        print(f"Loaded {len(self.documents)} documents for {self.__class__.__name__} retriever.")
        self.create_index()
        print(f"{self.__class__.__name__} index created.")

    def create_index(self):
        raise NotImplementedError("create_index method must be implemented in subclasses")

    def search(self, queries):
        raise NotImplementedError("search method must be implemented in subclasses")

class BM25(Retriever):
    def __init__(self, config):
        super().__init__(config)

    def create_index(self):
        tokenized_documents = [doc.split() for doc in self.documents]
        self.document_index = BM25Okapi(tokenized_documents)

    def search(self, queries):
        total_retrieved_docs = []
        for query in tqdm(queries, desc="BM25 retrieval"):
            tokenized_query = query.split()
            scores = self.document_index.get_scores(tokenized_query)
            sorted_documents = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)
            top_k_docs = [doc_score_tuple[0] for doc_score_tuple in sorted_documents][:self.top_k]
            total_retrieved_docs.append(top_k_docs)
        return total_retrieved_docs

class SentenceBERT_Retriever(Retriever):
    """
    Sentence-BERT-based document retriever.
    """
    def __init__(self, config):
        super().__init__(config)
        self.retrieval_model = None

    def create_index(self):
        self.retrieval_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.document_index = self.retrieval_model.encode(self.documents, convert_to_tensor=True, show_progress_bar=True)

    def search(self, queries):
        total_retrieved_docs = []
        for query in tqdm(queries, desc="SentenceBERT retrieval"):
            query_embedding = self.retrieval_model.encode(query, convert_to_tensor=True)
            scores = cosine_similarity(query_embedding.cpu().numpy().reshape(1, -1), self.document_index.cpu().numpy())[0]
            top_k_indices = scores.argsort()[-self.top_k:][::-1]
            top_k_docs = [self.documents[i] for i in top_k_indices]
            total_retrieved_docs.append(top_k_docs)
        return total_retrieved_docs

class MiniLM_Retriever(Retriever):
    """
    MiniLM-based document retriever.
    """
    def __init__(self, config):
        """
        Initializes the MiniLM retriever with the given configuration.

        Args:
            config (dict): Configuration settings for the retriever.
        """
        super().__init__(config)
        self.retrieval_model = None

    def create_index(self):
        self.retrieval_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.document_index = self.retrieval_model.encode(self.documents, convert_to_tensor=True, show_progress_bar=True)

    def search(self, queries):
        total_retrieved_docs = []
        for query in tqdm(queries, desc="MiniLM retrieval"):
            query_embedding = self.retrieval_model.encode(query, convert_to_tensor=True)
            scores = cosine_similarity(query_embedding.cpu().numpy().reshape(1, -1), self.document_index.cpu().numpy())[0]
            top_k_indices = scores.argsort()[-self.top_k:][::-1]
            top_k_docs = [self.documents[i] for i in top_k_indices]
            total_retrieved_docs.append(top_k_docs)
        return total_retrieved_docs

from sklearn.metrics.pairwise import cosine_similarity
from elasticsearch import Elasticsearch, helpers  
import os
class BM25_Elastic_Jina(Retriever):
    """
    A class that integrates BM25 retrieval with Elastic Search and reranking with Jina embeddings.
    """
    def __init__(self, config):
        """
        Initializes the pipeline with the given configuration.

        Args:
            config (dict): Configuration settings for the pipeline.
        """
        super().__init__(config)
        self.top_k = config.get('top_k', 3)
        self.documents_path = config.get('documents')
        self.elastic_host = config.get('elastic_host', 'localhost')
        self.elastic_port = config.get('elastic_port', 9200)
        self.index_name = config.get('index_name', 'documents')
        self.elastic_user = config.get('elastic_user', 'elastic')
        self.elastic_password = config.get('elastic_password', '<redacted>')
        self.elastic_scheme = config.get('elastic_scheme', 'https')
        self.elastic_ca_cert = config.get('elastic_ca_cert', '/future/u/manihani/elasticsearch-8.14.3/config/certs/http_ca.crt')  # Adjust this path
        self.es = None
        self.jina_model = None

        # self.initialize_retriever()
        # self.initialize_elasticsearch()
        # self.initialize_jina_embeddings()
        
    def initialize_retriever(self, documents):
        super().initialize_retriever(documents)
        self.create_index()
        self.initialize_elasticsearch()
        self.initialize_jina_embeddings()

    def create_index(self):
        tokenized_documents = [doc.split() for doc in self.documents]
        self.document_index = BM25Okapi(tokenized_documents)

    def initialize_elasticsearch(self):
        self.es = Elasticsearch(
            [f'{self.elastic_scheme}://{self.elastic_host}:{self.elastic_port}'],
            http_auth=(self.elastic_user, self.elastic_password),
            ca_certs=self.elastic_ca_cert
        )
        
        if not self.es.indices.exists(index=self.index_name):
            settings = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        }
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=settings)
        
        actions = [
            {
                "_index": self.index_name,
                "_source": {"content": doc}
            }
            for doc in self.documents
        ]
        helpers.bulk(self.es, actions)
        print("Elasticsearch index created and documents indexed.")
        
    def check_elasticsearch_connection(self):
        """
        Checks if the Elasticsearch server is reachable.
        """
        try:
            self.es.ping()
            print("Successfully connected to Elasticsearch")
            return True
        except Exception as e:
            print(f"Error connecting to Elasticsearch: {e}")
            return False

    def search_bm25(self, queries):
        """
        Searches for the given queries using the BM25 index.

        Args:
            queries (list): List of query strings.

        Returns:
            list: List of top-k retrieved documents for each query.
        """
        total_retrieved_docs = []
        for query in queries:
            if pd.isna(query) or not query.strip():
                print(f"Skipping empty or NaN query: {query}")
                total_retrieved_docs.append([])
                continue
            tokenized_query = query.split()
            scores = self.document_index.get_scores(tokenized_query)
            sorted_documents = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)
            top_k_docs = [doc_score_tuple[0] for doc_score_tuple in sorted_documents][:self.top_k]
            total_retrieved_docs.append(top_k_docs)
        return total_retrieved_docs

    def search_elasticsearch(self, query):
        """
        Searches for the given query using Elasticsearch.

        Args:
            query (str): The query string.

        Returns:
            list: List of retrieved documents.
        """
        if pd.isna(query) or not query.strip():
            print(f"Skipping empty or NaN query: {query}")
            return []
        body = {
            "query": {
                "match": {
                    "content": query
                }
            },
            "size": self.top_k
        }
        results = self.es.search(index=self.index_name, body=body)
        documents = [hit['_source']['content'] for hit in results['hits']['hits']]
        return documents

    def initialize_jina_embeddings(self):
        print("Initializing Jina embeddings model...")
        self.jina_model = SentenceTransformer('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
        self.jina_model.max_seq_length = 8192

    def get_embeddings(self, texts):
        try:
            embeddings = self.jina_model.encode(texts, convert_to_tensor=True).cpu().numpy()
            return embeddings
        except Exception as e:
            print(f"Error in get_embeddings: {str(e)}")
            print(f"Input texts: {texts}")
            raise
        
    def rerank_documents(self, query_embedding, docs, doc_embeddings):
        """
        Reranks the documents based on similarity to the query embedding.

        Args:
            query_embedding (ndarray): The embedding of the query.
            docs (list): List of document texts.
            doc_embeddings (list): List of embeddings of the documents.

        Returns:
            list: Reranked list of documents.
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if len(doc_embeddings.shape) == 1:
            doc_embeddings = doc_embeddings.reshape(1, -1)
        
        try:
            similarity_scores = cosine_similarity(query_embedding, doc_embeddings)[0]
            reranked_docs = sorted(zip(docs, similarity_scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in reranked_docs]
        except ValueError as e:
            print(f"Error in rerank_documents: {str(e)}")
            print("Query embedding:", query_embedding)
            print("Document embeddings:", doc_embeddings)
            raise

    def search(self, queries):
        """
        Executes the full search pipeline: BM25 retrieval, Elasticsearch retrieval, embedding, and reranking.

        Args:
            queries (list): List of query strings.

        Returns:
            list: List of reranked documents for each query.
        """
        total_retrieved_docs = []
        for query in tqdm(queries, desc="BM25_Elastic_Jina retrieval"):
            try:
                bm25_docs = self.search_bm25([query])[0]
                elastic_docs = self.search_elasticsearch(query)
                combined_docs = bm25_docs + elastic_docs
                
                unique_docs = []
                seen = set()
                for doc in combined_docs:
                    if doc not in seen:
                        unique_docs.append(doc)
                        seen.add(doc)
                
                query_embedding = self.get_embeddings([query])
                doc_embeddings = self.get_embeddings(unique_docs)
                
                reranked_docs = self.rerank_documents(query_embedding, unique_docs, doc_embeddings)
                
                total_retrieved_docs.append(reranked_docs[:self.top_k])
            except Exception as e:
                print(f"Error processing query: {query}")
                print(f"Error details: {str(e)}")
                total_retrieved_docs.append([])
        return total_retrieved_docs
    
class ReciprocalRankFusionRetriever(Retriever):
    def __init__(self, config, retrievers):
        self.k = int(config['rank_constant'])
        self.window_size = int(config['window_size'])
        self.top_k = int(config.get('top_k', 3))
        self.retrievers = retrievers

    def initialize_retriever(self):
        for retriever in self.retrievers:
            retriever.initialize_retriever()

    def search(self, queries):
        all_results = []
        for query in tqdm(queries, desc="RRF retrieval"):
            document_scores = {}
            for retriever in self.retrievers:
                results = retriever.search([query])[0]
                for rank, doc in enumerate(results[:self.window_size], start=1):
                    if doc not in document_scores:
                        document_scores[doc] = 0
                    document_scores[doc] += 1 / (self.k + rank)

            sorted_documents = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
            all_results.append([doc for doc, score in sorted_documents[:self.top_k]])
        return all_results
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os
import pandas as pd
from collections import defaultdict
import random

class AuRADynamicRetriever:
    def __init__(self, directory, top_k=3, max_rrf_fallback_ratio=0.1):
        self.directory = directory
        self.top_k = top_k
        self.max_rrf_fallback_ratio = max_rrf_fallback_ratio
        self.relevant_docs = defaultdict(list)
        self.all_queries = set()
        self.doc_relevance_count = defaultdict(lambda: defaultdict(int))
        self.rrf_results = {}
        self.rrf_fallback_count = 0
        self._load_relevant_documents()
        self._load_rrf_results()

    def _load_relevant_documents(self):
        for filename in os.listdir(self.directory):
            if filename.endswith('ares_scores.tsv') and filename != "RRF_ares_scores.tsv":
                file_path = os.path.join(self.directory, filename)
                df = pd.read_csv(file_path, sep='\t')
                
                if 'ARES_Context_Relevance_Prediction' in df.columns and 'Query' in df.columns and 'Document' in df.columns:
                    self.all_queries.update(df['Query'].unique())
                    relevant_docs = df[df['ARES_Context_Relevance_Prediction'] == 1][['Query', 'Document']]
                    for _, row in relevant_docs.iterrows():
                        self.relevant_docs[row['Query']].append(row['Document'])
                        self.doc_relevance_count[row['Query']][row['Document']] += 1

        print(f"Total queries: {len(self.all_queries)}")
        print(f"Queries with relevant documents: {len(self.relevant_docs)}")
        print(f"Total relevant documents: {sum(len(docs) for docs in self.relevant_docs.values())}")
        print(f"Queries without relevant documents: {len(self.all_queries) - len(self.relevant_docs)}")

    def _load_rrf_results(self):
        rrf_file = os.path.join(self.directory, "RRF_results.tsv")
        if os.path.exists(rrf_file):
            df = pd.read_csv(rrf_file, sep='\t')
            self.rrf_results = df.groupby('Query')['Document'].apply(list).to_dict()
        else:
            print("RRF_results.tsv not found. Fallback mechanism will not be available.")

    def retrieve(self, query):
        if query in self.relevant_docs:
            # Sort documents based on frequency (relevance count)
            sorted_docs = sorted(
                self.doc_relevance_count[query].items(),
                key=lambda x: (-x[1], x[0])  # Sort by count (descending) then by document (ascending)
            )
            
            # Select up to top_k documents
            top_docs = [doc for doc, _ in sorted_docs[:self.top_k]]
            
            return top_docs if top_docs else self._controlled_rrf_fallback(query)
        else:
            return self._controlled_rrf_fallback(query)

    def _controlled_rrf_fallback(self, query):
        """
        Controlled RRF fallback mechanism.
        
        This method ensures that RRF is used as a fallback for a maximum of 10% of queries
        for which AuRA was not able to get relevant documents.
        
        Returns:
        - List of documents if RRF fallback is allowed
        - ["No Relevant Information Available"] if RRF fallback limit is reached
        """
        max_allowed_fallbacks = int(len(self.all_queries) * self.max_rrf_fallback_ratio)
        
        if self.rrf_fallback_count < max_allowed_fallbacks:
            if query in self.rrf_results:
                self.rrf_fallback_count += 1
                return self.rrf_results[query][:self.top_k]
        
        return ["No Relevant Information Available"]

    def save_results(self, output_file):
        results = []
        queries_without_relevant_docs = []
        for query in self.all_queries:
            docs = self.retrieve(query)
            if docs == ["No Relevant Information Available"]:
                queries_without_relevant_docs.append(query)
                results.append({'Query': query, 'Document': "No Relevant Information Available"})
            else:
                for doc in docs:
                    results.append({'Query': query, 'Document': doc})
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, sep='\t', index=False)
        print(f"Results saved to {output_file}")
        print(f"Total queries processed: {len(self.all_queries)}")
        print(f"Total results saved: {len(results)}")
        print(f"Queries without relevant documents (after controlled RRF fallback): {len(queries_without_relevant_docs)}")
        print(f"RRF fallback used for {self.rrf_fallback_count} queries")
        
        # Print 5 random queries without relevant documents
        sample_size = min(5, len(queries_without_relevant_docs))
        random_queries = random.sample(queries_without_relevant_docs, sample_size)
        print("5 random queries without relevant documents (after controlled RRF fallback):")
        for query in random_queries:
            print(f"- {query}")
            
# class AuRA_Dynamic_Retriever(Retriever):
#     """
#     AuRA Dynamic Retriever that aggregates and ranks documents from multiple retrievers based on ARES evaluation scores.
#     """
#     def __init__(self, config, retrievers_results_files):
#         super().__init__(config)
#         self.retrievers_results_files = retrievers_results_files
#         self.top_k = int(config.get('top_k', 3))
#         self.rrf_retriever = ReciprocalRankFusionRetriever(config, [])  # Initialize RRF retriever for fallback
#         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     def create_index(self):
#         # This retriever doesn't require an index creation
#         pass

#     def retrieve(self, queries: List[str], all_retrievers_results: Dict[str, Dict[str, List[str]]]):
#         logging.info("Running AuRA Dynamic Retriever...")
#         aggregated_results = {}

#         all_queries = set(queries)  # Ensure we have a complete list of queries

#         for query in queries:
#             query_relevant_docs = []
#             logging.info(f"Processing query: {query}")

#             for retriever_name, results_file in self.retrievers_results_files.items():
#                 logging.info(f"Checking results from retriever: {retriever_name}")
#                 retriever_results = pd.read_csv(results_file, sep='\t')
#                 query_results = retriever_results[retriever_results['Query'] == query]

#                 # Filter relevant documents based on ARES evaluation scores
#                 if 'ARES_Context_Relevance_Prediction' in query_results.columns:
#                     relevant_docs = query_results[query_results['ARES_Context_Relevance_Prediction'] == 1]['Document'].tolist()
#                     logging.info(f"Found {len(relevant_docs)} relevant documents for query: {query}")
#                     query_relevant_docs.extend(relevant_docs)

#             # Aggregate and rank documents
#             ranked_docs = self.aggregate_and_rank_documents(query_relevant_docs, all_retrievers_results, query)
            
#             if not ranked_docs:
#                 # Fallback to RRF if no relevant documents found
#                 logging.info(f"No relevant documents found for query: {query}. Falling back to RRF.")
#                 rrf_results = self.rrf_retriever.search([query])[0]
#                 aggregated_results[query] = rrf_results
#             else:
#                 aggregated_results[query] = ranked_docs

#         # Ensure that all queries are included in the final results
#         for query in all_queries:
#             if query not in aggregated_results:
#                 logging.info(f"Query {query} missing from aggregated results. Falling back to RRF.")
#                 rrf_results = self.rrf_retriever.search([query])[0]
#                 aggregated_results[query] = rrf_results

#         logging.info(f"Total queries processed: {len(aggregated_results)}")
#         return aggregated_results

#     def aggregate_and_rank_documents(self, relevant_docs: List[str], all_retrievers_results: Dict[str, Dict[str, List[str]]], query: str):
#         if not relevant_docs:
#             logging.info(f"No relevant documents found for query: {query}")
#             return []

#         # Count the frequency of relevance
#         doc_frequency = Counter(relevant_docs)

#         # Create a list of documents with their frequency and initial rank
#         doc_ranks = []
#         for doc in doc_frequency.keys():
#             initial_ranks = [
#                 i + 1 for retriever in all_retrievers_results.keys() if query in all_retrievers_results[retriever]
#                 for i, retrieved_doc in enumerate(all_retrievers_results[retriever][query])
#                 if retrieved_doc == doc
#             ]
#             doc_ranks.append((doc, doc_frequency[doc], min(initial_ranks) if initial_ranks else float('inf')))

#         # Sort by frequency of relevance (primary) and initial rank (secondary)
#         doc_ranks.sort(key=lambda x: (-x[1], x[2]))

#         # Select up to top-k documents, but can be less if fewer relevant documents are found
#         top_k_docs = [doc for doc, _, _ in doc_ranks[:self.top_k]]
#         logging.info(f"Retrieved {len(top_k_docs)} documents for query {query}: {top_k_docs}")
#         return top_k_docs

#     def search(self, queries):
#         return self.retrieve(queries, {})  # Pass an empty dictionary as all_retrievers_results
    
# class AuRA_Dynamic_Retriever:
#     def __init__(self, config, result_files):
#         self.config = config
#         self.top_k = int(config.get('top_k', 3))
#         self.output_dir = config.get('LLM_prediction_folder_directory', './retriever_results')
#         self.ares_data = self.load_ares_data(result_files)
#         logging.info(f"AuRA Dynamic Retriever initialized with top_k={self.top_k}")

#     def load_ares_data(self, result_files):
#         ares_data = {}
#         total_unique_relevant_pairs = 0
#         for retriever_name, _ in result_files.items():
#             ares_file = os.path.join(self.output_dir, f"{retriever_name}_ares_scores.tsv")
#             if not os.path.exists(ares_file):
#                 logging.warning(f"ARES scores file not found for {retriever_name}: {ares_file}")
#                 continue

#             df = pd.read_csv(ares_file, sep='\t')
#             if 'ARES_Context_Relevance_Prediction' not in df.columns:
#                 logging.warning(f"ARES_Context_Relevance_Prediction column not found for {retriever_name}. Skipping.")
#                 continue

#             relevant_df = df[df['ARES_Context_Relevance_Prediction'] == 1]
#             ares_data[retriever_name] = relevant_df.groupby('Query')['Document'].apply(list).to_dict()
#             total_unique_relevant_pairs += len(relevant_df)
#             logging.info(f"Loaded {len(ares_data[retriever_name])} relevant queries for retriever: {retriever_name}")

#         logging.info(f"Total queries loaded: {sum(len(queries) for queries in ares_data.values())}")
#         logging.info(f"Total unique relevant query + document pairs found: {total_unique_relevant_pairs}")
#         return ares_data
    
# class AuRA_Dynamic_Retriever:
#     def __init__(self, config, result_files):
#         self.config = config
#         self.top_k = int(config.get('top_k', 3))
#         self.output_dir = config.get('LLM_prediction_folder_directory', './retriever_results')
#         self.ares_data = self.load_ares_data(result_files)
#         logging.info(f"AuRA Dynamic Retriever initialized with top_k={self.top_k}")

#     def load_ares_data(self, result_files):
#         ares_data = {}
#         total_unique_relevant_pairs = 0
#         for retriever_name, _ in result_files.items():
#             ares_file = os.path.join(self.output_dir, f"{retriever_name}_ares_scores.tsv")
#             if not os.path.exists(ares_file):
#                 logging.warning(f"ARES scores file not found for {retriever_name}: {ares_file}")
#                 continue

#             df = pd.read_csv(ares_file, sep='\t')
#             if 'ARES_Context_Relevance_Prediction' not in df.columns:
#                 logging.warning(f"ARES_Context_Relevance_Prediction column not found for {retriever_name}. Skipping.")
#                 continue

#             relevant_df = df[df['ARES_Context_Relevance_Prediction'] == 1]
#             ares_data[retriever_name] = relevant_df.groupby('Query')['Document'].apply(list).to_dict()
#             total_unique_relevant_pairs += len(relevant_df)
#             logging.info(f"Loaded {len(ares_data[retriever_name])} relevant queries for retriever: {retriever_name}")

#         logging.info(f"Total queries loaded: {sum(len(queries) for queries in ares_data.values())}")
#         logging.info(f"Total unique relevant query + document pairs found: {total_unique_relevant_pairs}")
        
#         breakpoint() 
        
#         return ares_data

#     def retrieve(self, queries, individual_retriever_results):
#         logging.info(f"Total queries to process: {len(queries)}")

#         aura_results = []
#         processed_queries = set()

#         all_queries_set = set(queries)
#         missing_queries = all_queries_set - processed_queries
#         logging.info(f"Missing queries before processing: {len(missing_queries)}")

#         for query in tqdm(queries, desc="AuRA Dynamic Retrieval"):
#             logging.info(f"Processing query: {query}")
#             relevant_docs = set()
#             doc_relevance_count = {}
            
#             for retriever_name, retriever_results in self.ares_data.items():
#                 if query not in retriever_results:
#                     continue
                
#                 ares_relevant_docs = set(retriever_results[query])
#                 logging.info(f"ARES relevant docs for {retriever_name}: {len(ares_relevant_docs)}")
                
#                 for doc in ares_relevant_docs:
#                     relevant_docs.add(doc)
#                     doc_relevance_count[doc] = doc_relevance_count.get(doc, 0) + 1
            
#             logging.info(f"Relevant documents found: {len(relevant_docs)}")
            
#             if not relevant_docs:
#                 aura_results.append({"Query": query, "Document": "No relevant documents found"})
#                 logging.info(f"No relevant documents found for query: {query}")
#             else:
#                 ranked_docs = sorted(relevant_docs, key=lambda x: -doc_relevance_count[x])
#                 top_k_docs = ranked_docs[:self.top_k]
#                 logging.info(f"Final top-k documents: {[doc[:50] + '...' for doc in top_k_docs]}")
                
#                 for doc in top_k_docs:
#                     aura_results.append({"Query": query, "Document": doc})
            
#             processed_queries.add(query)

#         logging.info(f"Total queries processed: {len(processed_queries)}")
#         logging.info(f"Total unique relevant query + document pairs found: {len(aura_results)}")
#         print(f"Length of results: {len(aura_results)}")

#         missing_queries = all_queries_set - processed_queries
#         logging.info(f"Missing queries after processing: {len(missing_queries)}")
#         assert len(missing_queries) == 0, f"Some queries were not processed. Missing: {missing_queries}"

#         return aura_results
