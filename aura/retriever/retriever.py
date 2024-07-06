# aura/retriever/retriever.py

import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

class Retriever:
    """
    Base class for document retrievers.
    """
    def __init__(self, config):
        """
        Initializes the Retriever with the given configuration.

        Args:
            config (dict): Configuration settings for the retriever.
        """
        self.retrieval_model = config.get("retrieval_model")
        self.documents = config.get("documents")
        self.top_k = config.get("top_k", 5)
        self.document_index = None

    def initialize_retriever(self):
        """
        Initializes the retriever. Must be implemented in subclasses.
        """
        raise NotImplementedError("initialize_retriever method must be implemented in subclasses")

    def search(self, queries):
        """
        Searches for the given queries. Must be implemented in subclasses.

        Args:
            queries (list): List of query strings.

        Returns:
            list: List of retrieved documents for each query.
        """
        raise NotImplementedError("search method must be implemented in subclasses")

class BM25(Retriever):
    """
    BM25-based document retriever.
    """
    def __init__(self, config):
        """
        Initializes the BM25 retriever with the given configuration.

        Args:
            config (dict): Configuration settings for the retriever.
        """
        super().__init__(config)

    def initialize_retriever(self):
        """
        Initializes the BM25 retriever by loading and indexing documents.
        """
        print("Initializing BM25 retriever...")
        self.documents = pd.read_csv(self.documents, sep='\t')
        self.documents = self.documents.drop_duplicates(subset="Document")
        print(f"Loaded {len(self.documents)} documents for BM25 retriever.")
        self.create_index()
        print("BM25 index created.")

    def create_index(self):
        """
        Creates the BM25 index from the loaded documents.
        """
        self.documents = self.documents['Document'].tolist()
        tokenized_documents = [doc.split() for doc in self.documents]
        self.document_index = BM25Okapi(tokenized_documents)

    def search(self, queries):
        """
        Searches for the given queries using the BM25 index.

        Args:
            queries (list): List of query strings.

        Returns:
            list: List of top-k retrieved documents for each query.
        """
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
        """
        Initializes the SentenceBERT retriever with the given configuration.

        Args:
            config (dict): Configuration settings for the retriever.
        """
        super().__init__(config)

    def initialize_retriever(self):
        """
        Initializes the SentenceBERT retriever by loading and indexing documents.
        """
        print("Initializing SentenceBERT retriever...")
        self.documents = pd.read_csv(self.documents, sep='\t')
        self.documents = self.documents.drop_duplicates(subset="Document")
        print(f"Loaded {len(self.documents)} documents for SentenceBERT retriever.")
        self.retrieval_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.create_index()
        print("SentenceBERT index created.")

    def create_index(self):
        """
        Creates the SentenceBERT index from the loaded documents.
        """
        self.document_index = self.documents.drop_duplicates(subset="Document")
        tqdm.pandas(desc="Generating SentenceBERT embeddings")
        self.document_index['embeddings'] = self.document_index["Document"].progress_apply(lambda x: self.retrieval_model.encode(x, convert_to_tensor=True).tolist())
        self.document_index = Dataset.from_pandas(self.document_index)
        self.document_index.add_faiss_index(column="embeddings")

    def search(self, queries):
        """
        Searches for the given queries using the SentenceBERT index.

        Args:
            queries (list): List of query strings.

        Returns:
            list: List of top-k retrieved documents for each query.
        """
        total_retrieved_docs = []
        for query in tqdm(queries, desc="SentenceBERT retrieval"):
            query_embedding = self.retrieval_model.encode(query, convert_to_tensor=True).tolist()
            query_embedding = np.array(query_embedding).astype(np.float32)
            scores, samples = self.document_index.get_nearest_examples("embeddings", query_embedding, k=self.top_k)
            current_docs = [samples["Document"][j] for j in range(len(samples["Document"]))]
            total_retrieved_docs.append(current_docs)
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

    def initialize_retriever(self):
        """
        Initializes the MiniLM retriever by loading and indexing documents.
        """
        print("Initializing MiniLM retriever...")
        self.documents = pd.read_csv(self.documents, sep='\t')
        self.documents = self.documents.drop_duplicates(subset="Document")
        print(f"Loaded {len(self.documents)} documents for MiniLM retriever.")
        self.retrieval_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.create_index()
        print("MiniLM index created.")

    def create_index(self):
        """
        Creates the MiniLM index from the loaded documents.
        """
        self.document_index = self.documents.drop_duplicates(subset="Document")
        tqdm.pandas(desc="Generating MiniLM embeddings")
        self.document_index['embeddings'] = self.document_index["Document"].progress_apply(lambda x: self.retrieval_model.encode(x, convert_to_tensor=True).tolist())
        self.document_index = Dataset.from_pandas(self.document_index)
        self.document_index.add_faiss_index(column="embeddings")

    def search(self, queries):
        """
        Searches for the given queries using the MiniLM index.

        Args:
            queries (list): List of query strings.

        Returns:
            list: List of top-k retrieved documents for each query.
        """
        total_retrieved_docs = []
        for query in tqdm(queries, desc="MiniLM retrieval"):
            query_embedding = self.retrieval_model.encode(query, convert_to_tensor=True).tolist()
            query_embedding = np.array(query_embedding).astype(np.float32)
            scores, samples = self.document_index.get_nearest_examples("embeddings", query_embedding, k=self.top_k)
            current_docs = [samples["Document"][j] for j in range(len(samples["Document"]))]
            total_retrieved_docs.append(current_docs)
        return total_retrieved_docs

from jina import Flow, Document
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
        self.top_k = config.get('top_k', 5)
        self.documents_path = config.get('documents')
        self.elastic_host = config.get('elastic_host', 'localhost')
        self.elastic_port = config.get('elastic_port', 9200)
        self.index_name = config.get('index_name', 'documents')
        self.elastic_user = config.get('elastic_user', 'elastic')
        self.elastic_password = config.get('elastic_password', 'Va+d9c+9Fgmg1DHnxDw3')
        self.elastic_scheme = config.get('elastic_scheme', 'https')
        self.elastic_ca_cert = config.get('elastic_ca_cert', '/future/u/manihani/elasticsearch-8.14.0/config/certs/http_ca.crt')  # Adjust this path

        # Ensure Jina authentication token is set
        os.environ['JINA_AUTH_TOKEN'] = 'jina_1972dc7c93a247f4ad8748c1db7672b1jI9nHbiks0ypDvgrJ16XhJUvywOE'

        self.initialize_retriever()
        self.initialize_elasticsearch()
        self.flow = Flow().add(uses='jinaai://jina-ai/TransformerTorchEncoder')

    def initialize_retriever(self):
        """
        Initializes the BM25 retriever by loading and indexing documents.
        """
        print("Initializing BM25 retriever...")
        self.documents_df = pd.read_csv(self.documents_path, sep='\t')
        self.documents_df = self.documents_df.drop_duplicates(subset="Document")
        print(f"Loaded {len(self.documents_df)} documents for BM25 retriever.")
        self.create_index()
        print("BM25 index created.")

    def create_index(self):
        """
        Creates the BM25 index from the loaded documents.
        """
        self.documents = self.documents_df['Document'].tolist()
        tokenized_documents = [doc.split() for doc in self.documents]
        self.document_index = BM25Okapi(tokenized_documents)

    def initialize_elasticsearch(self):
        """
        Initializes the Elastic Search index and indexes the documents.
        """
        print("Initializing Elasticsearch...")
        self.es = Elasticsearch(
            [f'{self.elastic_scheme}://{self.elastic_host}:{self.elastic_port}'],
            http_auth=(self.elastic_user, self.elastic_password),
            ca_certs=self.elastic_ca_cert
        )
        
        # Define the index settings and mappings
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

        # Create the index if it doesn't exist
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=settings)
        
        # Index the documents
        actions = [
            {
                "_index": self.index_name,
                "_source": {"content": doc}
            }
            for doc in self.documents
        ]
        helpers.bulk(self.es, actions)
        print("Elasticsearch index created and documents indexed.")

    def search_bm25(self, queries):
        """
        Searches for the given queries using the BM25 index.

        Args:
            queries (list): List of query strings.

        Returns:
            list: List of top-k retrieved documents for each query.
        """
        print("Performing BM25 retrieval...")
        total_retrieved_docs = []
        for query in tqdm(queries, desc="BM25 retrieval"):
            if pd.isna(query) or not query.strip():
                print(f"Skipping empty or NaN query: {query}")
                total_retrieved_docs.append([])
                continue
            tokenized_query = query.split()
            scores = self.document_index.get_scores(tokenized_query)
            sorted_documents = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)
            top_k_docs = [doc_score_tuple[0] for doc_score_tuple in sorted_documents][:self.top_k]
            total_retrieved_docs.append(top_k_docs)
        print("BM25 retrieval completed.")
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
        print(f"Querying Elasticsearch for: {query}")
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
        print(f"Retrieved {len(documents)} documents from Elasticsearch for query: {query}")
        return documents

    def get_embeddings(self, docs):
        """
        Generates embeddings for the given documents using Jina.

        Args:
            docs (list): List of document texts.

        Returns:
            list: List of embeddings and list of valid documents.
        """
        print("Generating embeddings for documents...")
        valid_embeddings = []
        valid_docs = []
        skipped_count = 0
        with self.flow:
            jina_docs = [Document(text=doc) for doc in docs]
            results = self.flow.post(on='/index', inputs=jina_docs, return_results=True)

        for doc, result in zip(docs, results):
            embedding = result.embedding
            if embedding is None:
                print(f"Skipping document due to None embedding: {doc}")
                logging.debug(f"None embedding for doc: {doc}. Embedding: {embedding}")
                skipped_count += 1
                continue
            if not isinstance(embedding, np.ndarray):
                print(f"Skipping document due to invalid embedding type: {doc}")
                logging.debug(f"Invalid embedding type for doc: {doc}. Embedding: {embedding}")
                skipped_count += 1
                continue
            if np.isnan(embedding).any():
                print(f"Skipping document due to NaN embedding: {doc}")
                logging.debug(f"NaN embedding for doc: {doc}. Embedding: {embedding}")
                skipped_count += 1
                continue
            valid_docs.append(doc)
            valid_embeddings.append(embedding)

        print(f"Total documents skipped due to NaN embeddings: {skipped_count}")

        # Ensure embeddings are 2D arrays
        valid_embeddings = np.array(valid_embeddings)
        return valid_embeddings, valid_docs

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
        print("Reranking documents based on embeddings...")
        query_embedding = np.array(query_embedding).reshape(1, -1)
        doc_embeddings = np.array(doc_embeddings)
        similarity_scores = cosine_similarity(query_embedding, doc_embeddings)[0]
        reranked_docs = sorted(
            zip(docs, similarity_scores),
            key=lambda x: x[1],
            reverse=True
        )
        reranked_texts = [doc for doc, score in reranked_docs]
        print("Reranking completed.")
        return reranked_texts

    def search(self, queries):
        """
        Executes the full search pipeline: BM25 retrieval, Elasticsearch retrieval, embedding, and reranking.

        Args:
            queries (list): List of query strings.

        Returns:
            list: List of reranked documents for each query.
        """
        # Step 1: Retrieve top-k documents using BM25
        print("Starting BM25 retrieval...")
        bm25_docs = self.search_bm25(queries)

        # Step 2: Retrieve documents using Elasticsearch
        print("Starting Elasticsearch retrieval...")
        elasticsearch_docs = []
        for query in queries:
            elasticsearch_docs.append(self.search_elasticsearch(query))
        
        # Combine BM25 and Elasticsearch results
        combined_docs = [bm25 + elasticsearch for bm25, elasticsearch in zip(bm25_docs, elasticsearch_docs)]
        
        # Step 3: Generate embeddings for the queries and documents
        print("Generating embeddings for queries...")
        query_embeddings, valid_queries = self.get_embeddings(queries)
        
        print("Generating embeddings for documents...")
        document_embeddings_list = []
        valid_combined_docs = []
        for docs in combined_docs:
            embeddings, valid_docs = self.get_embeddings(docs)
            document_embeddings_list.append(embeddings)
            valid_combined_docs.append(valid_docs)

        # Step 4: Rerank the documents based on embeddings
        print("Reranking documents...")
        reranked_results = []
        for query_embedding, doc_embeddings, docs in zip(query_embeddings, document_embeddings_list, valid_combined_docs):
            reranked_docs = self.rerank_documents(query_embedding, docs, doc_embeddings)
            reranked_results.append(reranked_docs)

        # Save reranked documents for each query
        results_data = []
        for query, docs in zip(valid_queries, reranked_results):
            for doc in docs:
                results_data.append({"Query": query, "Document": doc})
        
        retrieved_docs_file = "retrieved_docs_bm25_elastic_jina.csv"
        pd.DataFrame(results_data).to_csv(retrieved_docs_file, index=False, sep='\t')
        print(f"Saved retrieved documents to {retrieved_docs_file}")

        return reranked_results