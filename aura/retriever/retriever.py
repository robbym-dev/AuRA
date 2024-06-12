# aura/retriever/retriever.py

import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

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
