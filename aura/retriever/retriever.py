# aura/retriever/retriever.py

import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

class Retriever:
    def __init__(self, config):
        self.retrieval_model = config.get("retrieval_model")
        self.documents = config.get("documents")
        self.top_k = config.get("top_k", 5)
        self.document_index = None

    def initialize_retriever(self):
        raise NotImplementedError("initialize_retriever method must be implemented in subclasses")

    def search(self, queries):
        raise NotImplementedError("search method must be implemented in subclasses")

class BM25(Retriever):
    def __init__(self, config):
        super().__init__(config)

    def initialize_retriever(self):
        print("Initializing BM25 retriever...")
        self.documents = pd.read_csv(self.documents, sep='\t')
        self.documents = self.documents.drop_duplicates(subset="Document")
        print(f"Loaded {len(self.documents)} documents for BM25 retriever.")
        self.create_index()
        print("BM25 index created.")

    def create_index(self):
        self.documents = self.documents['Document'].tolist()
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
    def __init__(self, config):
        super().__init__(config)

    def initialize_retriever(self):
        print("Initializing SentenceBERT retriever...")
        self.documents = pd.read_csv(self.documents, sep='\t')
        self.documents = self.documents.drop_duplicates(subset="Document")
        print(f"Loaded {len(self.documents)} documents for SentenceBERT retriever.")
        self.retrieval_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.create_index()
        print("SentenceBERT index created.")

    def create_index(self):
        self.document_index = self.documents.drop_duplicates(subset="Document")
        tqdm.pandas(desc="Generating SentenceBERT embeddings")
        self.document_index['embeddings'] = self.document_index["Document"].progress_apply(lambda x: self.retrieval_model.encode(x, convert_to_tensor=True).tolist())
        self.document_index = Dataset.from_pandas(self.document_index)
        self.document_index.add_faiss_index(column="embeddings")

    def search(self, queries):
        total_retrieved_docs = []
        for query in tqdm(queries, desc="SentenceBERT retrieval"):
            query_embedding = self.retrieval_model.encode(query, convert_to_tensor=True).tolist()
            query_embedding = np.array(query_embedding).astype(np.float32)
            scores, samples = self.document_index.get_nearest_examples("embeddings", query_embedding, k=self.top_k)
            current_docs = [samples["Document"][j] for j in range(len(samples["Document"]))]
            total_retrieved_docs.append(current_docs)
        return total_retrieved_docs

class MiniLM_Retriever(Retriever):
    def __init__(self, config):
        super().__init__(config)

    def initialize_retriever(self):
        print("Initializing MiniLM retriever...")
        self.documents = pd.read_csv(self.documents, sep='\t')
        self.documents = self.documents.drop_duplicates(subset="Document")
        print(f"Loaded {len(self.documents)} documents for MiniLM retriever.")
        self.retrieval_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.create_index()
        print("MiniLM index created.")

    def create_index(self):
        self.document_index = self.documents.drop_duplicates(subset="Document")
        tqdm.pandas(desc="Generating MiniLM embeddings")
        self.document_index['embeddings'] = self.document_index["Document"].progress_apply(lambda x: self.retrieval_model.encode(x, convert_to_tensor=True).tolist())
        self.document_index = Dataset.from_pandas(self.document_index)
        self.document_index.add_faiss_index(column="embeddings")

    def search(self, queries):
        total_retrieved_docs = []
        for query in tqdm(queries, desc="MiniLM retrieval"):
            query_embedding = self.retrieval_model.encode(query, convert_to_tensor=True).tolist()
            query_embedding = np.array(query_embedding).astype(np.float32)
            scores, samples = self.document_index.get_nearest_examples("embeddings", query_embedding, k=self.top_k)
            current_docs = [samples["Document"][j] for j in range(len(samples["Document"]))]
            total_retrieved_docs.append(current_docs)
        return total_retrieved_docs
