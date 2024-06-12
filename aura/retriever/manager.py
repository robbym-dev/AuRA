from .retriever import BM25, SentenceBERT_Retriever, MiniLM_Retriever

class DocumentRetrievalManager:
    def __init__(self, config, retriever_name):
        self.config = config
        self.retriever_name = retriever_name
        self.retriever = self.initialize_retriever()

    def initialize_retriever(self):
        if self.retriever_name == 'bm25':
            retriever = BM25(self.config)
        elif self.retriever_name == 'sentencebert':
            retriever = SentenceBERT_Retriever(self.config)
        elif self.retriever_name == 'minilm':
            retriever = MiniLM_Retriever(self.config)
        else:
            raise ValueError(f"Retriever {self.retriever_name} not found")

        print(f"Initializing {self.retriever_name} retriever...")
        retriever.initialize_retriever()
        return retriever

    def search(self, query):
        return self.retriever.search([query])[0]
