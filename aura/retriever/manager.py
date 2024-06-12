from .retriever import BM25, SentenceBERT_Retriever, MiniLM_Retriever

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
