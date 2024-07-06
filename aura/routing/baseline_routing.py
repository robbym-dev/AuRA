# This implementation is hard coded for now and will later support dynamic baseline configuration.

from aura.retriever.retriever import BM25, SentenceBERT_Retriever
from aura.generator.llm_query import LLMQuery
import os
import pandas as pd
from tqdm import tqdm

class BaselineRouting:
    def __init__(self, aura_config):
        """
        Initializes the BaselineRouting class with the given configuration.

        Args:
            aura_config (dict): Configuration settings for the BaselineRouting.
        """
        self.aura_config = aura_config
        self.retriever = BM25(aura_config)
        self.llm_query = LLMQuery(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            together_ai_api_key=os.getenv("TOGETHER_AI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    
    def retrieve_documents(self):
        """
        Retrieves documents based on the queries specified in the configuration.

        Returns:
            tuple: A tuple containing the list of queries and the list of retrieved documents.
        """
        print("Retrieving documents for Baseline Routing...")
        queries_df = pd.read_csv(self.aura_config["queries"], sep='\t')
        queries = queries_df['Query'].tolist()
        self.retriever.initialize_retriever()
        documents = self.retriever.search(queries)
        print(f"Retrieved {len(documents)} documents for Baseline Routing.")
        return queries, documents

    def generate_answers(self, queries, documents):
        """
        Generates answers for the given queries and documents using the LLM.

        Args:
            queries (list): List of query strings.
            documents (list): List of lists containing retrieved documents for each query.

        Returns:
            list: A list of dictionaries containing the query, document, and generated answer.
        """
        print("Generating answers for Baseline Routing...")
        results = []
        for query, doc_list in tqdm(zip(queries, documents), total=len(queries), desc="Baseline Routing"):
            for doc in doc_list:
                prompt = f"Query: {query}\nDocument: {doc}\nAnswer:"
                answer = self.llm_query.query_gpt4(prompt, "You are an expert assistant specialized in answering questions based on provided documents. Given a query and a set of documents, provide the most relevant and concise answer based on the content of the documents.")
                results.append({"Query": query, "Document": doc, "Answer": answer})
        print("Generated answers for Baseline Routing.")
        return results

    def run(self):
        """
        Executes the baseline routing process: retrieves documents, generates answers, and saves the results.

        Returns:
            str: The file path where the results are saved.
        """
        queries, documents = self.retrieve_documents()
        answers = self.generate_answers(queries, documents)
        results_df = pd.DataFrame(answers)
        results_file = os.path.join(self.aura_config["LLM_prediction_folder_directory"], "baseline_routing_results.tsv")
        results_df.to_csv(results_file, index=False, sep='\t')
        print(f"Baseline routing results saved to: {results_file}")
        return results_file
