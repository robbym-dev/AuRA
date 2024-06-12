from aura.retriever.retriever import BM25, SentenceBERT_Retriever
from aura.generator.llm_query import LLMQuery
import os
import pandas as pd
from tqdm import tqdm

class BaselineRouting:
    def __init__(self, aura_config):
        self.aura_config = aura_config
        self.retriever = BM25(aura_config)
        self.llm_query = LLMQuery(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            together_ai_api_key=os.getenv("TOGETHER_AI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    
    def retrieve_documents(self, num_queries=None):
        print("Retrieving documents for Baseline Routing...")
        queries_df = pd.read_csv(self.aura_config["queries"], sep='\t')
        queries = queries_df['Query'].tolist()
        if num_queries:
            queries = queries[:num_queries]
        self.retriever.initialize_retriever()
        documents = self.retriever.search(queries)
        print(f"Retrieved {len(documents)} documents for Baseline Routing.")
        return queries, documents

    def generate_answers(self, queries, documents):
        print("Generating answers for Baseline Routing...")
        results = []
        for query, doc_list in tqdm(zip(queries, documents), total=len(queries), desc="Baseline Routing"):
            for doc in doc_list:
                prompt = f"Query: {query}\nDocument: {doc}\nAnswer:"
                answer = self.llm_query.query_gpt4(prompt, "You are an expert assistant specialized in answering questions based on provided documents. Given a query and a set of documents, provide the most relevant and concise answer based on the content of the documents.")
                results.append({"Query": query, "Document": doc, "Answer": answer})
        print("Generated answers for Baseline Routing.")
        return results

    def run(self, num_queries=None):
        queries, documents = self.retrieve_documents(num_queries)
        answers = self.generate_answers(queries, documents)
        results_df = pd.DataFrame(answers)
        results_file = os.path.join(self.aura_config["LLM_prediction_folder_directory"], f"baseline_routing_results_{num_queries}.tsv")
        results_df.to_csv(results_file, index=False, sep='\t')
        print(f"Baseline routing results saved to: {results_file}")
        return results_file
