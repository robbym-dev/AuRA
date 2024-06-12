import os
import pandas as pd
from aura.classifier.llm_classifier import CustomBERTModel
from aura.generator.llm_query import LLMQuery
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import logging

class DynamicRouting:
    def __init__(self, aura_config):
        self.aura_config = aura_config
        number_of_labels = len(["GPT-4", "Llama", "Claude Opus"])  

        self.classifier_model = CustomBERTModel.from_pretrained(aura_config["model_dir"], aura_config["model_choice"], number_of_labels)
        self.classifier_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(aura_config["model_choice"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_model.to(self.device)
        self.llm_query = LLMQuery(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            together_ai_api_key=os.getenv("TOGETHER_AI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        logging.basicConfig(filename='dynamic_routing.log', level=logging.INFO)

    def route_query(self, query, document):
        encoding = self.tokenizer(
            query, document,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            logits = self.classifier_model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
        best_llm_index = torch.argmax(logits).item()
        llm_names = ["GPT-4", "Llama", "Claude Opus"]
        best_llm = llm_names[best_llm_index]
        return best_llm

    def generate_answer(self, query, document, llm_name):
        prompt = f"Query: {query}\nDocument: {document}\nAnswer:"
        if llm_name == "GPT-4":
            return self.llm_query.query_gpt4(prompt, "You are an expert assistant specialized in answering questions based on provided documents. Given a query and a set of documents, provide the most relevant and concise answer based on the content of the documents.")
        elif llm_name == "Llama":
            return self.llm_query.query_llama(prompt, "You are an expert assistant specialized in answering questions based on provided documents. Given a query and a set of documents, provide the most relevant and concise answer based on the content of the documents.")
        elif llm_name == "Claude Opus":
            return self.llm_query.query_claude_opus(prompt, "You are an expert assistant specialized in answering questions based on provided documents. Given a query and a set of documents, provide the most relevant and concise answer based on the content of the documents.")
        return None

    def run(self, num_queries=None):
        print("Running Dynamic Routing...")
        logging.info("Starting Dynamic Routing...")
        
        queries_df = pd.read_csv(self.aura_config["queries"], sep='\t')
        queries = queries_df['Query'].tolist()
        if num_queries:
            queries = queries[:num_queries]
        docs_df = pd.read_csv(os.path.join(self.aura_config["LLM_prediction_folder_directory"], "best_retriever_docs.tsv"), sep='\t')
        documents = docs_df['Document'].tolist()
        if num_queries:
            documents = documents[:num_queries]
        
        results = []
        for i, (query, document) in enumerate(tqdm(zip(queries, documents), total=len(queries), desc="Dynamic Routing")):
            try:
                logging.info(f"Processing query {i+1}/{len(queries)}: {query}")
                best_llm = self.route_query(query, document)
                answer = self.generate_answer(query, document, best_llm)
                results.append({"Query": query, "Document": document, "Answer": answer, "Best_LLM": best_llm})
            except Exception as e:
                logging.error(f"Error processing query {i+1}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        results_file = os.path.join(self.aura_config["LLM_prediction_folder_directory"], "dynamic_routing_results.tsv")
        results_df.to_csv(results_file, index=False, sep='\t')
        print(f"Dynamic routing results saved to: {results_file}")
        logging.info(f"Dynamic routing results saved to: {results_file}")
        return results_file
