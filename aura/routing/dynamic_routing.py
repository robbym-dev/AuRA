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
        """
        Initializes the DynamicRouting class with the given configuration.

        Args:
            aura_config (dict): Configuration settings for the DynamicRouting.
        """
        self.aura_config = aura_config
        number_of_labels = len(aura_config["llm_names"])  # Define the number of labels for the classifier

        # Load the pre-trained classifier model
        self.classifier_model = CustomBERTModel.from_pretrained(aura_config["model_dir"], aura_config["model_choice"], number_of_labels)
        self.classifier_model.eval()  # Set the model to evaluation mode
        self.tokenizer = AutoTokenizer.from_pretrained(aura_config["model_choice"])  # Load the tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device to GPU if available, else CPU
        self.classifier_model.to(self.device)  # Move the model to the device

        # Initialize the LLMQuery with API keys
        self.llm_query = LLMQuery(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            together_ai_api_key=os.getenv("TOGETHER_AI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    def route_query(self, query, document):
        """
        Routes the query to the best LLM based on the classifier's prediction.

        Args:
            query (str): The query string.
            document (str): The document string.

        Returns:
            str: The name of the best LLM.
        """
        # Tokenize the query and document
        encoding = self.tokenizer(
            query, document,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        # Get the logits from the classifier model
        with torch.no_grad():
            logits = self.classifier_model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])

        # Determine the best LLM based on the logits
        best_llm_index = torch.argmax(logits).item()
        llm_names = ["GPT-4", "Llama", "Claude Opus"]
        best_llm = llm_names[best_llm_index]
        return best_llm

    def generate_answer(self, query, document, llm_name):
        """
        Generates an answer using the specified LLM.

        Args:
            query (str): The query string.
            document (str): The document string.
            llm_name (str): The name of the LLM to use.

        Returns:
            str: The generated answer.
        """
        prompt = f"Query: {query}\nDocument: {document}\nAnswer:"
        if llm_name == "GPT-4":
            return self.llm_query.query_gpt4(prompt, "You are an expert assistant specialized in answering questions based on provided documents. Given a query and a set of documents, provide the most relevant and concise answer based on the content of the documents.")
        elif llm_name == "Llama":
            return self.llm_query.query_llama(prompt, "You are an expert assistant specialized in answering questions based on provided documents. Given a query and a set of documents, provide the most relevant and concise answer based on the content of the documents.")
        elif llm_name == "Claude Opus":
            return self.llm_query.query_claude_opus(prompt, "You are an expert assistant specialized in answering questions based on provided documents. Given a query and a set of documents, provide the most relevant and concise answer based on the content of the documents.")
        return None

    def run(self):
        """
        Executes the dynamic routing process: routes queries, generates answers, and saves the results.

        Returns:
            str: The file path where the results are saved.
        """
        print("Running Dynamic Routing...")

        # Load documents and queries from the specified file
        df = pd.read_csv(os.path.join(self.aura_config["LLM_prediction_folder_directory"], "best_retriever_docs.tsv"), sep='\t')
        documents = df['Document'].tolist()
        queries = df['Query'].tolist()

        results = []
        # Process each query-document pair
        for i, (query, document) in enumerate(tqdm(zip(queries, documents), total=len(queries), desc="Dynamic Routing")):
            try:
                logging.info(f"Processing query {i+1}/{len(queries)}: {query}")
                best_llm = self.route_query(query, document)  # Route the query to the best LLM
                answer = self.generate_answer(query, document, best_llm)  # Generate the answer using the best LLM
                results.append({"Query": query, "Document": document, "Answer": answer, "Best_LLM": best_llm})
            except Exception as e:
                logging.error(f"Error processing query {i+1}: {e}")
                continue

        # Save the results to a file
        results_df = pd.DataFrame(results)
        results_file = os.path.join(self.aura_config["LLM_prediction_folder_directory"], "dynamic_routing_results.tsv")
        results_df.to_csv(results_file, index=False, sep='\t')
        print(f"Dynamic routing results saved to: {results_file}")
        return results_file