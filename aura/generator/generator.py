import openai
from tqdm import tqdm
import os

class Generator:
    def __init__(self, config):
        self.llm_models = config.get("llm_models")
        self.documents = config.get("documents")
        self.queries = config.get("queries")
        self.top_k = config.get("top_k")

    def generate_answers(self):
        answers = {}
        for model in self.llm_models:
            answers[model] = []
            for query in tqdm(self.queries, desc=f"Generating answers with {model}"):
                answers[model].append(self.query_llm(model, query))
        return answers

    def query_llm(self, model, query):
        response = openai.Completion.create(
            engine=model,
            prompt=query,
            max_tokens=150
        )
        return response.choices[0].text.strip()

def prepare_data_for_classification(retriever, generator, aura_config):
    retriever.initialize_retriever()
    documents = retriever.search(aura_config["queries"])

    generator_config = {
        "llm_models": ["text-davinci-003", "text-curie-001"], 
        "documents": documents,
        "queries": aura_config["queries"],
        "top_k": aura_config["top_k"]
    }
    generator = Generator(generator_config)
    answers = generator.generate_answers()

    ares_results = {}
    for model, model_answers in answers.items():
        ppi_config = {
            "evaluation_datasets": model_answers,
            "few_shot_examples_filepath": aura_config["few_shot_examples_file_path"],
            "checkpoints": aura_config["checkpoints"],
            "rag_type": aura_config["rag_type"],
            "labels": ["Answer_Relevance_Label"],
            "gold_label_path": aura_config["gold_label_path"],
        }
        ares = ARES(ppi=ppi_config)
        ares_results[model] = ares.evaluate_RAG()
        
    classifier_data = {
        "documents": documents,
        "queries": aura_config["queries"],
        "answers": answers,
        "ares_results": ares_results
    }

    return classifier_data