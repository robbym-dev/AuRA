# LLM_routing_manager.py

from aura.routing.baseline_routing import BaselineRouting
from aura.routing.dynamic_routing import DynamicRouting
from aura.generator.llm_query import LLMQuery
from ares import ARES
import pandas as pd
import tqdm
import os

#### Helper Functions #####

def construct_results_filename(base_name):
    """
    Constructs a results filename based on the base name.

    Args:
        base_name (str): Base name for the file.

    Returns:
        str: Constructed filename.
    """
    return f"{base_name}.tsv"

def run_baseline_routing(config):
    """
    Runs baseline routing and saves the results.

    Args:
        config (dict): Configuration dictionary containing necessary parameters.

    Returns:
        str: File path of the baseline routing results.
    """
    results_filename = construct_results_filename("baseline_routing_results")
    results_filepath = os.path.join(config["LLM_prediction_folder_directory"], results_filename)
    
    if os.path.exists(results_filepath):
        print(f"Baseline routing results already exist. Skipping baseline routing.")
        return results_filepath
    
    baseline_router = BaselineRouting(config)
    results_file = baseline_router.run()
    os.rename(results_file, results_filepath)
    print(f"Baseline routing results saved to: {results_filepath}")
    return results_file

def run_dynamic_routing(config):
    """
    Runs dynamic routing and saves the results.

    Args:
        config (dict): Configuration dictionary containing necessary parameters.

    Returns:
        str: File path of the dynamic routing results.
    """
    results_filename = construct_results_filename("dynamic_routing_results")
    results_filepath = os.path.join(config["LLM_prediction_folder_directory"], results_filename)
    
    if os.path.exists(results_filepath):
        print(f"Dynamic routing results already exist. Skipping dynamic routing.")
        return results_filepath
    
    dynamic_router = DynamicRouting(config)
    results_file = dynamic_router.run()
    os.rename(results_file, results_filepath)
    print(f"Dynamic routing results saved to: {results_filepath}")
    return results_file

def evaluate_routing_results(config, results_file, evaluation_label):
    """
    Evaluates routing results using ARES.

    Args:
        config (dict): Configuration dictionary containing necessary parameters.
        results_file (str): File path of the routing results.
        evaluation_label (str): Label for evaluation.

    Returns:
        dict: Evaluation results.
    """
    ppi_config = {
        "evaluation_datasets": [results_file],
        "few_shot_examples_filepath": config["few_shot_examples_file_path"],
        "checkpoints": [config["checkpoints"][1] if evaluation_label == "Answer_Relevance_Label" else config["checkpoints"][0]],
        "rag_type": config["rag_type"],
        "labels": [evaluation_label],
        "gold_label_path": config["gold_label_path"],
        "debug_mode": True,
    }
    ares = ARES(ppi=ppi_config)
    routing_evaluation_results = ares.evaluate_RAG()
    print(f"{evaluation_label} evaluation results: {routing_evaluation_results}")
    return routing_evaluation_results

def query_and_evaluate_llms(aura_config, best_retriever_docs_file):
    """
    Evaluates different LLMs based on the best retriever documents.

    Args:
        aura_config (dict): Configuration dictionary containing necessary parameters.
        best_retriever_docs_file (str): File path of the best retriever documents.

    Returns:
        str: Name of the best LLM.
    """
    SYSTEM_PROMPT = "You are an expert assistant specialized in answering questions based on provided documents. Given a query and a set of documents, provide the most relevant and concise answer based on the content of the documents."

    docs_df = pd.read_csv(best_retriever_docs_file, sep='\t')
    queries = docs_df['Query'].tolist()
    documents = docs_df['Document'].tolist()

    llm_query = LLMQuery(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        together_ai_api_key=os.getenv("TOGETHER_AI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    llm_names = ["GPT-4", "Llama", "Claude Opus"]
    llm_methods = {
        "GPT-4": llm_query.query_gpt4,
        "Llama": llm_query.query_llama,
        "Claude Opus": llm_query.query_claude_opus
    }

    debug_mode = aura_config.get("debug_mode", False)

    for llm_name in llm_names:
        llm_results_file = os.path.join(aura_config["LLM_prediction_folder_directory"], f"{llm_name}_results.tsv")
        prediction_file_path = os.path.join(aura_config["LLM_prediction_folder_directory"], f"{llm_name}_ares_predictions.tsv")

        if os.path.exists(llm_results_file):
            llm_results_df = pd.read_csv(llm_results_file, sep='\t')
            if 'Answer' in llm_results_df.columns:
                print(f"{llm_name} results file already exists, skipping querying.")
            else:
                raise ValueError(f"{llm_name} results file exists but does not contain 'Answer' column.")
        else:
            print(f"Querying {llm_name}...")
            llm_results = []
            for query, doc in tqdm(zip(queries, documents), total=len(queries), desc=f"Querying {llm_name}"):
                prompt = f"Query: {query}\nDocument: {doc}\nAnswer:"
                answer = llm_methods[llm_name](prompt, SYSTEM_PROMPT)
                if debug_mode:
                    print(f"Answer from {llm_name}: {answer}")
                llm_results.append({
                    "Query": query,
                    "Document": doc,
                    "Answer": answer
                })

            print(f"Saving results for {llm_name}...")
            pd.DataFrame(llm_results).to_csv(llm_results_file, index=False, sep='\t')
            llm_results_df = pd.read_csv(llm_results_file, sep='\t')

        if os.path.exists(prediction_file_path):
            llm_predictions_df = pd.read_csv(prediction_file_path, sep='\t')
            if 'ARES_Answer_Relevance_Prediction' in llm_predictions_df.columns:
                print(f"{llm_name} predictions file already exists and is correct, skipping evaluation.")
                continue
            else:
                print(f"{llm_name} predictions file exists but 'ARES_Answer_Relevance_Prediction' column is missing, re-evaluating.")
        else:
            print(f"Prediction file for {llm_name} does not exist, proceeding with evaluation.")

        print(f"Evaluating results for {llm_name} with ARES...")
        ppi_config = {
            "evaluation_datasets": [llm_results_file],
            "few_shot_examples_filepath": aura_config["few_shot_examples_file_path"],
            "checkpoints": [aura_config["checkpoint"]],
            "rag_type": aura_config["rag_type"],
            "labels": ["Answer_Relevance_Label"],
            "gold_label_path": aura_config["gold_label_path"],
            "debug_mode": True,
            "prediction_filepath": prediction_file_path
        }
        ares = ARES(ppi=ppi_config)
        llm_evaluation_results = ares.evaluate_RAG()
        print(f"LLM evaluation results for {llm_name}: {llm_evaluation_results}")

        if not llm_evaluation_results:
            raise ValueError(f"ARES evaluation for {llm_name} returned no results.")

    best_llm_name = None
    best_llm_score = float('-inf')
    for llm_name in llm_names:
        prediction_file_path = os.path.join(aura_config["LLM_prediction_folder_directory"], f"{llm_name}_ares_predictions.tsv")
        llm_predictions_df = pd.read_csv(prediction_file_path, sep='\t')
        if 'ARES_Answer_Relevance_Prediction' not in llm_predictions_df.columns:
            raise KeyError(f"'ARES_Answer_Relevance_Prediction' column missing in {llm_name} predictions file.")
        print(f"Scores for {llm_name}: {llm_predictions_df['ARES_Answer_Relevance_Prediction'].tolist()[:10]}")
        score = llm_predictions_df["ARES_Answer_Relevance_Prediction"].mean()
        if score > best_llm_score:
            best_llm_name = llm_name
            best_llm_score = score

    print(f"Best LLM: {best_llm_name} with score: {best_llm_score}")

    return best_llm_name

def run_llm_routing_pipeline(config):
    """
    Runs the complete LLM routing pipeline including baseline, dynamic routing, and evaluation.

    Args:
        config (dict): Configuration dictionary containing necessary parameters.

    Returns:
        dict: A dictionary containing the results of baseline and dynamic routing evaluations.
    """
    print("Running LLM Routing Pipeline...")
    
    # Run baseline routing
    print("Running Baseline Routing...")
    baseline_results_file = run_baseline_routing(config)
    
    # Run dynamic routing
    print("Running Dynamic Routing...")
    dynamic_results_file = run_dynamic_routing(config)
    
    # Evaluate results
    print("Evaluating Routing Results...")
    baseline_context_results = evaluate_routing_results(config, baseline_results_file, "Context_Relevance_Label")
    baseline_answer_results = evaluate_routing_results(config, baseline_results_file, "Answer_Relevance_Label")
    dynamic_context_results = evaluate_routing_results(config, dynamic_results_file, "Context_Relevance_Label")
    dynamic_answer_results = evaluate_routing_results(config, dynamic_results_file, "Answer_Relevance_Label")
    
    return {
        "baseline_context_results": baseline_context_results,
        "baseline_answer_results": baseline_answer_results,
        "dynamic_context_results": dynamic_context_results,
        "dynamic_answer_results": dynamic_answer_results
    }