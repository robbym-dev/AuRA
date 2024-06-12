import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from aura.retriever.retriever import BM25, SentenceBERT_Retriever, MiniLM_Retriever
from aura.generator.llm_query import LLMQuery
from aura.classifier.llm_classifier import CustomBERTModel, prepare_data_for_training, train_model, prepare_classifier_data_from_files
from aura.routing.baseline_routing import BaselineRouting
from aura.routing.dynamic_routing import DynamicRouting
from ares import ARES

def find_best_retriever(aura_config):
    """
    Finds the best retriever based on the given configuration.

    Args:
        aura_config (dict): Configuration dictionary containing necessary parameters.

    Returns:
        tuple: Best retriever name and the file path of the best retriever documents.
    """
    queries_df = pd.read_csv(aura_config["queries"], sep='\t')
    queries = queries_df['Query'].tolist()
    
    best_retriever_docs_file = os.path.join(aura_config["LLM_prediction_folder_directory"], "best_retriever_docs.tsv")
    
    if os.path.exists(best_retriever_docs_file):
        print(f"Best retriever docs file already exists: {best_retriever_docs_file}")
        best_retriever_name = "pre-evaluated"
        return best_retriever_name, best_retriever_docs_file
    
    max_queries = aura_config.get("max_queries", len(queries))
    queries = queries[:max_queries]

    retrievers = {
        "bm25": BM25(aura_config),
        "sentence_bert": SentenceBERT_Retriever(aura_config),
        "minilm": MiniLM_Retriever(aura_config)
    }

    best_retriever_name = None
    best_retriever_score = float('-inf')

    for name, retriever in retrievers.items():
        print(f"Evaluating {name} retriever...")
        retriever.initialize_retriever()
        documents = retriever.search(queries)

        results_data = []
        for query, doc_list in zip(queries, documents):
            for doc in doc_list:
                results_data.append({"Query": query, "Document": doc})

        retrieved_docs_file = f"retrieved_docs_{name}.csv"
        pd.DataFrame(results_data).to_csv(retrieved_docs_file, index=False, sep='\t')

        ppi_config = {
            "evaluation_datasets": [retrieved_docs_file],
            "few_shot_examples_filepath": aura_config["few_shot_examples_file_path"],
            "checkpoints": aura_config["checkpoints"][:1],
            "rag_type": aura_config["rag_type"],
            "labels": ["Context_Relevance_Label"],
            "gold_label_path": aura_config["gold_label_path"],
        }
        ares = ARES(ppi=ppi_config)
        results = ares.evaluate_RAG()
        print(f"Results for {name} retriever: {results}")

        if results:
            first_result = results[0]
            confidence_interval = first_result.get("ARES_Confidence_Interval", [0, 0])
            print(f"Confidence Interval for {name}: {confidence_interval}")
            score = confidence_interval[0]

            if score > best_retriever_score:
                best_retriever_name = name
                best_retriever_score = score

    print(f"Best retriever: {best_retriever_name} with score: {best_retriever_score}")

    best_retriever = retrievers[best_retriever_name]
    documents = best_retriever.search(queries)
    best_retriever_docs_file = os.path.join(aura_config["LLM_prediction_folder_directory"], "best_retriever_docs.tsv")
    results_data = []
    for query, doc_list in zip(queries, documents):
        for doc in doc_list:
            results_data.append({"Query": query, "Document": doc})
    pd.DataFrame(results_data).to_csv(best_retriever_docs_file, index=False, sep='\t')

    return best_retriever_name, best_retriever_docs_file

def evaluate_llms(aura_config, best_retriever_docs_file):
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

    max_queries = aura_config.get("max_queries", len(queries))
    queries = queries[:max_queries]
    documents = documents[:max_queries]

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
            "checkpoints": [aura_config["checkpoints"][1]],
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

def load_ares_predictions(aura_config, llm_names):
    """
    Loads ARES predictions for the given LLM names.

    Args:
        aura_config (dict): Configuration dictionary containing necessary parameters.
        llm_names (list): List of LLM names.

    Returns:
        dict: A dictionary with LLM names as keys and their ARES predictions as values.
    """
    ares_results = {}
    for llm_name in llm_names:
        prediction_file_path = os.path.join(aura_config["LLM_prediction_folder_directory"], f"{llm_name}_ares_predictions.tsv")
        if not os.path.exists(prediction_file_path):
            raise FileNotFoundError(f"Prediction file for {llm_name} not found: {prediction_file_path}")
        ares_results[llm_name] = pd.read_csv(prediction_file_path, sep='\t')["ARES_Answer_Relevance_Prediction"].tolist()
    return ares_results

def prepare_classifier_data_from_files(aura_config, llm_names):
    """
    Prepares classifier data from files.

    Args:
        aura_config (dict): Configuration dictionary containing necessary parameters.
        llm_names (list): List of LLM names.

    Returns:
        dict: A dictionary containing documents, queries, answers, and ARES results.
    """
    documents = []
    queries = []
    answers = {llm_name: [] for llm_name in llm_names}

    docs_df = pd.read_csv(os.path.join(aura_config["LLM_prediction_folder_directory"], "best_retriever_docs.tsv"), sep='\t')
    queries = docs_df['Query'].tolist()
    documents = docs_df['Document'].tolist()

    for llm_name in llm_names:
        results_file = os.path.join(aura_config["LLM_prediction_folder_directory"], f"{llm_name}_results.tsv")
        results_df = pd.read_csv(results_file, sep='\t')
        answers[llm_name] = results_df['Answer'].tolist()

    ares_results = load_ares_predictions(aura_config, llm_names)

    classifier_data = {
        "documents": documents,
        "queries": queries,
        "answers": answers,
        "ares_results": ares_results
    }

    return classifier_data

def print_final_parameters(params):
    """
    Prints the final parameters used for training the classifier.

    Args:
        params (dict): Dictionary of parameters.
    """
    print("Final parameters used for training the classifier:")
    for key, value in params.items():
        print(f"{key}: {value}")

def train_llm_classifier(aura_config):
    """
    Trains the LLM classifier based on the given configuration.

    Args:
        aura_config (dict): Configuration dictionary containing necessary parameters.
    """
    llm_names = ["GPT-4", "Llama", "Claude Opus"]

    classifier_data = prepare_classifier_data_from_files(aura_config, llm_names)
    documents = classifier_data["documents"]
    queries = classifier_data["queries"]
    answers = classifier_data["answers"]
    ares_results = classifier_data["ares_results"]

    rows = []
    for model, model_answers in answers.items():
        for i, (query, doc, answer) in enumerate(zip(queries, documents, model_answers)):
            relevance_score = ares_results[model][i]
            rows.append({"Query": query, "Document": doc, "Answer": answer, "ARES_Answer_Relevance_Prediction": relevance_score, "Model": model})

    df = pd.DataFrame(rows)
    df.to_csv(aura_config["classifier_training_data"], index=False, sep='\t')

    model_name = aura_config.get("model_choice", "bert-base-uncased")
    num_labels = len(answers.keys())
    learning_rate = aura_config.get("learning_rate", 2e-5)
    num_epochs = aura_config.get("num_epochs", 3)
    batch_size = aura_config.get("assigned_batch_size", 16)
    validation_path = aura_config.get("classifier_validation_data", "")
    max_length = 512
    device = aura_config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    final_params = {
        "model_name": model_name,
        "num_labels": num_labels,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "validation_path": validation_path,
        "device": device
    }
    print_final_parameters(final_params)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = prepare_data_for_training(aura_config["classifier_training_data"], tokenizer, max_length, label_column="ARES_Answer_Relevance_Prediction")
    val_dataset = prepare_data_for_training(aura_config["classifier_validation_data"], tokenizer, max_length, label_column="Answer_Relevance_Label")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CustomBERTModel(num_labels, model_name)
    trained_model = train_model(model, train_dataloader, val_dataloader, num_epochs, learning_rate, device)

    checkpoint_path = os.path.join(aura_config["model_dir"], "model_checkpoint.pt")
    torch.save(trained_model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

def construct_results_filename(base_name, num_queries):
    """
    Constructs a results filename based on the base name and number of queries.

    Args:
        base_name (str): Base name for the file.
        num_queries (int): Number of queries.

    Returns:
        str: Constructed filename.
    """
    return f"{base_name}_{num_queries}.tsv"

def run_baseline_routing(aura_config, num_queries=None):
    """
    Runs baseline routing and saves the results.

    Args:
        aura_config (dict): Configuration dictionary containing necessary parameters.
        num_queries (int, optional): Number of queries. Defaults to None.

    Returns:
        str: File path of the baseline routing results.
    """
    results_filename = construct_results_filename("baseline_routing_results", num_queries)
    results_filepath = os.path.join(aura_config["LLM_prediction_folder_directory"], results_filename)
    
    if os.path.exists(results_filepath):
        print(f"Baseline routing results for {num_queries} queries already exist. Skipping baseline routing.")
        return results_filepath

    baseline_router = BaselineRouting(aura_config)
    results_file = baseline_router.run(num_queries)
    os.rename(results_file, results_filepath)
    print(f"Baseline routing results saved to: {results_filepath}")
    return results_filepath

def run_dynamic_routing(aura_config, num_queries=None):
    """
    Runs dynamic routing and saves the results.

    Args:
        aura_config (dict): Configuration dictionary containing necessary parameters.
        num_queries (int, optional): Number of queries. Defaults to None.

    Returns:
        str: File path of the dynamic routing results.
    """
    results_filename = construct_results_filename("dynamic_routing_results", num_queries)
    results_filepath = os.path.join(aura_config["LLM_prediction_folder_directory"], results_filename)
    
    if os.path.exists(results_filepath):
        print(f"Dynamic routing results for {num_queries} queries already exist. Skipping dynamic routing.")
        return results_filepath

    dynamic_router = DynamicRouting(aura_config)
    results_file = dynamic_router.run(num_queries)
    os.rename(results_file, results_filepath)
    print(f"Dynamic routing results saved to: {results_filepath}")
    return results_filepath

def evaluate_routing_results(aura_config, results_file, evaluation_label):
    """
    Evaluates routing results using ARES.

    Args:
        aura_config (dict): Configuration dictionary containing necessary parameters.
        results_file (str): File path of the routing results.
        evaluation_label (str): Label for evaluation.

    Returns:
        dict: Evaluation results.
    """
    ppi_config = {
        "evaluation_datasets": [results_file],
        "few_shot_examples_filepath": aura_config["few_shot_examples_file_path"],
        "checkpoints": [aura_config["checkpoints"][1] if evaluation_label == "Answer_Relevance_Label" else aura_config["checkpoints"][0]],
        "rag_type": aura_config["rag_type"],
        "labels": [evaluation_label],
        "gold_label_path": aura_config["gold_label_path"],
        "debug_mode": True,
    }
    ares = ARES(ppi=ppi_config)
    routing_evaluation_results = ares.evaluate_RAG()
    print(f"{evaluation_label} evaluation results: {routing_evaluation_results}")
    return routing_evaluation_results

def run_aura_pipeline(aura_config, num_queries=None):
    """
    Runs the entire AuRA pipeline.

    Args:
        aura_config (dict): Configuration dictionary containing necessary parameters.
        num_queries (int, optional): Number of queries. Defaults to None.
    """
    print("Starting the AuRA pipeline...")
    best_retriever_name, best_retriever_docs_file = find_best_retriever(aura_config)
    print(f"Best retriever: {best_retriever_name}, documents saved to: {best_retriever_docs_file}")
    
    best_llm_name = evaluate_llms(aura_config, best_retriever_docs_file)
    print(f"Best LLM: {best_llm_name}")
    
    print("Training the classifier...")
    train_llm_classifier(aura_config)
    print("Classifier trained.")
    
    # Evaluate baseline and dynamic routing
    print("Running baseline routing...")
    baseline_results_file = run_baseline_routing(aura_config, num_queries)
    print(f"Baseline routing results saved to: {baseline_results_file}")
    
    print("Running dynamic routing...")
    dynamic_results_file = run_dynamic_routing(aura_config, num_queries)
    print(f"Dynamic routing results saved to: {dynamic_results_file}")
    
    print("Evaluating Baseline Routing for Context Relevance:")
    evaluate_routing_results(aura_config, baseline_results_file, "Context_Relevance_Label")
    
    print("Evaluating Baseline Routing for Answer Relevance:")
    evaluate_routing_results(aura_config, baseline_results_file, "Answer_Relevance_Label")
    
    print("Evaluating Dynamic Routing for Context Relevance:")
    evaluate_routing_results(aura_config, dynamic_results_file, "Context_Relevance_Label")
    
    print("Evaluating Dynamic Routing for Answer Relevance:")
    evaluate_routing_results(aura_config, dynamic_results_file, "Answer_Relevance_Label")

# Function to only run routing and evaluation
def run_routing_and_evaluation(aura_config, num_queries=None):
    # Evaluate baseline and dynamic routing
    print("Running baseline routing...")
    baseline_results_file = run_baseline_routing(aura_config, num_queries)
    print(f"Baseline routing results saved to: {baseline_results_file}")
    
    print("Running dynamic routing...")
    dynamic_results_file = run_dynamic_routing(aura_config, num_queries)
    print(f"Dynamic routing results saved to: {dynamic_results_file}")
    
    print("Evaluating Baseline Routing for Context Relevance:")
    baseline_context_results = evaluate_routing_results(aura_config, baseline_results_file, "Context_Relevance_Label")
    
    print("Evaluating Baseline Routing for Answer Relevance:")
    baseline_answer_results = evaluate_routing_results(aura_config, baseline_results_file, "Answer_Relevance_Label")
    
    print("Evaluating Dynamic Routing for Context Relevance:")
    dynamic_context_results = evaluate_routing_results(aura_config, dynamic_results_file, "Context_Relevance_Label")
    
    print("Evaluating Dynamic Routing for Answer Relevance:")
    dynamic_answer_results = evaluate_routing_results(aura_config, dynamic_results_file, "Answer_Relevance_Label")
    
    return {
        "baseline_context_results": baseline_context_results,
        "baseline_answer_results": baseline_answer_results,
        "dynamic_context_results": dynamic_context_results,
        "dynamic_answer_results": dynamic_answer_results
    }
    
def run_classifier_training_pipeline(aura_config):
    train_llm_classifier(aura_config)
