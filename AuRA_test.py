# from aura.aura import find_best_retriever, evaluate_llms, run_classifier_training_pipeline, run_routing_and_evaluation

# aura_config = {
#     "documents": "/future/u/manihani/AuRA-Folder/AuRA/datasets/nq_ratio_0.6_filtered.tsv",
#     "queries": "/future/u/manihani/AuRA-Folder/AuRA/datasets/nq_ratio_0.6_filtered.tsv",
#     "few_shot_examples_file_path": "/future/u/manihani/AuRA-Folder/AuRA/datasets/nq_few_shot_prompt_for_judge_scoring.tsv",
#     "rag_type": "question_answering",
#     "checkpoints": ["/future/u/manihani/ARES/checkpoints/microsoft-deberta-v3-large/Context_Relevance_Label_joint_datasets_2024-04-30_01:01:01.pt", "/future/u/manihani/ARES/datasets/example_checkpoints/5e-06_1_True_Answer_Relevance_Label_ratio_0.6_reformatted_full_articles_False_validation_with_negatives_428380.pt"],
#     "gold_label_path": "/future/u/manihani/AuRA-Folder/AuRA/datasets/nq_labeled_output.tsv",
#     "LLM_prediction_folder_directory": "/future/u/manihani/AuRA-Folder/AuRA/llm_predictions",
#     "classifier_training_data": "/future/u/manihani/AuRA-Folder/AuRA/classifier_training/primary_classifier.tsv",
#     "model_dir": "/future/u/manihani/AuRA-Folder/AuRA/classifier_training",
#     "model_choice": "microsoft/deberta-v3-large",
#     "classifier_validation_data": "/future/u/manihani/ARES/datasets/eval_datasets/nq/nq_ratio_0.6.tsv",
#     "debug_mode": True,
#     "num_epochs": 10, 
#     "patience_value": 3, 
#     "learning_rate": 5e-6,
#     "assigned_batch_size": 1,  
#     "gradient_accumulation_multiplier": 32, 
#     "top_k": 1 
# }

# num_queries = 500


# routing_results = run_routing_and_evaluation(aura_config, num_queries)
# print(routing_results)

# # best_retriever_docs_file = "/future/u/manihani/AuRA-Folder/AuRA/llm_predictions/best_retriever_docs.tsv"

# # # Evaluate LLMs
# # evaluate_llms(aura_config, best_retriever_docs_file)

# run_classifier_training_pipeline(aura_config)

# from aura.aura import find_best_retriever, evaluate_llms, run_classifier_training_pipeline, run_aura_pipeline, run_routing_and_evaluation

# aura_config = {
#     "documents": "/future/u/manihani/ARES/datasets/eval_datasets/hotpotqa/hotpotqa_ratio_0.6.tsv",
#     "queries": "/future/u/manihani/ARES/datasets/eval_datasets/hotpotqa/hotpotqa_ratio_0.6.tsv",
#     "few_shot_examples_file_path": "/future/u/manihani/ARES/datasets/few_shot_datasets/judge_scoring/hotpotqa_few_shot_prompt_for_judge_scoring.tsv",
#     "rag_type": "question_answering",
#     "checkpoints": ["/future/u/manihani/ARES/checkpoints/microsoft-deberta-v3-large/Context_Relevance_Label_joint_datasets_2024-04-30_01:01:01.pt", "/future/u/manihani/ARES/checkpoints/microsoft-deberta-v3-large/Answer_Relevance_Label_joint_datasets_2024-04-30_01:01:01.pt"],
#     "gold_label_path": "/future/u/manihani/ARES/datasets/eval_datasets/hotpotqa/hotpotqa_ratio_0.6.tsv",
#     "LLM_prediction_folder_directory": "/future/u/manihani/AuRA-Folder/AuRA/hotpotqa_llm_predictions",
#     "classifier_training_data": "/future/u/manihani/AuRA-Folder/AuRA/classifier_training/hotpotqa_primary_classifier.tsv",
#     "model_dir": "/future/u/manihani/AuRA-Folder/AuRA/classifier_training_new",
#     "model_choice": "microsoft/deberta-v3-large",
#     "classifier_validation_data": "/future/u/manihani/ARES/datasets/eval_datasets/hotpotqa/hotpotqa_ratio_0.6.tsv",
#     "max_queries": 500,
#     "debug_mode": True,
#     "num_epochs": 10, 
#     "patience_value": 3, 
#     "learning_rate": 5e-6,
#     "assigned_batch_size": 1,  
#     "gradient_accumulation_multiplier": 32, 
#     "top_k": 1 
# }

# num_queries = 300

# run_routing_and_evaluation(aura_config, num_queries)

# run_aura_pipeline(aura_config)

from aura.aura import find_best_retriever, evaluate_llms, run_classifier_training_pipeline, run_aura_pipeline, run_routing_and_evaluation

aura_config = {
    "documents": "/future/u/manihani/ARES/datasets/eval_datasets/nq/nq_ratio_0.6.tsv",
    "queries": "/future/u/manihani/ARES/datasets/eval_datasets/nq/nq_ratio_0.6.tsv",
    "few_shot_examples_file_path": "/future/u/manihani/ARES/datasets/few_shot_datasets/judge_scoring/nq_few_shot_prompt_for_judge_scoring.tsv",
    "rag_type": "question_answering",
    "checkpoints": ["/future/u/manihani/ARES/checkpoints/microsoft-deberta-v3-large/Context_Relevance_Label_joint_datasets_2024-04-30_01:01:01.pt", "/future/u/manihani/ARES/training_classifier/checkpoints/microsoft-deberta-v3-large/Answer_Relevance_Label_nq_ratio_0.6_2024-06-06_22:25:01.pt"],
    "gold_label_path": "/future/u/manihani/ARES/datasets/eval_datasets/nq/nq_ratio_0.6.tsv",
    "LLM_prediction_folder_directory": "/future/u/manihani/AuRA-Folder/AuRA/nq_llm_predictions",
    "classifier_training_data": "/future/u/manihani/AuRA-Folder/AuRA/classifier_training/nq_primary_classifier.tsv",
    "model_dir": "/future/u/manihani/AuRA-Folder/AuRA/nq_classifier_training",
    "model_choice": "microsoft/deberta-v3-large",
    "classifier_validation_data": "/future/u/manihani/ARES/datasets/eval_datasets/nq/nq_ratio_0.6.tsv",
    "top_k": 1,
    "max_queries": 500, # OPTIONAL
    "num_epochs": 10, # OPTIONAL
    "patience_value": 3, # OPTIONAL
    "learning_rate": 5e-6, # OPTIONAL
    "assigned_batch_size": 1, # OPTIONAL
    "gradient_accumulation_multiplier": 32, # OPTIONAL
}

num_queries = 200

run_routing_and_evaluation(aura_config, num_queries)

# run_aura_pipeline(aura_config)