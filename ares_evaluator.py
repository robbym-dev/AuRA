from ares import ARES 

ppi_config = { 
    "evaluation_datasets": ['/future/u/manihani/AuRA-Folder/AuRA/output-datasets/bm25_nq_0.6.tsv'], 
    "few_shot_examples_filepath": "/future/u/manihani/AuRA-Folder/AuRA/datasets/nq_few_shot_prompt_for_judge_scoring.tsv",
    "checkpoints": ["/future/u/manihani/AuRA-Folder/AuRA/checkpoint/ares_context_relevance_joint.pt"], 
    "rag_type": "question_answering", 
    "labels": ["Context_Relevance_Label"], 
    "gold_label_path": "/future/u/manihani/AuRA-Folder/AuRA/datasets/nq_labeled_output.tsv", 
}

ares = ARES(ppi=ppi_config)
results = ares.evaluate_RAG()
print(results)