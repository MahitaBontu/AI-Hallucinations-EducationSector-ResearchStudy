import openai
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Set your API key
openai.api_key = "your-openai-api-key"  # Replace with your actual key

# Embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# HuggingFace QA pipelines
hf_models = {
    "bert": pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"),
    "distilbert": pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
}

# Factual & hallucinated datasets
factual_questions = [
    {"question": "What is the powerhouse of the cell?", "context": "The mitochondria is the powerhouse of the cell.", "answer": "The mitochondria."},
    {"question": "Who was the first President of the United States?", "context": "George Washington was the first President of the United States.", "answer": "George Washington."},
    {"question": "What is the capital of France?", "context": "The capital of France is Paris.", "answer": "Paris."}
]

hallucinated_prompts = [
    {"question": "What is the powerhouse of the cell?", "context": "The ribosome is the powerhouse of the cell.", "answer": "The ribosome."},
    {"question": "Who was the first President of the United States?", "context": "Abraham Lincoln was the first President of the United States.", "answer": "Abraham Lincoln."},
    {"question": "What is the capital of France?", "context": "The capital of France is Lyon.", "answer": "Lyon."}
]

def get_gpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message["content"]

def get_hf_response(pipeline, question, context):
    result = pipeline(question=question, context=context)
    return result['answer']

def cosine_sim(a, b):
    emb1 = embedder.encode(a, convert_to_tensor=True)
    emb2 = embedder.encode(b, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def evaluate_model(name, model_func):
    results = {"baseline_correct": 0, "post_correct": 0, "hallucination_agree": 0}
    total = len(factual_questions)

    # Baseline
    for qa in factual_questions:
        answer = model_func(qa["question"], qa["context"])
        if cosine_sim(answer, qa["answer"]) > 0.95:
            results["baseline_correct"] += 1

    # Hallucination
    for qa in hallucinated_prompts:
        answer = model_func(qa["question"], qa["context"])
        if cosine_sim(answer, qa["answer"]) > 0.95:
            results["hallucination_agree"] += 1

    # Post-exposure
    for qa in factual_questions:
        answer = model_func(qa["question"], qa["context"])
        if cosine_sim(answer, qa["answer"]) > 0.95:
            results["post_correct"] += 1

    results["baseline_accuracy"] = results["baseline_correct"] / total
    results["post_accuracy"] = results["post_correct"] / total
    results["hallucination_rate"] = results["hallucination_agree"] / len(hallucinated_prompts)
    results["fla_score"] = (
        results["baseline_accuracy"] - results["post_accuracy"]
    ) / (1 + results["hallucination_rate"])

    print(f"--- Results for {name} ---")
    print(f"Baseline Accuracy: {results['baseline_accuracy']*100:.2f}%")
    print(f"Post Exposure Accuracy: {results['post_accuracy']*100:.2f}%")
    print(f"Hallucination Agreement Rate: {results['hallucination_rate']*100:.2f}%")
    print(f"FLA Score: {results['fla_score']:.4f}\\n")
    return results

# Run for GPT-3.5
evaluate_model("GPT-3.5", lambda q, c: get_gpt_response(q))

# Run for HuggingFace models
for name, pipe in hf_models.items():
    evaluate_model(name, lambda q, c, p=pipe: get_hf_response(p, q, c))
