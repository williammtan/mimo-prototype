import json
import pathlib
from typing import List, Tuple, Dict, Any
import tempfile
from openai import OpenAI
import faiss
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def create_batch_input_file(texts: List[str], model: str) -> str:
    """Create a JSONL file for batch processing of embeddings.
    
    Args:
        texts: List of texts to embed
        model: Embedding model to use
        
    Returns:
        Path to the created JSONL file
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i, text in enumerate(texts):
            request = {
                "custom_id": f"embed-{i}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": model,
                    "input": text
                }
            }
            f.write(json.dumps(request) + '\n')
        return f.name

def process_batch_embeddings(texts: List[str], model: str, client: OpenAI) -> List[List[float]]:
    """Process embeddings in batch using OpenAI's Batch API.
    
    Args:
        texts: List of texts to embed
        model: Embedding model to use
        client: OpenAI client instance
        
    Returns:
        List of embedding vectors
    """
    # Create batch input file
    input_file_path = create_batch_input_file(texts, model)
    
    try:
        # Upload the batch input file
        with open(input_file_path, 'rb') as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )
        
        # Create and start the batch
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h"
        )
        
        # Wait for batch completion
        while True:
            batch = client.batches.retrieve(batch.id)
            if batch.status == "completed":
                break
            elif batch.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Batch processing failed with status: {batch.status}")
        
        # Get results
        output_file = client.files.content(batch.output_file_id)
        results = []
        
        # Parse results and maintain order using custom_id
        for line in output_file.text.splitlines():
            result = json.loads(line)
            if result.get("error"):
                raise Exception(f"Error in batch processing: {result['error']}")
            
            # Extract embedding from response
            embedding = result["response"]["body"]["data"][0]["embedding"]
            results.append(embedding)
        
        return results
        
    finally:
        # Cleanup
        pathlib.Path(input_file_path).unlink(missing_ok=True)
        if 'batch_input_file' in locals():
            client.files.delete(batch_input_file.id)

def batch_embed_and_index(texts: List[str], id_map: List[str], model: str, client: OpenAI) -> Tuple[faiss.IndexFlatIP, List[str]]:
    """Process embeddings in batch and create FAISS index.
    
    Args:
        texts: List of texts to embed
        id_map: List of IDs corresponding to texts
        model: Embedding model to use
        client: OpenAI client instance
        
    Returns:
        Tuple of (FAISS index, id_map)
    """
    # Get embeddings in batch
    embeds = process_batch_embeddings(texts, model, client)
    
    # Convert to numpy array and normalize
    X = np.vstack(embeds).astype("float32")
    faiss.normalize_L2(X)
    
    # Create and populate FAISS index
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    
    return index, id_map 

def create_llm_batch_input_file(job_desc: str, resume_texts: List[str], model: str) -> str:
    """Create a JSONL file for batch processing of LLM scoring.
    
    Args:
        job_desc: Job description text
        resume_texts: List of resume texts to score
        model: LLM model to use
        
    Returns:
        Path to the created JSONL file
    """
    system_prompt = """You are a hiring-evaluation assistant.
Your goal is to judge how well one applicant fits one job, using the evidence provided.
Use only the information given in the "Job Description" and "Candidate Profile".
Think step-by-step before deciding; do not invent facts."""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i, resume_text in enumerate(resume_texts):
            user_prompt = f"""
### Job Description ###
{job_desc}

### Candidate Profile ###
{resume_text}

### INSTRUCTIONS ###
1. Under "reasoning", explain (2-5 sentences each) why you gave the score for:
   • work_experience • skills • education • certifications_extracurricular
2. Then output numeric 1-5 ratings for each, plus overall_match = mean.
"""
            request = {
                "custom_id": f"score-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "response_format": {"type": "json_object"}
                }
            }
            f.write(json.dumps(request) + '\n')
        return f.name

def process_llm_batch(job_desc: str, resume_texts: List[str], model: str, client: OpenAI) -> List[Dict[str, Any]]:
    """Process LLM scoring in batch using OpenAI's Batch API.
    
    Args:
        job_desc: Job description text
        resume_texts: List of resume texts to score
        model: LLM model to use
        client: OpenAI client instance
        
    Returns:
        List of score reports
    """
    input_file_path = create_llm_batch_input_file(job_desc, resume_texts, model)
    
    try:
        # Upload the batch input file
        with open(input_file_path, 'rb') as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )
        
        # Create and start the batch
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        # Wait for batch completion
        while True:
            batch = client.batches.retrieve(batch.id)
            if batch.status == "completed":
                break
            elif batch.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Batch processing failed with status: {batch.status}")
        
        # Get results
        output_file = client.files.content(batch.output_file_id)
        results = []
        
        # Parse results and maintain order using custom_id
        for line in output_file.text.splitlines():
            result = json.loads(line)
            if result.get("error"):
                raise Exception(f"Error in batch processing: {result['error']}")
            
            # Extract score report from response
            score_report = json.loads(result["response"]["body"]["choices"][0]["message"]["content"])
            results.append(score_report)
        
        return results
        
    finally:
        # Cleanup
        pathlib.Path(input_file_path).unlink(missing_ok=True)
        if 'batch_input_file' in locals():
            client.files.delete(batch_input_file.id)

def process_resumes_in_parallel(job_desc: str, hits: List[Tuple[str, float]], model: str, client: OpenAI, batch_size: int = 10) -> List[Dict[str, Any]]:
    """Process resume scoring in parallel batches.
    
    Args:
        job_desc: Job description text
        hits: List of (resume_path, similarity_score) tuples
        model: LLM model to use
        client: OpenAI client instance
        batch_size: Number of resumes to process in each batch
        
    Returns:
        List of results with scoring information
    """
    results = []
    
    # Process in batches
    for i in range(0, len(hits), batch_size):
        batch_hits = hits[i:i + batch_size]
        
        # Prepare batch data
        batch_texts = []
        for path, _ in batch_hits:
            data = json.loads(pathlib.Path(path).read_text("utf-8"))
            resume_text = json.dumps(data, ensure_ascii=False)
            batch_texts.append(resume_text)
        
        # Process batch
        batch_scores = process_llm_batch(job_desc, batch_texts, model, client)
        
        # Combine results
        for (path, score_sim), score_report in zip(batch_hits, batch_scores):
            data = json.loads(pathlib.Path(path).read_text("utf-8"))
            candidate_name = extract_name(data)
            
            results.append({
                "name": candidate_name,
                "similarity": round(score_sim, 3),
                "overall_match": score_report["ratings"]["overall_match"],
                "detail": score_report,
            })
    
    return results 