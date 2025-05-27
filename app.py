# app.py
# -------------------------------------------------------------
# Streamlit-based Applicant-to-Job ranking tool
# -------------------------------------------------------------
import os, json, pickle, textwrap, pathlib, typing as T
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import faiss
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from utils import batch_embed_and_index

load_dotenv()

# ---------- CONFIG ---------- #
RESUME_DIR = pathlib.Path("resumes")          # each file: *.json
INDEX_PATH  = pathlib.Path("resume_faiss.idx")
IDMAP_PATH  = pathlib.Path("resume_id_map.pkl")
EMBED_MODEL = "text-embedding-3-small"        # or any other embedding model
LLM_MODEL   = "gpt-4o-mini"             # model that supports structured outputs
TOP_K_DEFAULT = 10
DEFAULT_WEIGHTS = {
    "work_experience": 0.4,
    "skills": 0.3,
    "education": 0.2,
    "certifications_extracurricular": 0.1
}
# ---------------------------- #

client = OpenAI()  # relies on OPENAI_API_KEY env var


# ---------- Pydantic schema for structured output ---------- #
class Reasoning(BaseModel):
    work_experience: str
    skills: str
    education: str
    certifications_extracurricular: str


class Ratings(BaseModel):
    work_experience: int
    skills: int
    education: int
    certifications_extracurricular: int
    overall_match: float


class ScoreReport(BaseModel):
    reasoning: Reasoning
    ratings: Ratings
# ----------------------------------------------------------- #


@st.cache_data(persist=True)
def build_or_load_index() -> tuple[faiss.IndexFlatIP, list[str]]:
    """
    Returns:
        faiss index (Inner-Product, L2-normalized)  
        id_map:   list of résumé file paths, index-aligned with FAISS vectors
    """
    # load if already cached on disk
    if INDEX_PATH.exists() and IDMAP_PATH.exists():
        index = faiss.read_index(INDEX_PATH.as_posix())
        id_map = pickle.loads(IDMAP_PATH.read_bytes())
        return index, id_map

    # otherwise build from scratch
    resume_files = list(RESUME_DIR.glob("*.json"))
    if not resume_files:
        st.error(f"No résumé JSON files found in {RESUME_DIR.resolve()}")
        st.stop()

    texts, id_map = [], []
    for fp in resume_files:
        data = json.loads(fp.read_text("utf-8"))
        # very simple flatten → embed (tweak as needed)
        flattened = json.dumps(data, ensure_ascii=False)
        texts.append(flattened)
        id_map.append(fp.as_posix())

    # Process embeddings in batch
    progress_bar = st.progress(0.0, text="Generating embeddings...")
    index, id_map = batch_embed_and_index(texts, id_map, EMBED_MODEL, client)
    progress_bar.progress(1.0)
    progress_bar.empty()

    # persist
    faiss.write_index(index, INDEX_PATH.as_posix())
    IDMAP_PATH.write_bytes(pickle.dumps(id_map))

    return index, id_map


def embed_query(text: str):
    e = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    import numpy as np
    v = np.array(e, dtype="float32")[None, :]
    faiss.normalize_L2(v)
    return v


def llm_score(job_desc: str, resume_text: str) -> ScoreReport:
    """Call GPT with the scoring prompt, parse into ScoreReport."""
    system_prompt = textwrap.dedent("""
        You are a hiring-evaluation assistant.
        Your goal is to judge how well one applicant fits one job, using the evidence provided.
        Use only the information given in the "Job Description" and "Candidate Profile".
        Think step-by-step before deciding; do not invent facts.
    """)

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

    # Structured Outputs via .responses.parse
    resp = client.responses.parse(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        text_format=ScoreReport
    )
    return resp.output_parsed


def extract_name(resume_json: dict) -> str:
    """Return best-effort applicant name."""
    name_obj = resume_json.get("personal_infos", {}).get("name", {}) or {}
    parts = [name_obj.get("first_name"), name_obj.get("middle"), name_obj.get("last_name")]
    raw   = name_obj.get("raw_name")
    cleaned = " ".join([p for p in parts if p]) or (raw or "Unknown Name")
    return cleaned.title()


def process_candidate(path: str, score_sim: float, jd_input: str, weights: dict) -> dict:
    """Process a single candidate in parallel."""
    data = json.loads(pathlib.Path(path).read_text("utf-8"))
    candidate_name = extract_name(data)
    resume_text = json.dumps(data, ensure_ascii=False)
    score_report = llm_score(jd_input, resume_text)
    
    # Calculate weighted score
    weighted_score = (
        score_report.ratings.work_experience * weights["work_experience"] +
        score_report.ratings.skills * weights["skills"] +
        score_report.ratings.education * weights["education"] +
        score_report.ratings.certifications_extracurricular * weights["certifications_extracurricular"]
    )
    
    return {
        "name": candidate_name,
        "similarity": round(score_sim, 3),
        "overall_match": score_report.ratings.overall_match,
        "weighted_score": round(weighted_score, 2),
        "detail": score_report,
        "ai_summary": data.get("personal_infos", {}).get("ai_summary", [])
    }


# ------------------------  Streamlit UI  ------------------------ #
st.title("🔎 Mimo Prototype")

index, id_map = build_or_load_index()

with st.sidebar:
    st.markdown("### Search parameters")
    jd_input = st.text_area("Job Description", height=300, placeholder="Paste the full JD …")
    top_n    = st.number_input("Top-N candidates to score", min_value=1, max_value=50,
                               value=TOP_K_DEFAULT, step=1)
    
    st.markdown("### Score Weights")
    st.markdown("Adjust the importance of each scoring component:")
    weights = {
        "work_experience": st.slider("Work Experience", 0.0, 1.0, DEFAULT_WEIGHTS["work_experience"], 0.1),
        "skills": st.slider("Skills", 0.0, 1.0, DEFAULT_WEIGHTS["skills"], 0.1),
        "education": st.slider("Education", 0.0, 1.0, DEFAULT_WEIGHTS["education"], 0.1),
        "certifications_extracurricular": st.slider("Certs/Extra", 0.0, 1.0, DEFAULT_WEIGHTS["certifications_extracurricular"], 0.1)
    }
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    run_btn  = st.button("Search & Rank")

if run_btn:
    if not jd_input.strip():
        st.warning("Please paste a job description first.")
        st.stop()

    # ---- retrieve nearest candidates ---- #
    query_vec = embed_query(jd_input)
    D, I = index.search(query_vec, int(top_n))
    hits = [(id_map[idx], float(D[0][rank])) for rank, idx in enumerate(I[0])]

    results = []
    progress = st.progress(0.0, text="Scoring candidates …")
    
    # Process candidates in parallel batches of 10
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create future tasks
        future_to_candidate = {
            executor.submit(process_candidate, path, score_sim, jd_input, weights): (path, score_sim)
            for path, score_sim in hits
        }
        
        # Process completed tasks as they finish
        completed = 0
        for future in concurrent.futures.as_completed(future_to_candidate):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                progress.progress(completed / len(hits))
            except Exception as e:
                st.error(f"Error processing candidate: {str(e)}")

    # sort by weighted score (desc)
    results.sort(key=lambda r: r["weighted_score"], reverse=True)

    st.success(f"Found {len(results)} candidates. Sorted by weighted score ↓")

    # ---------- results table ---------- #
    import pandas as pd
    table_df = pd.DataFrame([
        {
            "Name": r["name"],
            "Weighted Score": r["weighted_score"],
            "Overall Match": r["overall_match"],
            "Similarity": r["similarity"],
            "Work Exp": r["detail"].ratings.work_experience,
            "Skills":   r["detail"].ratings.skills,
            "Education": r["detail"].ratings.education,
            "Certs/Extra": r["detail"].ratings.certifications_extracurricular,
            "AI Summary": r["ai_summary"][0] if r["ai_summary"] else "No summary available"
        }
        for r in results
    ])
    st.dataframe(table_df, hide_index=True, use_container_width=True)

    # ---------- expandable reasoning ---------- #
    st.markdown("---")
    for r in results:
        with st.expander(f"📝 {r['name']} — why these scores?"):
            st.write("**AI Summary**")
            st.write(r["ai_summary"][0] if r["ai_summary"] else "No summary available")
            st.write("**Ratings**")
            st.json(r["detail"].ratings.model_dump())
            st.write("**Reasoning**")
            st.json(r["detail"].reasoning.model_dump())
