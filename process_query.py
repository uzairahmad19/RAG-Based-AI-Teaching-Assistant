"""
Processes user query, retrieves relevant chunks, generates RAG answer using Ollama.

Usage: python process_query.py
Output: Prints answer and saves to response.txt.
"""

import logging
import numpy as np
import joblib
import requests

from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_FILE = "chunks_with_embeddings.joblib"
RESPONSE_OUTPUT_FILE = "response.txt"

# Ollama endpoints
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

# Models
EMBEDDING_MODEL = "bge-m3"
GENERATION_MODEL = "llama3.2"

# Retrieval settings
TOP_K_RESULTS = 5
MAX_CHUNK_TEXT_CHARS = 400

COURSE_SUBJECT = "Data Analysis using Python"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """
    Converts seconds to MM:SS format.
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"


def create_embedding(text_list: list) -> list:
    """
    Embeds text list using Ollama.
    """
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBEDDING_MODEL, "input": text_list},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def run_inference(prompt: str) -> str:
    """
    Sends prompt to LLaMA model via Ollama and returns response.
    """
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": GENERATION_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def retrieve_top_chunks(df, query_embedding: list, top_k: int) -> object:
    """
    Retrieves top-K chunks by cosine similarity to query embedding.
    """
    # Stack embeddings into matrix
    chunk_matrix = np.vstack(df["embedding"].values)
    query_vector = np.array(query_embedding).reshape(1, -1)

    # Compute similarities
    similarities = cosine_similarity(chunk_matrix, query_vector).flatten()

    # Get top indices
    top_indices = similarities.argsort()[::-1][:top_k]

    # Format results
    result_df = df.loc[top_indices].copy()
    result_df["start"] = result_df["start"].apply(format_timestamp)
    result_df["end"] = result_df["end"].apply(format_timestamp)
    result_df["text"] = result_df["text"].str[:MAX_CHUNK_TEXT_CHARS]

    return result_df


def build_prompt(query: str, context_df) -> str:
    """
    Builds RAG prompt with retrieved context and query.
    """
    context_json = context_df[["name", "number", "start", "end", "text"]].to_json(orient="records")

    prompt = f"""You are a helpful teaching assistant for a course on {COURSE_SUBJECT}.

Below are transcript segments from the course videos. Each entry includes:
- "name": the video title
- "number": the video episode number
- "start" / "end": the timestamp range (MM:SS format)
- "text": what is being discussed at that time

Transcript segments:
{context_json}

---------------------------------
Student's question: "{query}"

Instructions:
- Answer naturally and conversationally — do NOT mention the JSON format.
- Tell the student which video(s) cover their question and at what timestamp.
- Reference the video by its title and episode number for clarity.
- If the question is unrelated to the course content, politely say you can only
  answer questions about {COURSE_SUBJECT}.
- Use "minutes and seconds" format when citing timestamps (e.g. "at 5 minutes 12 seconds").
"""
    return prompt


def main():
    """
    Loads embeddings, processes query, retrieves context, generates RAG response.
    """
    # Load embeddings
    logger.info("Loading chunk embeddings from '%s'...", EMBEDDINGS_FILE)
    try:
        df = joblib.load(EMBEDDINGS_FILE)
    except FileNotFoundError:
        logger.error(
            "'%s' not found. Please run Stage 3 (read_chunks.py) first.",
            EMBEDDINGS_FILE
        )
        return

    logger.info("Loaded %d chunks.", len(df))

    # Get user query
    query = input("Ask a question about the course: ").strip()

    if not query:
        logger.warning("Empty query received. Exiting.")
        return

    # Embed query
    logger.info("Embedding query...")
    try:
        query_embedding = create_embedding([query])[0]
    except Exception as e:
        logger.error("Failed to embed query: %s", e)
        return

    # Retrieve top chunks
    logger.info("Retrieving top %d relevant chunks...", TOP_K_RESULTS)
    top_chunks = retrieve_top_chunks(df, query_embedding, TOP_K_RESULTS)
    logger.info("Retrieved chunks from videos: %s", top_chunks["name"].tolist())

    # Build prompt
    prompt = build_prompt(query, top_chunks)

    # Generate response
    logger.info("Generating response with '%s'...", GENERATION_MODEL)
    try:
        response = run_inference(prompt)
    except Exception as e:
        logger.error("LLM inference failed: %s", e)
        return

    # Output response
    print("\n" + "=" * 60)
    print("RESPONSE")
    print("=" * 60)
    print(response)
    print("=" * 60 + "\n")

    with open(RESPONSE_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(response)

    logger.info("Response saved to '%s'.", RESPONSE_OUTPUT_FILE)


if __name__ == "__main__":
    main()
