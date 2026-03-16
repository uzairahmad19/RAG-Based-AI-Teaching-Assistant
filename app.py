from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

EMBEDDINGS_FILE = "chunks_with_embeddings.joblib"

OLLAMA_EMBED_URL   = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"


COURSE_SUBJECT = "Data Analysis using Python"

df = joblib.load(EMBEDDINGS_FILE)



def format_timestamp(sec):
    return f"{int(sec//60):02}:{int(sec%60):02}"

def embed(text):
    r = requests.post(OLLAMA_EMBED_URL, json={"model": "bge-m3", "input": [text]})
    return r.json()["embeddings"][0]

def retrieve(query_vec, top_k=5):
    matrix = np.vstack(df["embedding"].values)
    sims = cosine_similarity(matrix, [query_vec]).flatten()
    idx = sims.argsort()[::-1][:top_k]
    result = df.loc[idx].copy()
    result["similarity"] = sims[idx].tolist()
    result["start"] = result["start"].apply(format_timestamp)
    result["end"]   = result["end"].apply(format_timestamp)
    result["text"]  = result["text"].str[:400] # limit text length for output
    return result

def generate(prompt):
    r = requests.post(OLLAMA_GENERATE_URL, json={
        "model": "llama3.2", "prompt": prompt, "stream": False
    })
    return r.json()["response"]

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    query_vec   = embed(question)
    top_chunks  = retrieve(query_vec)

    context = top_chunks[["name","number","start","end","text"]].to_json(orient="records")
    prompt = f"""You are a teaching assistant for a course on {COURSE_SUBJECT}.
Here are transcript chunks from lecture videos (name, number, start, end, text):
{context}
---
Question: "{question}"
Answer naturally, mention which video and timestamp covers this topic. Use "minutes and seconds" format when citing timestamps (e.g. "at 5 minutes 12 seconds").If the answer is not in the provided chunks, say "Sorry, I don't know". Be concise and to the point"""

    answer = generate(prompt)

    # # Format chunks for output
    # chunks_out = top_chunks[["name","number","start","end","text","similarity"]].to_dict(orient="records")
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5000, debug=True)