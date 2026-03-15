from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests


def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    return r.json()

def format_time(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02}:{s:02}"

df = joblib.load('chunks_with_embeddings.joblib')


incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0] 

# Find similarities of question_embedding with other embeddings
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
new_df = df.loc[max_indx]
new_df["start"] = new_df["start"].apply(format_time)
new_df["end"] = new_df["end"].apply(format_time)
new_df["text"] = new_df["text"].str[:400]

prompt = f'''I am teaching Data Analysis using Python in this course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["name", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course. Also convert time frame into minutes and seconds format in your answer.
'''

# with open("prompt.txt", "w") as f:
#     f.write(prompt)

# Get response from the model
response = inference(prompt)["response"]

with open("response.txt", "w") as f:
    f.write(response)