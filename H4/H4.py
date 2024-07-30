# %%
from sentence_transformers import SentenceTransformer

import numpy as np
import requests
import pandas as pd
from tqdm import tqdm
from rouge import Rouge

# %%
github_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/data/results-gpt4o-mini.csv"

url = f"{github_url}?raw=1"
df = pd.read_csv(url)
df = df.iloc[:300]
# %%
model_name = "multi-qa-mpnet-base-dot-v1"
embedding_model = SentenceTransformer(model_name)
# %% Q1
print(embedding_model.encode(df.iloc[0].answer_llm)[0])


# %%
def compute_similarity(embedding_model, record):
    answer_orig = record["answer_orig"]
    answer_llm = record["answer_llm"]

    v_llm = embedding_model.encode(answer_llm)
    v_orig = embedding_model.encode(answer_orig)

    v_llm_norm = np.sqrt((v_llm * v_llm).sum())
    v_llm_normalized = v_llm / v_llm_norm

    v_orig_norm = np.sqrt((v_orig * v_orig).sum())
    v_orig_normalized = v_orig / v_orig_norm

    return v_llm.dot(v_orig), v_llm_normalized.dot(v_orig_normalized)


# %%
similarity = []
similarity_normalized = []
for record in tqdm(df.to_dict(orient="records")):
    sim, sim_normalized = compute_similarity(embedding_model, record)
    similarity.append(sim)
    similarity_normalized.append(sim_normalized)
# %%
df["similarity"] = similarity
df["similarity_normalized"] = similarity_normalized

# %% Q2
df["similarity"].describe()

# %% Q3
df["similarity_normalized"].describe()

# %% Q4
rouge_scorer = Rouge()
r = df.loc[df["document"] == "5170565b"]
score = rouge_scorer.get_scores(r["answer_llm"], r["answer_orig"])[0]
print(score["rouge-1"])


# %% Q5
rouge_f1_1 = score["rouge-1"]["f"]
rouge_f1_2 = score["rouge-2"]["f"]
rouge_f1_l = score["rouge-l"]["f"]
rouge_f1_avg = (rouge_f1_1 + rouge_f1_2 + rouge_f1_l) / 3
print(rouge_f1_avg)


# %% Q6
def get_rouge_2(r):
    return rouge_scorer.get_scores(r["answer_llm"], r["answer_orig"])[0]["rouge-2"]["f"]


scores_2 = df.apply(get_rouge_2, axis=1)
print(np.mean(scores_2))

# %%
