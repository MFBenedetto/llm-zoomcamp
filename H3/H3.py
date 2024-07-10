# %%
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import pandas as pd
from tqdm import tqdm

# %%
model_name = "multi-qa-distilbert-cos-v1"
embedding_model = SentenceTransformer(model_name)

# %%
user_question = "I just discovered the course. Can I still join it?"

# %%
user_question_encoding = embedding_model.encode(user_question)
user_question_encoding[:5]

# %%
base_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main"
relative_url = "03-vector-search/eval/documents-with-ids.json"
docs_url = f"{base_url}/{relative_url}?raw=1"
docs_response = requests.get(docs_url)
documents = docs_response.json()

# %% Filter by course
filtered_documents = [
    d for d in documents if d["course"] == "machine-learning-zoomcamp"
]
print(len(filtered_documents))

# %%
embeddings = []
for d in filtered_documents:
    qa_text = f"{d['question']} {d['text']}"
    embeddings.append(embedding_model.encode(qa_text))

# %%
X = np.array(embeddings)

# %%
print(X.shape)

# %%
scores = X.dot(user_question_encoding)


# %%
class VectorSearchEngine:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [self.documents[i] for i in idx]


search_engine = VectorSearchEngine(documents=filtered_documents, embeddings=X)

# %%
base_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main"
relative_url = "03-vector-search/eval/ground-truth-data.csv"
ground_truth_url = f"{base_url}/{relative_url}?raw=1"

df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == "machine-learning-zoomcamp"]
ground_truth = df_ground_truth.to_dict(orient="records")


# %%
def get_hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q["document"]
        results = search_function.search(
            embedding_model.encode(q["question"]), num_results=5
        )
        relevance = [d["id"] == doc_id for d in results]
        relevance_total.append(relevance)

    return get_hit_rate(relevance_total)


hit_rate = evaluate(ground_truth, search_engine)
# %%
print(hit_rate)

# %%
from elasticsearch import Elasticsearch

es_client = Elasticsearch("http://localhost:9200")

index_settings = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine",
            },
            "text_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine",
            },
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}

index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)

# %%
for doc in tqdm(filtered_documents):
    es_client.index(index=index_name, document=doc)


# %%
def elastic_search_knn(field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {"term": {"course": course}},
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"],
    }

    es_results = es_client.search(index=index_name, body=search_query)

    result_docs = []

    for hit in es_results["hits"]["hits"]:
        result_docs.append(hit["_source"])

    return result_docs


elastic_search_knn(
    "question_vector", user_question_encoding, "machine-learning-zoomcamp"
)


# %%
def evaluate_elastic_search(ground_truth, elastic_search_knn):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q["document"]
        results = elastic_search_knn(
            "question_vector",
            embedding_model.encode(q["question"]),
            "machine-learning-zoomcamp",
        )
        relevance = [d["id"] == doc_id for d in results]
        relevance_total.append(relevance)

    return get_hit_rate(relevance_total)


hit_rate_elasticsearch = evaluate_elastic_search(ground_truth, elastic_search_knn)
# %%
print(hit_rate_elasticsearch)

# %%
