# %%
import requests
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm
import tiktoken

# %%
docs_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1"
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course["course"]

    for doc in course["documents"]:
        doc["course"] = course_name
        documents.append(doc)
# %%
es_client = Elasticsearch("http://localhost:9200")
# %%
es_client.info()
# %%
settings = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
        }
    },
}

# %%
index_name = "llm_course"
es_client.indices.create(index=index_name, body=settings)

# %%
for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)

# %%
query = "How do I execute a command in a running docker container?"

# %%
search_query = {
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^4", "text"],
                    "type": "best_fields",
                }
            },
        }
    },
}

# %%
response = es_client.search(index=index_name, body=search_query)

# %%
response["hits"]

# %%
search_query_2 = {
    "size": 3,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^4", "text"],
                    "type": "best_fields",
                }
            },
            "filter": {"term": {"course": "machine-learning-zoomcamp"}},
        }
    },
}
# %%
response_2 = es_client.search(
    index=index_name,
    body=search_query_2,
)

# %%
response_2["hits"]

# %%
result_docs = []

for hit in response_2["hits"]["hits"]:
    result_docs.append(hit["_source"])

# %%
question = "How do I execute a command in a running docker container?"

context_template = """
Q: {question}
A: {text}
""".strip()

# %%
context = ""
for c in result_docs:
    context += context_template.format(question=question, text=c["text"])
    context += "\n\n"

# %%
print(context)

# %%
prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT:
    {context}
""".strip()

# %%
prompt = prompt_template.format(question=question, context=context)

# %%
len(prompt)

# %%
encoding = tiktoken.encoding_for_model("gpt-4o")

# %%
len(encoding.encode(prompt))

# %%
encoding.decode_single_token_bytes(63842)
