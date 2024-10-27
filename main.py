import random
import os
from openai import OpenAI
from fastapi import FastAPI, HTTPException
import requests
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of origins or "*" for open access
    allow_credentials=True,
    allow_methods=["*"],  # Specify methods or use "*" for all
    allow_headers=["*"],  # Specify headers or use "*" for all
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def citrus_query(query_vectors, index_name, top_k=25):
    url = os.environ.get("CITRUS_URL")
    headers = {
        "x-api-key": os.environ.get("CITRUS_KEY")
    }
    body = {
        "index_name": index_name,
        "query_vectors": query_vectors,
        "include": ["document"],
        "top_k": top_k
    }
    response = requests.post(url, json=body, headers=headers)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        return {"error": f"Failed with status code {response.status_code}", "message": response.text}

@app.get("/search")
def search(query: str):
    try:
        embedding = get_embedding(query)

        # Query both indexes
        # Query both indexes
        repped_sellers_results = citrus_query(embedding, index_name=os.environ.get("SELLER_INDEX"))
        dummy_sellers_results = citrus_query(embedding, index_name=os.environ.get("DUMMY_INDEX"))

        # Combine and shuffle results
        combined_results = repped_sellers_results + dummy_sellers_results

        return {"results": combined_results}
    except HTTPException as e:
        return {"error": "Request failed", "message": e.detail}
