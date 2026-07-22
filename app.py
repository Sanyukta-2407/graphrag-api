import os
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)
class ExtractRequest(BaseModel):
    chunk_id: str
    text: str


class GraphQueryRequest(BaseModel):
    question: str
    graph: dict


class CommunityRequest(BaseModel):
    community_id: str
    entities: list[str]
    relationships: list[dict]
@app.post("/extract-graph")
def extract_graph(request: ExtractRequest):
    prompt = f"""
Extract entities and relationships from the text.

Entity types:
- Person
- Organization
- Product
- Framework

Relationship types:
- FOUNDED
- DEVELOPED
- INTEGRATED_INTO
- HIRED
- AUTHORED
- CREATED

Return ONLY valid JSON in this format:

{{
  "entities": [
    {{"name":"...","type":"..."}}
  ],
  "relationships": [
    {{"source":"...","target":"...","relation":"..."}}
  ]
}}

Text:
{request.text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/graph-query")
def graph_query(request: GraphQueryRequest):
    prompt = f"""
You are given a knowledge graph.

Answer the question using ONLY the graph.

Return ONLY valid JSON in this format:

{{
  "answer": "...",
  "reasoning_path": ["node1", "node2"],
  "hops": 2
}}

Knowledge Graph:
{json.dumps(request.graph, indent=2)}

Question:
{request.question}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/community-summary")
def community_summary(request: CommunityRequest):
    prompt = f"""
You are given a graph community.

Write a concise summary using only the provided entities and relationships.

Return ONLY valid JSON in this format:

{{
  "community_id": "{request.community_id}",
  "summary": "..."
}}

Entities:
{json.dumps(request.entities, indent=2)}

Relationships:
{json.dumps(request.relationships, indent=2)}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        result = json.loads(response.choices[0].message.content)
        result["community_id"] = request.community_id
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))