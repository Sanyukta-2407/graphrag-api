import json
import os
from typing import List, Dict

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="GraphRAG API")

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)


class ExtractRequest(BaseModel):
    chunk_id: str
    text: str


class GraphQueryRequest(BaseModel):
    question: str
    graph: Dict


class CommunityRequest(BaseModel):
    community_id: str
    entities: List[str]
    relationships: List[Dict]


@app.get("/")
def root():
    return {"status": "ok", "service": "GraphRAG API"}


@app.post("/extract-graph")
def extract_graph(req: ExtractRequest):
    prompt = f"""
Extract a knowledge graph from the following text.

Allowed entity types:
- Person
- Organization
- Product
- Framework

Allowed relationship types:
- FOUNDED
- DEVELOPED
- INTEGRATED_INTO
- HIRED
- AUTHORED
- CREATED

Return ONLY valid JSON in exactly this format:

{{
  "entities": [
    {{
      "name": "",
      "type": ""
    }}
  ],
  "relationships": [
    {{
      "source": "",
      "target": "",
      "relation": ""
    }}
  ]
}}

Text:
{req.text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are an expert knowledge graph extractor."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return json.loads(response.choices[0].message.content)


@app.post("/graph-query")
def graph_query(req: GraphQueryRequest):
    prompt = f"""
You are a GraphRAG reasoning engine.

Question:
{req.question}

Knowledge Graph:
{json.dumps(req.graph)}

Return ONLY valid JSON.

{{
  "answer": "",
  "reasoning_path": [],
  "hops": 0
}}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Answer questions using multi-hop reasoning over the graph."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return json.loads(response.choices[0].message.content)


@app.post("/community-summary")
def community_summary(req: CommunityRequest):
    prompt = f"""
Summarize this graph community.

Entities:
{json.dumps(req.entities)}

Relationships:
{json.dumps(req.relationships)}

Return ONLY valid JSON.

{{
  "community_id": "{req.community_id}",
  "summary": ""
}}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Summarize graph communities."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = json.loads(response.choices[0].message.content)
    result["community_id"] = req.community_id
    return result