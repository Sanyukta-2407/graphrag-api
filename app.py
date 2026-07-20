import json
import os
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="GraphRAG API")

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openrouter/v1",
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
    return {
        "status": "ok",
        "service": "GraphRAG API"
    }


@app.post("/extract-graph")
def extract_graph(req: ExtractRequest):
    try:
        prompt = f"""
Extract entities and relationships from the text.

Allowed entity types:
Person
Organization
Product
Framework

Allowed relationship types:
FOUNDED
DEVELOPED
INTEGRATED_INTO
HIRED
AUTHORED
CREATED

Return ONLY valid JSON.

{{
  "entities":[
    {{
      "name":"",
      "type":""
    }}
  ],
  "relationships":[
    {{
      "source":"",
      "target":"",
      "relation":""
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
                    "content": "You extract knowledge graphs."
                },
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
def graph_query(req: GraphQueryRequest):
    try:
        prompt = f"""
You are a GraphRAG reasoning engine.

Question:

{req.question}

Knowledge Graph:

{json.dumps(req.graph)}

Return ONLY JSON.

{{
    "answer":"",
    "reasoning_path":[],
    "hops":0
}}
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Answer questions using graph reasoning."
                },
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
def community_summary(req: CommunityRequest):
    try:
        prompt = f"""
Summarize this graph community.

Entities:

{json.dumps(req.entities)}

Relationships:

{json.dumps(req.relationships)}

Return ONLY JSON.

{{
    "community_id":"{req.community_id}",
    "summary":""
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))