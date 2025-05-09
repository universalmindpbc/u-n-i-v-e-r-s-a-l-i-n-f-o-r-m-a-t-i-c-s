"""
backend_database.py   ·   UNIVERSAL INFORMATICS  ·  STEP 9 — SECURE DATA STORAGE
===============================================================================

This is the **minimum‑viable core** that every other module can call today.
– Natural‑language in ➜ structured call out.
– No external SDK required until you paste a key into AWS Secrets Manager.
– Sections are separated by double‑line banners for fast scrolling.

Later we’ll merge Claude’s deep implementation *into* these sections
without breaking the public surface.

──────────────────────────────────────────────────────────────────────────────
"""

# ── 1. STANDARD LIBS ────────────────────────────────────────────────────────
import json, logging, os
from datetime import datetime
from typing import Dict, Any

# ── 2. OPTIONAL AWS & FASTAPI IMPORTS (fail gracefully) ─────────────────────
try:
    import boto3
    _HAS_AWS = True
except ImportError:
    _HAS_AWS = False

try:
    from fastapi import FastAPI, APIRouter
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

logger = logging.getLogger("ui.backend_database")

# ── 3. CONFIG  ––  central map of secret IDs  (edit only this) ──────────────
SECRET_IDS = {
    "openai":   "ui/openai-api-key",            # OpenAI (o3) key
    # add more when you have them
}

# ── 4. SECRETS HELPER  ––  single gateway to AWS Secrets Manager ────────────
def get_secret(name: str) -> str:
    """
    Fetch `name` from AWS Secrets Manager or raise a helpful error.
    """
    if name not in SECRET_IDS:
        raise KeyError(f"No secret mapping for '{name}'")

    if not _HAS_AWS:
        raise RuntimeError("AWS SDK missing – install boto3 or run in Lambda")

    sm = boto3.client("secretsmanager")
    try:
        return sm.get_secret_value(SecretId=SECRET_IDS[name])["SecretString"]
    except sm.exceptions.ResourceNotFoundException:
        raise RuntimeError(
            f"Secret '{SECRET_IDS[name]}' not found. "
            "Create it in AWS Secrets Manager whenever you’re ready."
        )

# ── 5. PLACEHOLDER SERVICE CALLS  –– keep them tiny for now ─────────────────
async def call_openai(prompt: str) -> Dict[str, Any]:
    """
    Minimal OpenAI (o3) call – replace with openai.ChatCompletion later.
    """
    api_key = get_secret("openai")       # will raise friendly error if missing
    # TODO: real call with openai client
    return {"reply": f"(simulated) You asked: {prompt[:40]}...", "model": "o3"}

# Stub slots for future services
async def call_langgraph(*_, **__):   return {"notice": "LangGraph stub"}
async def call_langchain(*_, **__):   return {"notice": "LangChain stub"}
async def call_mcp(*_, **__):         return {"notice": "MCP stub"}
async def call_a2a(*_, **__):         return {"notice": "A2A stub"}

# ── 6. NATURAL‑LANGUAGE ROUTER (very first draft) ───────────────────────────
INTENT_TABLE = {
    "ask gpt":         call_openai,
    # add more keywords → functions here
}

async def process_request(sentence: str) -> Dict[str, Any]:
    """
    Convert a plain‑English sentence into a service call.

    Very simple: looks for keyword in INTENT_TABLE.
    Later we can swap this for a regex parser or GPT‑router.
    """
    for keyword, func in INTENT_TABLE.items():
        if keyword in sentence.lower():
            return await func(sentence)

    return {"error": "Sorry, I don’t recognise that request yet."}

# ── 7. AWS LAMBDA ENTRY POINT  –– one line for production ───────────────────
def lambda_handler(event, _ctx=None):
    """
    Lambda front door – expects {'command': 'plain english here'}
    """
    sentence = event.get("command", "")
    return asyncio.run(process_request(sentence))

# ── 8. FASTAPI WRAPPER (optional, runs only if FastAPI is installed) ────────
if _HAS_FASTAPI:
    app  = FastAPI(title="UI Secure Storage Core")
    api  = APIRouter()

    class NLRequest(BaseModel):
        command: str

    @api.post("/process")
    async def process(req: NLRequest):
        return await process_request(req.command)

    app.include_router(api)

# ── 9. CLI TEST HARNESS  –– python backend_database.py "ask gpt hello" ──────
if __name__ == "__main__":
    import sys, asyncio
    if len(sys.argv) < 2:
        print("Usage: python backend_database.py \"ask gpt tell me a joke\"")
        exit(1)
    print(asyncio.run(process_request(" ".join(sys.argv[1:]))))