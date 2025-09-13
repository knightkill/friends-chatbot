# rag_graph.py
# Minimal LangGraph RAG with Qdrant + Gemma 3 chat. Small comments only.

import os
from typing import TypedDict, List

# --- Vector DB (Qdrant) ---
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- LLM (Gemma 3 via HF) ---
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# --- Prompt + Graph ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END


# =========================
# Environment (set these)
# =========================
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("QDRANT_COLLECTION", "friends-rag")
EMBED_MODEL = os.getenv("EMBED_MODEL", "jinaai/jina-embeddings-v2-base-en")
GEMMA_ID = os.getenv("GEMMA_ID", "google/gemma-3-4b-it")  # use 1b/4b it-variants
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


# =========================
# Vector store (assumes you already ingested)
# If you want ingestion here, drop txt files into DATA_DIR.
# =========================
def get_vector_store() -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    # create collection if missing (no-op if exists)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            on_disk_payload=True,
        )
    emb = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    return QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=emb)


vdb = get_vector_store()


# =========================
# LLM: Gemma 3 chat
# =========================
tok = AutoTokenizer.from_pretrained(GEMMA_ID)
mdl = AutoModelForCausalLM.from_pretrained(GEMMA_ID, device_map="auto")

# Minimal template if model lacks one (270m cases)
if not getattr(tok, "chat_template", None):
    tok.chat_template = """{% for m in messages -%}
<start_of_turn>{{ m['role'] }}
{{ m['content'] }}
<end_of_turn>
{% endfor -%}
<start_of_turn>model
"""

# Stop at end-of-turn
try:
    EOT_ID = tok.convert_tokens_to_ids("<end_of_turn>")
except Exception:
    EOT_ID = tok.eos_token_id

gen_pipe = pipeline(
    "text-generation",
    model=mdl,
    tokenizer=tok,
    max_new_tokens=192,          # short replies
    do_sample=False,             # deterministic
    repetition_penalty=1.1,      # reduce loops
    return_full_text=False,      # don't echo prompt
    eos_token_id=[i for i in {EOT_ID, tok.eos_token_id} if i is not None],
)

hf_llm = HuggingFacePipeline(pipeline=gen_pipe)
llm = ChatHuggingFace(llm=hf_llm, tokenizer=tok)  # applies chat template


# =========================
# Prompt
# =========================
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer questions about the Friends TV series. Use CONTEXT. If not enough info, say 'I don't know'. Be brief."),
    ("user", "CONTEXT:\n{context}\n\nQUESTION: {question}"),
])

parser = StrOutputParser()


# =========================
# Graph state + nodes
# =========================
class State(TypedDict, total=False):
    question: str
    k: int
    context: str
    answer: str
    sources: List[str]


def retrieve(s: State) -> State:
    k = s.get("k", 6)
    # MMR search for variety
    docs = vdb.max_marginal_relevance_search(s["question"], k=k)
    # keep context small (helps small models)
    joined = "\n\n---\n\n".join(d.page_content for d in docs)
    s["context"] = joined[:6000]
    s["sources"] = [d.metadata.get("source", "") for d in docs]
    return s


def generate(s: State) -> State:
    chain = (prompt | llm | parser)
    s["answer"] = chain.invoke({"context": s.get("context", ""), "question": s.get("question", "")})
    return s


# =========================
# Compile graph
# =========================
g = StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("generate", generate)
g.set_entry_point("retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)

app = g.compile()  # <- exported for Streamlit


# =========================
# Optional helper for imperative calls
# =========================
def answer_question(question: str, k: int = 6) -> dict:
    out = app.invoke({"question": question, "k": k})
    return {"answer": out.get("answer", ""), "sources": out.get("sources", [])}


# =========================
# Quick CLI test
# =========================
if __name__ == "__main__":
    q = "What was the name of Rachel’s childhood dog?"
    res = answer_question(q, k=6)
    print(res["answer"])
    print(res["sources"][:3])