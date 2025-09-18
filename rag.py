from langchain_google_genai import ChatGoogleGenerativeAI

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from langchain_core.prompts import ChatPromptTemplate

from typing import TypedDict, List
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
import os
import re
from collections import Counter
import math

import llm_providers

load_dotenv()

client = QdrantClient(url=os.environ.get("QDRANT_URL"), api_key=os.environ.get("QDRANT_API_KEY"))
emb = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
vdb = QdrantVectorStore(client=client, collection_name="friends-rag", embedding=emb)

class State(TypedDict, total=False):
  question: str
  k: int
  context: str
  answer: str
  sources: List[str]
  chat_history: List[tuple]  # [(role, message), ...]

llm = llm_providers.get_llm_provider().get_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly Friends expert who loves chatting about the show! You have deep knowledge of all characters, episodes, and storylines from Friends (1994-2004).

CHAT STYLE:
- Be conversational and enthusiastic about Friends
- Use natural, friendly language like you're talking to a fellow fan
- Feel free to add fun details or trivia when relevant
- Keep responses engaging but concise

ACCURACY:
- Only answer using the provided context
- If you're not sure, just say "I'm not certain about that one" or "I don't have that info handy"
- Mention episodes or seasons when they're in the context

EXAMPLES:
"Tell me about Rachel's sisters"
→ "Oh, Rachel had two sisters! There's Jill Green, who appeared in Season 6 (played by Reese Witherspoon), and Amy Green from Seasons 9-10 (Christina Applegate). Both visits were pretty chaotic for the group!"

"Who is Gunther?"
→ "Gunther is the manager of Central Perk! He's got that distinctive bleached blonde hair and has been harboring a major crush on Rachel for years. He's always there in the background serving coffee."

Remember: Keep it friendly and conversational, like chatting with a Friends fan!"""),
    ("user", "Context from Friends episodes:\n{context}\n\nChat History:\n{chat_history}\n\nCurrent Question: {question}\n\nAnswer:")
])

def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)"""
    return len(text) // 4

def format_chat_history(history: List[tuple], max_turns: int = 3) -> str:
    """Format recent chat history for context (limit to last few turns)"""
    if not history:
        return "No previous conversation."

    # Take last max_turns conversations
    recent_history = history[-max_turns*2:] if len(history) > max_turns*2 else history

    formatted = []
    for role, message in recent_history:
        if role == "user":
            formatted.append(f"Human: {message}")
        else:
            formatted.append(f"Assistant: {message}")

    return "\n".join(formatted) if formatted else "No previous conversation."

def format_context(docs: List, max_tokens: int = 2000) -> str:
    """Enhanced context formatting with metadata and token limits"""
    formatted_contexts = []
    total_tokens = 0

    for doc in docs:
        metadata = doc.metadata
        content = doc.page_content.strip()

        # Add source information if available
        source_info = ""
        if "episode" in metadata:
            source_info = f"[Episode {metadata['episode']}] "
        elif "season" in metadata:
            source_info = f"[Season {metadata['season']}] "
        elif "source" in metadata and metadata["source"]:
            source_info = f"[{metadata['source']}] "

        # Format context with source info
        formatted_content = f"{source_info}{content}"
        content_tokens = estimate_tokens(formatted_content)

        # Check token limits
        if total_tokens + content_tokens <= max_tokens:
            formatted_contexts.append(formatted_content)
            total_tokens += content_tokens
        else:
            # Truncate last document if it would exceed limit
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 100:  # Only add if meaningful content fits
                truncated_content = formatted_content[:remaining_tokens * 4]
                formatted_contexts.append(truncated_content + "...")
            break

    return "\n\n".join(formatted_contexts)

def retrieve(s: State) -> State:
    k = s.get("k", 3)

    try:
        all_docs = []

        # Dense search
        mmr_docs = vdb.max_marginal_relevance_search(s["question"], k=k)
        all_docs.extend(mmr_docs)

        # Sparse search
        sparse_docs = vdb.similarity_search(s["question"], k=k//2 + 1)
        all_docs.extend(sparse_docs)

        if not all_docs:
            s["context"] = "No relevant information found in the database."
            s["sources"] = []
            return s

        docs = all_docs[:k]

        s["context"] = format_context(docs, max_tokens=2000)
        s["sources"] = [d.metadata.get("source", "Unknown") for d in docs]

    except Exception as e:
        s["context"] = f"Error retrieving information: {str(e)}"
        s["sources"] = []

    return s

parser = StrOutputParser()

def generate(s: State) -> State:
  chain = (prompt | llm | parser)
  chat_history_formatted = format_chat_history(s.get("chat_history", []))
  s["answer"] = chain.invoke({
      "context": s.get("context", ""),
      "question": s.get("question", ""),
      "chat_history": chat_history_formatted
  })
  return s

g = StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("generate", generate)

g.set_entry_point("retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)

app = g.compile()

if __name__ == "__main__":
    out = app.invoke({
        "question": "Who is gunther?",
        "k": 3
    })
    print(out["answer"])
    print(out["sources"][:3])