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
import os

load_dotenv()

client = QdrantClient(url=os.environ.get("QDRANT_URL"), api_key=os.environ.get("QDRANT_API_KEY"))
# Initialize vector database
emb = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
vdb = QdrantVectorStore(client=client, collection_name="friends-rag", embedding=emb)

class State(TypedDict, total=False):
  question: str
  k: int
  context: str
  answer: str
  sources: List[str]

MODEL_ID = "google/gemma-3-1b-it"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto"
)

gen_pipe = pipeline(
    "text-generation",
    model=mdl,
    tokenizer=tok,
    max_new_tokens=512,          # Increased for more complete answers
    do_sample=True,              # Enable sampling for more natural responses
    temperature=0.3,             # Low temperature for factual consistency
    top_p=0.9,                   # Nucleus sampling for better quality
    repetition_penalty=1.2,      # Slightly higher to avoid repetition
    return_full_text=False,
    pad_token_id=tok.eos_token_id  # Proper padding token
)
hf_llm = HuggingFacePipeline(pipeline=gen_pipe)
llm = ChatHuggingFace(llm=hf_llm, tokenizer=tok)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert on the TV series Friends (1994-2004) with comprehensive knowledge of all characters, episodes, storylines, and relationships.

Your task is to answer questions about Friends using the provided context. Follow these guidelines:

ACCURACY RULES:
- Only answer if the context contains relevant information
- If uncertain, say "I don't have enough information to answer that confidently"
- Distinguish between facts and speculation clearly
- Cite specific episodes/seasons when mentioned in context

RESPONSE FORMAT:
- Give direct, factual answers first
- Include character full names when relevant (e.g., "Rachel Green" not just "Rachel")
- For list questions, provide complete lists when available in context
- Add helpful context about relationships or episodes when it enhances understanding

CONFIDENCE LEVELS:
- High confidence: "Based on the show..." or "According to Friends..."
- Medium confidence: "From what I can find in the context..."
- Low confidence: "I don't have enough information to answer that confidently"

CHARACTER REFERENCE (use full names when appropriate):
- Rachel Green, Ross Geller, Monica Geller, Chandler Bing, Joey Tribbiani, Phoebe Buffay

EXAMPLES:
Q: What was the name of Rachel's sisters?
A: Based on the show, Rachel Green had two sisters: Jill Green and Amy Green. Jill appeared in Season 6 and was played by Reese Witherspoon, while Amy appeared in Season 9-10 and was played by Christina Applegate.

Q: How many times was Ross married?
A: According to Friends, Ross Geller was married three times throughout the series: first to Carol Willick, then to Emily Waltham, and finally to Rachel Green (though they got married twice if you count the Vegas wedding separately).

Q: What is Central Perk?
A: Central Perk is the coffee shop where the main characters frequently hang out throughout the series. It serves as one of the primary meeting places for the group."""),
    ("user", "Context from Friends episodes:\n{context}\n\nQuestion: {question}\n\nAnswer:")
])

def retrieve(s: State) -> State:
  k = s.get("k", 3)
  docs = vdb.max_marginal_relevance_search(s["question"], k=k)
  s["context"] = "\n\n---\n\n".join(d.page_content for d in docs)
  s["sources"] = [d.metadata.get("source","") for d in docs]
  return s

parser = StrOutputParser()

def generate(s: State) -> State:
  chain = (prompt | llm | parser)
  s["answer"] = chain.invoke({
      "context": s.get("context", ""),
      "question": s.get("question","")
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
        "question": "What was the name of Rachel's sisters?",
        "k": 3
    })
    print(out["answer"])
    print(out["sources"][:3])