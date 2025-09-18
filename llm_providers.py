from abc import ABC, abstractmethod
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

class LLMProvider(ABC):

    @abstractmethod
    def get_llm(self):
        pass

class HuggingFaceProvider(LLMProvider):
    def get_llm(self):
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
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            return_full_text=False,
            pad_token_id=tok.eos_token_id  # Proper padding token
        )
        hf_llm = HuggingFacePipeline(pipeline=gen_pipe)
        return ChatHuggingFace(llm=hf_llm, tokenizer=tok)

class AzureOpenAIProvider(LLMProvider):
    def get_llm(self):
        return AzureChatOpenAI(
            azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            max_tokens=None
        )

def get_llm_provider(provider_type=None):
    if provider_type is None:
        provider_type = os.getenv("LLM_PROVIDER", "huggingface")

    if provider_type == "azure_openai":
        return AzureOpenAIProvider()
    return HuggingFaceProvider()