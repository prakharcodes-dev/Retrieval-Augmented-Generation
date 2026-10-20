import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import dotenv_values
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_yaml_config(config_path: Path = Path("config.yaml")) -> Dict[str, Any]:
    """Loads configuration parameters from config.yaml if present."""
    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load {config_path}: {e}")
    return {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    HF_TOKEN: str = ""

    # Provider Options
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-3.8-flash"
    GROQ_FALLBACK_MODEL: str = "qwen/qwen3.8-27b"
    LLM_TEMPERATURE: float = 0.0

    LOCAL_MODEL_PATH: str = ""
    BASE_MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"
    CPU_FALLBACK_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"

    EMBEDDING_PROVIDER: str = "auto"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Chunking & Retrieval & Validation Parameters
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 4
    MAX_FILE_SIZE_MB: int = 25
    MAX_PDF_PAGES: int = 150

    # Vectorstore & Paths
    VECTOR_STORE_PROVIDER: str = "chroma"
    CHROMA_PERSIST_DIR: str = "./vectorstore"
    COLLECTION_NAME: str = "rag_documents"
    PROMPT_PATH: str = "./prompts/rag_answer_v1.txt"

    def __init__(self, **values: Any):
        super().__init__(**values)
        # Apply config.yaml overrides if available
        yaml_cfg = load_yaml_config()
        if "chunking" in yaml_cfg:
            self.CHUNK_SIZE = yaml_cfg["chunking"].get("chunk_size", self.CHUNK_SIZE)
            self.CHUNK_OVERLAP = yaml_cfg["chunking"].get("chunk_overlap", self.CHUNK_OVERLAP)
        if "retrieval" in yaml_cfg:
            self.TOP_K = yaml_cfg["retrieval"].get("vector_top_k", self.TOP_K)
        if "vector_store" in yaml_cfg:
            self.VECTOR_STORE_PROVIDER = yaml_cfg["vector_store"].get("provider", self.VECTOR_STORE_PROVIDER)
            self.CHROMA_PERSIST_DIR = yaml_cfg["vector_store"].get("persist_dir", self.CHROMA_PERSIST_DIR)
            self.COLLECTION_NAME = yaml_cfg["vector_store"].get("collection_name", self.COLLECTION_NAME)
        if "embedding" in yaml_cfg:
            self.EMBEDDING_PROVIDER = yaml_cfg["embedding"].get("provider", self.EMBEDDING_PROVIDER)
            self.EMBEDDING_MODEL = yaml_cfg["embedding"].get("model", self.EMBEDDING_MODEL)
        if "llm" in yaml_cfg:
            self.LLM_PROVIDER = yaml_cfg["llm"].get("provider", self.LLM_PROVIDER)
            self.LLM_MODEL = yaml_cfg["llm"].get("model", self.LLM_MODEL)
            self.GROQ_FALLBACK_MODEL = yaml_cfg["llm"].get("groq_fallback_model", self.GROQ_FALLBACK_MODEL)
            self.LLM_TEMPERATURE = yaml_cfg["llm"].get("temperature", self.LLM_TEMPERATURE)
            if "adapter_path" in yaml_cfg["llm"]:
                adapter_p = yaml_cfg["llm"]["adapter_path"]
                if adapter_p and os.path.exists(adapter_p):
                    self.LOCAL_MODEL_PATH = adapter_p
                else:
                    self.LOCAL_MODEL_PATH = ""
            if "base_model" in yaml_cfg["llm"]:
                self.BASE_MODEL_NAME = yaml_cfg["llm"]["base_model"]

        # Apply local .env values, Streamlit secrets, then real environment
        # variables in increasing precedence. Only names are inspected here;
        # credentials are never logged or included in errors.
        config_keys = (
            "GEMINI_API_KEY", "GROQ_API_KEY",
            "LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "GROQ_FALLBACK_MODEL",
            "EMBEDDING_PROVIDER", "EMBEDDING_MODEL",
            "VECTOR_STORE_PROVIDER", "CHROMA_PERSIST_DIR", "COLLECTION_NAME",
        )
        local_env = dotenv_values(Path(".env"))
        for key in config_keys:
            value = local_env.get(key)
            if value is not None:
                setattr(self, key, value)

        # Streamlit secrets are optional during CLI use and local development.
        try:
            import streamlit as st
            for key in config_keys:
                if key in st.secrets:
                    setattr(self, key, st.secrets[key])
        except Exception:
            pass

        for key in config_keys:
            if key in os.environ:
                setattr(self, key, os.environ[key])

        # A legacy Streamlit secret or environment variable can override config.yaml.
        # Migrate the retired Gemini model so an old LLM_MODEL override cannot
        # silently keep the deployed app on gemini-2.5-flash.
        if (
            self.LLM_PROVIDER.strip().lower() == "gemini"
            and self.LLM_MODEL.strip() in {"gemini-2.5-flash", "models/gemini-2.5-flash"}
        ):
            self.LLM_MODEL = "gemini-3.8-flash"

    def get_prompt_template(self) -> str:
        prompt_file = Path(self.PROMPT_PATH)
        if not prompt_file.is_file():
            prompt_file = Path("./prompts/rag_prompt.txt")

        if prompt_file.is_file():
            return prompt_file.read_text(encoding="utf-8")

        return (
            "You are a strict, factual Q&A assistant. Answer strictly based on the context.\n"
            "If the answer cannot be found, respond with 'I couldn't find sufficient evidence in the provided documents to answer that question.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )


settings = Settings()
