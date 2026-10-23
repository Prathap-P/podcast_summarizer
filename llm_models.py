from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _create_groq_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-20b"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_completion_tokens=65000,
    )

gemma_local_llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="test",
    temperature=0.7,
    model=os.getenv("GEMMA_MODEL_ID", "google/gemma-3-27b")
)

nemotron_local_llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="test",
    temperature=0.3,
    model=os.getenv("NEMOTRON_MODEL_ID", "nvidia/nemotron-3-nano"),
    top_p= 0.70,
    max_completion_tokens= 10000,
    frequency_penalty=1.3,
    presence_penalty=0.3,
    timeout= 3600
)

nemotron_stream_local_llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="test",
    temperature=0.7,
    model=os.getenv("NEMOTRON_STREAM_MODEL_ID", "nvidia/nemotron-3-nano"),
    top_p= 0.85,
    max_completion_tokens= 15000,
    frequency_penalty=1,
    presence_penalty=0.5,
    extra_body={
        "min_p": 0.05,
        "repeat_penalty": 1.1
    },
    # streaming=True,
    # stream_usage=True,
    timeout= 3600
)

nexveridian_qwen_stream_local_llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="test",
    temperature=0.7,
    model=os.getenv("NEXVERIDIAN_QWEN_MODEL_ID", "nexveridian/qwen3.5-35b-a3b"),
    top_p=0.85,
    max_completion_tokens=20000,
    frequency_penalty=0.8,
    presence_penalty=0.6,
    extra_body={
        "min_p": 0.05,
        "repeat_penalty": 1.15
    },
    # streaming=True,
    # stream_usage=True,
    timeout=3600
)

mlx_community_qwen_stream_local_llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="test",
    temperature=0.6,
    model=os.getenv("MLX_QWEN_MODEL_ID", "mlx-community/Local-3.5-35B-A3B-4bit"),
    top_p=0.9,
    max_completion_tokens=15000,
    frequency_penalty=0.3,
    presence_penalty=0.2,
    extra_body={
        "min_p": 0,
        "repeat_penalty": 1.15
    },
    # streaming=True,
    # stream_usage=True,
    timeout=3600
)

google_gemma_4_26b_a4b = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="test",
    temperature=0.6,
    model=os.getenv("GEMMA_4_MODEL_ID", "google/gemma-4-26b-a4b"),
    top_p=0.9,
    max_completion_tokens=15000,
    frequency_penalty=0.3,
    presence_penalty=0.2,
    extra_body={
        "min_p": 0,
        "repeat_penalty": 1.15
    },
    # streaming=True,
    # stream_usage=True,
    timeout=3600
)

deepseekR1_local_llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="test",
    temperature=0.7,
    model=os.getenv("DEEPSEEK_MODEL_ID", "deepseek/deepseek-r1-0528-qwen3-8b")
)

gpt_oss_20b_local_llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="lm-studio",
    model=os.getenv("GPT_OSS_MODEL_ID", "openai/gpt-oss-20b"),
    extra_body={"reasoning_effort": "high"},
    max_completion_tokens=12800,
    temperature=0.5,
    # streaming=True,
)

mistral_local_llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="test",
    temperature=0.7,
    model=os.getenv("MISTRAL_MODEL_ID", "mlx-community/Mistral-7B-Instruct-v0.3-4bit"),
    top_p=0.85,
    max_completion_tokens=15000,
    frequency_penalty=1,
    presence_penalty=0.5,
    extra_body={
        "min_p": 0.05,
        "repeat_penalty": 1.1
    },
    # streaming=True,
    # stream_usage=True,
    timeout=3600
)


models_collection = {
    "gemma_local_llm": gemma_local_llm,
    "nemotron_local_llm": nemotron_local_llm,
    "nemotron_stream_local_llm": nemotron_stream_local_llm,
    "nexveridian_qwen_stream_local_llm": nexveridian_qwen_stream_local_llm,
    "mlx_community_qwen_stream_local_llm": mlx_community_qwen_stream_local_llm,
    "google_gemma_4_26b_a4b": google_gemma_4_26b_a4b,
    "deepseekR1_local_llm": deepseekR1_local_llm,
    "gpt-oss_20b_local_llm": gpt_oss_20b_local_llm,
    "mistral_local_llm": mistral_local_llm
}

def get_model(model_name):
    if model_name == "groq_llm":
        return _create_groq_llm()
    if model_name in models_collection:
        return models_collection[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")
